//! XPBD joint constraints.

mod shared;
pub use shared::{FixedAngleConstraintShared, PointConstraintShared};

mod distance;
mod fixed;
mod prismatic;
mod revolute;
#[cfg(feature = "3d")]
mod spherical;

pub use distance::DistanceJointSolverData;
pub use fixed::FixedJointSolverData;
pub use prismatic::PrismaticJointSolverData;
pub use revolute::RevoluteJointSolverData;
#[cfg(feature = "3d")]
pub use spherical::SphericalJointSolverData;

/// Positional multiplier for an implicit spring-damper impulse at an already
/// integrated coordinate. In XPBD, physical impulse is this multiplier / dt.
fn motor_lagrange(
    model: crate::prelude::MotorModel,
    position_error: f32,
    velocity_error: f32,
    inverse_mass: f32,
    max_force: f32,
    dt: f32,
) -> f32 {
    use crate::prelude::MotorModel;
    if inverse_mass == 0.0 || dt == 0.0 || max_force == 0.0 {
        return 0.0;
    }
    // Evaluate coefficients in f64 so finite f32 gains do not overflow while
    // forming the implicit denominator. Solver state and the result remain f32.
    let h = f64::from(dt);
    let w = f64::from(inverse_mass);
    let (stiffness, damping, mobility) = match model {
        MotorModel::ForceBased { stiffness, damping } => {
            (f64::from(stiffness), f64::from(damping), w)
        }
        MotorModel::AccelerationBased { stiffness, damping } => {
            (f64::from(stiffness), f64::from(damping), 1.0)
        }
        MotorModel::SpringDamper {
            frequency,
            damping_ratio,
        } => {
            let omega = core::f64::consts::TAU * f64::from(frequency);
            (omega * omega, 2.0 * f64::from(damping_ratio) * omega, 1.0)
        }
    };
    let acceleration = mobility
        * (stiffness * f64::from(position_error) + damping * f64::from(velocity_error))
        / (1.0 + mobility * h * (damping + stiffness * h));
    let force = acceleration / w;
    (force.clamp(-f64::from(max_force), f64::from(max_force)) * h * h) as f32
}

#[cfg(test)]
mod motor_tests {
    use super::motor_lagrange;
    use crate::prelude::MotorModel;

    #[test]
    fn motor_impulse_has_dimensional_timestep_and_mass_scaling() {
        let h = 0.002;
        for w in [0.1, 1.0, 1000.0] {
            for (k, c) in [(1000.0, 0.0), (0.0, 100.0), (1000.0, 100.0)] {
                let model = MotorModel::ForceBased {
                    stiffness: k,
                    damping: c,
                };
                let actual = motor_lagrange(model, 0.1, -0.25, w, f32::INFINITY, h);
                let expected = (k * 0.1 - c * 0.25) * h * h / (1.0 + w * h * (c + k * h));
                assert!((actual - expected).abs() <= expected.abs() * 2.0e-6 + 1.0e-12);
            }
        }
    }

    #[test]
    fn normalized_motor_models_share_the_same_implicit_response() {
        let frequency = 3.0;
        let damping_ratio = 0.7;
        let omega = core::f32::consts::TAU * frequency;
        let spring = MotorModel::SpringDamper {
            frequency,
            damping_ratio,
        };
        let acceleration = MotorModel::AccelerationBased {
            stiffness: omega * omega,
            damping: 2.0 * damping_ratio * omega,
        };
        for w in [0.1, 1.0, 1000.0] {
            let a = motor_lagrange(spring, 0.1, -0.25, w, f32::INFINITY, 0.002);
            let b = motor_lagrange(acceleration, 0.1, -0.25, w, f32::INFINITY, 0.002);
            assert!((a - b).abs() <= a.abs() * 2.0e-6);
        }
    }

    #[test]
    fn motor_force_cap_includes_zero_and_both_directions() {
        let model = MotorModel::ForceBased {
            stiffness: 1000.0,
            damping: 100.0,
        };
        for error in [-10.0, 10.0] {
            for cap in [0.0, 2.0] {
                let actual = motor_lagrange(model, error, 0.0, 1.0, cap, 0.002);
                let expected = error.signum() * cap * 0.002 * 0.002;
                assert!((actual - expected).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn motor_retains_small_physical_corrections_and_large_finite_gains() {
        for stiffness in [1.0e-6, f32::MAX] {
            let value = motor_lagrange(
                MotorModel::ForceBased {
                    stiffness,
                    damping: 0.0,
                },
                0.1,
                0.0,
                1000.0,
                f32::INFINITY,
                0.002,
            );
            assert!(value.is_finite() && value > 0.0);
        }
    }
}

#[cfg(feature = "3d")]
mod islands;
#[cfg(feature = "3d")]
pub(crate) use islands::solve_joint_contact_islands;
