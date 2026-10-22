//! Implicit physical drive rows, coupled to the mechanism's other constraints.
use super::constraint_rows::ConstraintRow;
use crate::{dynamics::solver::xpbd::ConstraintSolveError, prelude::*};

pub(crate) struct MotorInput {
    pub model: MotorModel,
    pub coordinate: f32,
    pub velocity: f32,
    pub target_position: f32,
    pub target_velocity: f32,
    pub max_force: f32,
    pub angular: bool,
    pub jacobian: ConstraintRow,
}
impl MotorInput {
    pub(crate) fn normalized(&self) -> bool {
        !matches!(self.model, MotorModel::ForceBased { .. })
    }
    pub(crate) fn prepare(
        &self,
        inverse_mass: f64,
        dt: f32,
    ) -> Result<Option<MotorConstraint>, ConstraintSolveError> {
        super::super::validate_motor_parameters(
            self.model,
            self.max_force,
            self.target_position,
            self.target_velocity,
        )?;
        if !inverse_mass.is_finite() || inverse_mass < 0.0 {
            return Err(ConstraintSolveError::InvalidEffectiveMass);
        }
        if !dt.is_finite() || dt < 0.0 || !self.coordinate.is_finite() || !self.velocity.is_finite()
        {
            return Err(ConstraintSolveError::NonFiniteSystem);
        }
        if self.max_force == 0.0 || dt == 0.0 || (self.normalized() && inverse_mass == 0.0) {
            return Ok(None);
        }
        let (k, c) = super::super::motor_gains(self.model, inverse_mass);
        let h = f64::from(dt);
        let denominator = k * h + c;
        if denominator == 0.0 {
            return Ok(None);
        }
        let position_error = if self.angular {
            wrap_angle(f64::from(self.coordinate) - f64::from(self.target_position))
        } else {
            f64::from(self.coordinate) - f64::from(self.target_position)
        };
        let error = h
            * (k * position_error
                + c * (f64::from(self.velocity) - f64::from(self.target_velocity)))
            / denominator;
        Ok(Some(MotorConstraint {
            initial_coordinate: f64::from(self.coordinate),
            error,
            compliance: h / denominator,
            cap: f64::from(self.max_force) * h * h,
            angular: self.angular,
            error_scale: self
                .jacobian
                .error_scale
                .max((h * f64::from(self.velocity)).abs()),
        }))
    }
}

pub(crate) struct MotorConstraint {
    initial_coordinate: f64,
    error: f64,
    compliance: f64,
    cap: f64,
    angular: bool,
    error_scale: f64,
}
impl MotorConstraint {
    pub(crate) fn row(&self, input: MotorInput) -> ConstraintRow {
        let delta = f64::from(input.coordinate) - self.initial_coordinate;
        let delta = if self.angular {
            wrap_angle(delta)
        } else {
            delta
        };
        let mut row = input.jacobian;
        row.error = self.error + delta;
        row.compliance = self.compliance;
        row.bounds = [-self.cap, self.cap];
        row.error_scale = self.error_scale.max(row.error_scale);
        row
    }
}
fn wrap_angle(value: f64) -> f64 {
    (value + core::f64::consts::PI).rem_euclid(core::f64::consts::TAU) - core::f64::consts::PI
}

pub(crate) fn prepare_pair_motor(
    input: Option<&MotorInput>,
    rows: &[ConstraintRow],
    inertias: [&crate::dynamics::solver::solver_body::SolverBodyInertia; 2],
    dt: f32,
) -> Result<Option<MotorConstraint>, ConstraintSolveError> {
    let Some(input) = input else {
        return Ok(None);
    };
    let inverse_mass = if input.normalized() {
        super::constraint_rows::pair_projected_mass(rows, &input.jacobian, inertias, dt)?
    } else {
        super::constraint_rows::pair_inverse_mass(&input.jacobian, &input.jacobian, inertias)
    };
    input.prepare(inverse_mass, dt)
}
pub(crate) fn rows_with_motor(
    frames: [ConstraintRow; 6],
    motor: Option<&MotorConstraint>,
    input: Option<MotorInput>,
) -> [ConstraintRow; 7] {
    let mut frames = frames.into_iter();
    let mut motor_row = Some(match motor {
        Some(motor) => {
            motor.row(input.expect("prepared motor configuration is immutable during a solve"))
        }
        None => ConstraintRow::default(),
    });
    core::array::from_fn(|index| {
        if index < 6 {
            frames.next().unwrap()
        } else {
            motor_row.take().unwrap()
        }
    })
}
