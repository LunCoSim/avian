use super::PointConstraintShared;
use crate::{
    dynamics::solver::{
        solver_body::{SolverBody, SolverBodyInertia},
        xpbd::*,
    },
    prelude::*,
};
use bevy::prelude::*;

#[cfg(feature = "2d")]
use core::f32::consts::{PI, TAU};

/// Constraint data required by the XPBD constraint solver for a [`RevoluteJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct RevoluteJointSolverData {
    pub(super) point_constraint: PointConstraintShared,
    #[cfg(feature = "2d")]
    pub(super) rotation_difference: f32,
    #[cfg(feature = "3d")]
    pub(super) a1: Vector,
    #[cfg(feature = "3d")]
    pub(super) a2: Vector,
    #[cfg(feature = "3d")]
    pub(super) b1: Vector,
    #[cfg(feature = "3d")]
    pub(super) b2: Vector,
    pub(super) total_align_lagrange: AngularVector,
    pub(super) total_limit_lagrange: AngularVector,
    /// Accumulated motor Lagrange multiplier for this frame.
    pub(super) total_motor_lagrange: AngularVector,
}

impl XpbdConstraintSolverData for RevoluteJointSolverData {
    fn clear_lagrange_multipliers(&mut self) {
        self.point_constraint.clear_lagrange_multipliers();
        self.total_align_lagrange = AngularVector::default();
        self.total_limit_lagrange = AngularVector::default();
        self.total_motor_lagrange = AngularVector::default();
    }

    fn total_motor_lagrange(&self) -> f32 {
        #[cfg(feature = "2d")]
        {
            self.total_motor_lagrange
        }
        #[cfg(feature = "3d")]
        {
            self.total_motor_lagrange.length()
        }
    }

    fn total_position_lagrange(&self) -> Vector {
        self.point_constraint.total_position_lagrange()
    }

    fn total_rotation_lagrange(&self) -> AngularVector {
        self.total_align_lagrange + self.total_limit_lagrange + self.total_motor_lagrange
    }
}

impl XpbdConstraint<2> for RevoluteJoint {
    type SolverData = RevoluteJointSolverData;

    fn prepare(
        &mut self,
        bodies: [&RigidBodyQueryReadOnlyItem; 2],
        solver_data: &mut RevoluteJointSolverData,
    ) {
        let Some(local_anchor1) = self.local_anchor1() else {
            return;
        };
        let Some(local_anchor2) = self.local_anchor2() else {
            return;
        };
        let Some(local_basis1) = self.local_basis1() else {
            return;
        };
        let Some(local_basis2) = self.local_basis2() else {
            return;
        };

        // Prepare the point-to-point constraint.
        solver_data
            .point_constraint
            .prepare(bodies, local_anchor1, local_anchor2);

        // Prepare the base rotation difference.
        #[cfg(feature = "2d")]
        {
            solver_data.rotation_difference = (Rot::from(*bodies[0].rotation) * local_basis1)
                .angle_to(Rot::from(*bodies[1].rotation) * local_basis2);
        }
        #[cfg(feature = "3d")]
        {
            // Prepare the base axes.
            solver_data.a1 = Rot::from(*bodies[0].rotation) * local_basis1 * self.hinge_axis;
            solver_data.a2 = Rot::from(*bodies[1].rotation) * local_basis2 * self.hinge_axis;
            solver_data.b1 = Rot::from(*bodies[0].rotation)
                * local_basis1
                * self.hinge_axis.any_orthonormal_vector();
            solver_data.b2 = Rot::from(*bodies[1].rotation)
                * local_basis2
                * self.hinge_axis.any_orthonormal_vector();
        }
    }

    fn solve(
        &mut self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        solver_data: &mut RevoluteJointSolverData,
        dt: f32,
    ) -> Result<(), ConstraintSolveError> {
        let [body1, body2] = bodies;
        #[cfg(feature = "2d")]
        let [inertia1, inertia2] = inertias;

        #[cfg(feature = "2d")]
        {
            let inv_angular_inertia1 = inertia1.effective_inv_angular_inertia();
            let inv_angular_inertia2 = inertia2.effective_inv_angular_inertia();
            if self.motor.enabled {
                super::validate_motor_parameters(
                    self.motor.motor_model,
                    self.motor.max_torque,
                    self.motor.target_position,
                    self.motor.target_velocity,
                )?;
            }
            self.apply_motor(
                body1,
                body2,
                inv_angular_inertia1,
                inv_angular_inertia2,
                solver_data,
                dt,
            );
            self.apply_angle_limits(
                body1,
                body2,
                inv_angular_inertia1,
                inv_angular_inertia2,
                solver_data,
                dt,
            );
            solver_data
                .point_constraint
                .solve([body1, body2], inertias, self.point_compliance, dt);
        }
        #[cfg(feature = "3d")]
        {
            let mut bound = None;
            let initial_rows = self.frame_rows(body1, body2, solver_data, &mut bound);
            let input = self.motor_input(body1, body2, solver_data);
            let motor =
                super::shared::prepare_pair_motor(input.as_ref(), &initial_rows, inertias, dt)?;
            let mut accumulated = [0.0; 7];
            for iteration in 0..4 {
                let frames = self.frame_rows(body1, body2, solver_data, &mut bound);
                let rows = super::shared::rows_with_motor(
                    frames,
                    motor.as_ref(),
                    self.motor_input(body1, body2, solver_data),
                );
                if (iteration != 0 || motor.is_none())
                    && rows
                        .iter()
                        .zip(accumulated)
                        .all(|(row, lambda)| row.violation(lambda, dt) <= row.tolerance())
                {
                    break;
                }
                let update = super::shared::solve_constraint_rows(
                    [body1, body2],
                    inertias,
                    &rows,
                    &mut accumulated,
                    dt,
                )?;
                for (index, row) in rows.iter().enumerate() {
                    let impulse = update[index];
                    if index < 3 {
                        solver_data.point_constraint.total_lagrange -= row.linear2 * impulse;
                    } else if index < 5 {
                        solver_data.total_align_lagrange -= row.angular2 * impulse;
                    } else if index == 5 {
                        solver_data.total_limit_lagrange -= row.angular2 * impulse;
                    } else {
                        solver_data.total_motor_lagrange -= row.angular2 * impulse;
                    }
                }
            }
        }
        Ok(())
    }
}

impl RevoluteJoint {
    /// Applies angle limits to limit the relative rotation of the bodies around the `hinge_axis`.
    #[cfg(feature = "2d")]
    fn apply_angle_limits(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inv_angular_inertia1: SymmetricTensor,
        inv_angular_inertia2: SymmetricTensor,
        solver_data: &mut RevoluteJointSolverData,
        dt: f32,
    ) {
        let Some(Some(correction)) = self.angle_limit.map(|angle_limit| {
            #[cfg(feature = "2d")]
            {
                let rotation_difference = solver_data.rotation_difference
                    + body1.delta_rotation.angle_to(body2.delta_rotation);
                angle_limit.compute_correction(rotation_difference, PI)
            }
        }) else {
            return;
        };

        solver_data.total_limit_lagrange += self.align_orientation(
            body1,
            body2,
            inv_angular_inertia1,
            inv_angular_inertia2,
            correction,
            0.0,
            self.limit_compliance,
            dt,
        );
    }

    /// Applies motor forces to drive the joint towards the target velocity and/or position.
    ///
    /// Uses a PD controller approach with optional implicit Euler integration improved stability.
    #[cfg(feature = "2d")]
    fn apply_motor(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inv_angular_inertia1: SymmetricTensor,
        inv_angular_inertia2: SymmetricTensor,
        solver_data: &mut RevoluteJointSolverData,
        dt: f32,
    ) {
        let motor = &self.motor;

        if !motor.enabled {
            return;
        }

        let current_angle =
            solver_data.rotation_difference + body1.delta_rotation.angle_to(body2.delta_rotation);
        let relative_angular_velocity = body2.angular_velocity - body1.angular_velocity;
        let w_sum = inv_angular_inertia1 + inv_angular_inertia2;
        if w_sum <= 0.0 {
            return;
        }

        let velocity_error = motor.target_velocity - relative_angular_velocity;

        // Wrap position error to [-PI, PI] for shortest path rotation.
        let raw_error = motor.target_position - current_angle;
        let position_error = (raw_error + PI).rem_euclid(TAU) - PI;

        let delta_lagrange = super::motor_lagrange(
            motor.motor_model,
            position_error,
            velocity_error,
            w_sum,
            motor.max_torque,
            dt,
        );

        {
            solver_data.total_motor_lagrange -= delta_lagrange;
        }
        // Positive delta_lagrange increases body2's angular velocity relative to body1.
        self.apply_angular_lagrange_update(
            body1,
            body2,
            inv_angular_inertia1,
            inv_angular_inertia2,
            delta_lagrange,
        );
    }
}

impl PositionConstraint for RevoluteJoint {}

impl AngularConstraint for RevoluteJoint {}

#[cfg(feature = "3d")]
impl RevoluteJoint {
    pub(super) fn motor_input(
        &self,
        body1: &SolverBody,
        body2: &SolverBody,
        data: &RevoluteJointSolverData,
    ) -> Option<super::shared::MotorInput> {
        if !self.motor.enabled {
            return None;
        }
        let axis = body1.delta_rotation * data.a1;
        let b1 = body1.delta_rotation * data.b1;
        let b2 = body2.delta_rotation * data.b2;
        let row = super::shared::ConstraintRow {
            angular1: -axis,
            angular2: axis,
            kind: ConstraintResidualKind::Angular,
            ..default()
        };
        Some(super::shared::MotorInput {
            model: self.motor.motor_model,
            coordinate: b1.cross(b2).dot(axis).atan2(b1.dot(b2)),
            velocity: row.velocity([body1, body2]),
            target_position: self.motor.target_position,
            target_velocity: self.motor.target_velocity,
            max_force: self.motor.max_torque,
            angular: true,
            jacobian: row,
        })
    }

    pub(super) fn frame_rows(
        &self,
        body1: &SolverBody,
        body2: &SolverBody,
        data: &RevoluteJointSolverData,
        bound: &mut Option<(f32, [f64; 2])>,
    ) -> [super::shared::ConstraintRow; 6] {
        let r1 = body1.delta_rotation * data.point_constraint.world_r1;
        let r2 = body2.delta_rotation * data.point_constraint.world_r2;
        let separation = body2.delta_position - body1.delta_position + r2 - r1
            + data.point_constraint.center_difference;
        let a1 = body1.delta_rotation * data.a1;
        let a2 = body2.delta_rotation * data.a2;
        let tangent = a1.any_orthonormal_vector();
        let bitangent = a1.cross(tangent);
        // Evaluate the arc in f64 so the library's near-parallel cutoff stays
        // below the native angular residual precision.
        let error = bevy::math::DQuat::from_rotation_arc(
            a1.as_dvec3().normalize(),
            a2.as_dvec3().normalize(),
        )
        .to_scaled_axis()
        .as_vec3();
        let mut points =
            super::shared::point_rows(r1, r2, separation, self.point_compliance).into_iter();
        let mut rows = core::array::from_fn(|index| {
            if index < 3 {
                points.next().unwrap()
            } else {
                super::shared::ConstraintRow::default()
            }
        });
        for (index, axis) in [tangent, bitangent].into_iter().enumerate() {
            rows[index + 3].angular1 = -axis;
            rows[index + 3].angular2 = axis;
            rows[index + 3].error = f64::from(error.dot(axis));
            rows[index + 3].kind = ConstraintResidualKind::Angular;
            rows[index + 3].compliance = f64::from(self.align_compliance);
        }
        let b1 = body1.delta_rotation * data.b1;
        let b2 = body2.delta_rotation * data.b2;
        let angle = b1.cross(b2).dot(a1).atan2(b1.dot(b2));
        if let Some(limits) = self.angle_limit {
            if limits.min == limits.max {
                *bound = Some((limits.min, [f64::NEG_INFINITY, f64::INFINITY]));
            } else if angle < limits.min {
                *bound = Some((limits.min, [0.0, f64::INFINITY]));
            } else if angle > limits.max {
                *bound = Some((limits.max, [f64::NEG_INFINITY, 0.0]));
            }
        }
        if let Some((limit, bounds)) = *bound {
            rows[5].angular1 = -a1;
            rows[5].angular2 = a1;
            rows[5].error = f64::from(angle - limit);
            rows[5].kind = ConstraintResidualKind::Angular;
            rows[5].compliance = f64::from(self.limit_compliance);
            rows[5].bounds = bounds;
        }
        rows
    }
}

#[cfg(all(test, feature = "3d"))]
mod tests {
    use super::*;
    #[test]
    fn hinge_alignment_retains_small_finite_angles() {
        let joint = RevoluteJoint::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER);
        for angle in [0.0001, 0.1, 1.0] {
            let data = RevoluteJointSolverData {
                a1: Vec3::Y,
                a2: Quat::from_rotation_z(angle) * Vec3::Y,
                b1: Vec3::X,
                b2: Vec3::X,
                ..default()
            };
            let rows = joint.frame_rows(&SolverBody::DUMMY, &SolverBody::DUMMY, &data, &mut None);
            let magnitude = rows[3].error.hypot(rows[4].error);
            assert!(
                (magnitude - f64::from(angle)).abs() < 1e-7,
                "angle={angle}, residual={magnitude}"
            );
        }
    }
}
