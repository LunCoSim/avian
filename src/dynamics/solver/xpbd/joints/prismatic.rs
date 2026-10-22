use super::FixedAngleConstraintShared;
use crate::{
    dynamics::solver::{
        solver_body::{SolverBody, SolverBodyInertia},
        xpbd::*,
    },
    prelude::*,
};
use bevy::prelude::*;

/// Constraint data required by the XPBD constraint solver for a [`PrismaticJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct PrismaticJointSolverData {
    pub(super) world_r1: Vector,
    pub(super) world_r2: Vector,
    pub(super) center_difference: Vector,
    pub(super) free_axis1: Vector,
    pub(super) total_position_lagrange: Vector,
    pub(super) angle_constraint: FixedAngleConstraintShared,
    /// Accumulated motor Lagrange multiplier for this frame.
    pub(super) total_motor_lagrange: f32,
}

impl XpbdConstraintSolverData for PrismaticJointSolverData {
    fn clear_lagrange_multipliers(&mut self) {
        self.total_position_lagrange = Vector::ZERO;
        self.angle_constraint.clear_lagrange_multipliers();
        self.total_motor_lagrange = 0.0;
    }

    fn total_position_lagrange(&self) -> Vector {
        self.total_position_lagrange
    }

    fn total_rotation_lagrange(&self) -> AngularVector {
        self.angle_constraint.total_rotation_lagrange()
    }

    fn total_motor_lagrange(&self) -> f32 {
        self.total_motor_lagrange
    }
}

impl XpbdConstraint<2> for PrismaticJoint {
    type SolverData = PrismaticJointSolverData;

    fn prepare(
        &mut self,
        bodies: [&RigidBodyQueryReadOnlyItem; 2],
        solver_data: &mut PrismaticJointSolverData,
    ) {
        let [body1, body2] = bodies;

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
        solver_data.angle_constraint.prepare(
            (*body1.rotation).into(),
            (*body2.rotation).into(),
            local_basis1,
            local_basis2,
        );

        // Prepare the prismatic joint.
        solver_data.world_r1 = body1.rotation * (local_anchor1 - body1.center_of_mass.0);
        solver_data.world_r2 = body2.rotation * (local_anchor2 - body2.center_of_mass.0);
        solver_data.center_difference = (body2.position.0 - body1.position.0).f32()
            + (body2.rotation * body2.center_of_mass.0 - body1.rotation * body1.center_of_mass.0);
        solver_data.free_axis1 = Rot::from(*body1.rotation) * local_basis1 * self.slider_axis;
    }

    fn solve(
        &mut self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        solver_data: &mut PrismaticJointSolverData,
        dt: f32,
    ) -> Result<(), ConstraintSolveError> {
        let [body1, body2] = bodies;

        #[cfg(feature = "2d")]
        solver_data
            .angle_constraint
            .solve([body1, body2], inertias, self.angle_compliance, dt);

        // Solve motors before limits to give limits higher priority.
        #[cfg(feature = "2d")]
        {
            if self.motor.enabled {
                super::validate_motor_parameters(
                    self.motor.motor_model,
                    self.motor.max_force,
                    self.motor.target_position,
                    self.motor.target_velocity,
                )?;
            }
            self.apply_motor(body1, body2, inertias[0], inertias[1], solver_data, dt);
        }

        #[cfg(feature = "2d")]
        self.constrain_positions(body1, body2, inertias[0], inertias[1], solver_data, dt);
        #[cfg(feature = "3d")]
        self.constrain_frames([body1, body2], inertias, solver_data, dt)?;
        Ok(())
    }
}

impl PrismaticJoint {
    /// Constrains the relative positions of the bodies, only allowing translation along one free axis.
    #[cfg(feature = "2d")]
    fn constrain_positions(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inertia1: &SolverBodyInertia,
        inertia2: &SolverBodyInertia,
        solver_data: &mut PrismaticJointSolverData,
        dt: f32,
    ) {
        // Compute the effective inverse masses and angular inertias of the bodies.
        let inv_mass1 = inertia1.effective_inv_mass();
        let inv_mass2 = inertia2.effective_inv_mass();
        let inv_angular_inertia1 = inertia1.effective_inv_angular_inertia();
        let inv_angular_inertia2 = inertia2.effective_inv_angular_inertia();

        let world_r1 = body1.delta_rotation * solver_data.world_r1;
        let world_r2 = body2.delta_rotation * solver_data.world_r2;

        let mut delta_x = Vector::ZERO;

        let axis1 = body1.delta_rotation * solver_data.free_axis1;
        if let Some(limits) = self.limits {
            let separation = (body2.delta_position - body1.delta_position)
                + (world_r2 - world_r1)
                + solver_data.center_difference;
            delta_x += limits.compute_correction_along_axis(separation, axis1);
        }

        let zero_distance_limit = DistanceLimit::ZERO;

        #[cfg(feature = "2d")]
        {
            let axis2 = Vec2::new(axis1.y, -axis1.x);

            let separation = (body2.delta_position - body1.delta_position)
                + (world_r2 - world_r1)
                + solver_data.center_difference;
            delta_x += zero_distance_limit.compute_correction_along_axis(separation, axis2);
        }

        let magnitude = delta_x.length();

        if magnitude <= f32::EPSILON {
            return;
        }

        let dir = delta_x / magnitude;

        // Compute generalized inverse masses
        let w1 = PositionConstraint::compute_generalized_inverse_mass(
            self,
            inv_mass1.max_element(),
            inv_angular_inertia1,
            world_r1,
            dir,
        );
        let w2 = PositionConstraint::compute_generalized_inverse_mass(
            self,
            inv_mass2.max_element(),
            inv_angular_inertia2,
            world_r2,
            dir,
        );

        // Compute Lagrange multiplier update
        let delta_lagrange =
            compute_lagrange_update(0.0, magnitude, &[w1, w2], self.align_compliance, dt);
        let impulse = delta_lagrange * dir;
        solver_data.total_position_lagrange += impulse;

        // Apply positional correction to align the positions of the bodies
        self.apply_positional_impulse(
            body1, body2, inertia1, inertia2, impulse, world_r1, world_r2,
        );
    }
}

#[cfg(feature = "3d")]
impl PrismaticJoint {
    /// Couple orientation and anchor alignment so positional impulses cannot
    /// reintroduce a locked rotation. The slider axis belongs to body 1's frame.
    fn constrain_frames(
        &self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut PrismaticJointSolverData,
        dt: f32,
    ) -> Result<(), ConstraintSolveError> {
        let [body1, body2] = bodies;
        let mut bound = None;
        let initial_rows = self.frame_rows(body1, body2, data, &mut bound);
        let input = self.motor_input(body1, body2, data);
        let motor = super::shared::prepare_pair_motor(input.as_ref(), &initial_rows, inertias, dt)?;
        let mut accumulated = [0.0_f64; 7];
        for iteration in 0..4 {
            let frames = self.frame_rows(body1, body2, data, &mut bound);
            let rows = super::shared::rows_with_motor(
                frames,
                motor.as_ref(),
                self.motor_input(body1, body2, data),
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
                if index < 3 || index == 6 {
                    data.total_position_lagrange -= row.linear2 * update[index];
                } else {
                    data.angle_constraint.total_lagrange -= row.angular2 * update[index];
                }
            }
            data.total_motor_lagrange += update[6];
        }
        Ok(())
    }
    pub(super) fn motor_input(
        &self,
        body1: &SolverBody,
        body2: &SolverBody,
        data: &PrismaticJointSolverData,
    ) -> Option<super::shared::MotorInput> {
        if !self.motor.enabled {
            return None;
        }
        let r1 = body1.delta_rotation * data.world_r1;
        let r2 = body2.delta_rotation * data.world_r2;
        let separation =
            body2.delta_position - body1.delta_position + r2 - r1 + data.center_difference;
        let axis = body1.delta_rotation * data.free_axis1;
        let row = super::shared::ConstraintRow {
            linear1: -axis,
            linear2: axis,
            angular1: axis.cross(r1 + separation),
            angular2: r2.cross(axis),
            error_scale: super::shared::coordinate_scale(&[
                r1,
                r2,
                separation,
                data.center_difference,
            ]),
            ..default()
        };
        Some(super::shared::MotorInput {
            model: self.motor.motor_model,
            coordinate: separation.dot(axis),
            velocity: row.velocity([body1, body2]),
            target_position: self.motor.target_position,
            target_velocity: self.motor.target_velocity,
            max_force: self.motor.max_force,
            angular: false,
            jacobian: row,
        })
    }

    pub(super) fn frame_rows(
        &self,
        body1: &SolverBody,
        body2: &SolverBody,
        data: &PrismaticJointSolverData,
        bound: &mut Option<(f32, [f64; 2])>,
    ) -> [super::shared::ConstraintRow; 6] {
        let r1 = body1.delta_rotation * data.world_r1;
        let r2 = body2.delta_rotation * data.world_r2;
        let separation =
            body2.delta_position - body1.delta_position + r2 - r1 + data.center_difference;
        let axis = body1.delta_rotation * data.free_axis1;
        let tangent = axis.any_orthonormal_vector();
        let bitangent = axis.cross(tangent);
        let angular_error = data
            .angle_constraint
            .error(body1.delta_rotation, body2.delta_rotation);
        let slide = separation.dot(axis);
        if let Some(limits) = self.limits {
            if limits.min == limits.max {
                *bound = Some((limits.min, [f64::NEG_INFINITY, f64::INFINITY]));
            } else if slide < limits.min {
                *bound = Some((limits.min, [0.0, f64::INFINITY]));
            } else if slide > limits.max {
                *bound = Some((limits.max, [f64::NEG_INFINITY, 0.0]));
            }
        }
        let mut rows = core::array::from_fn(|_| super::shared::ConstraintRow::default());
        for (index, direction) in [tangent, bitangent, axis].into_iter().enumerate() {
            if index == 2 && bound.is_none() {
                continue;
            }
            rows[index].linear1 = -direction;
            rows[index].linear2 = direction;
            // Differentiating the moving axis adds direction x separation.
            rows[index].angular1 = direction.cross(r1 + separation);
            rows[index].angular2 = r2.cross(direction);
            rows[index].error = f64::from(separation.dot(direction));
            rows[index].error_scale =
                super::shared::coordinate_scale(&[r1, r2, separation, data.center_difference]);
            rows[index].compliance = f64::from(self.align_compliance);
            if index == 2 {
                let (limit, bounds) = (*bound).unwrap();
                rows[index].error -= f64::from(limit);
                rows[index].compliance = f64::from(self.limit_compliance);
                rows[index].bounds = bounds;
            }
        }
        for (index, direction) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
            rows[index + 3].angular1 = -direction;
            rows[index + 3].angular2 = direction;
            rows[index + 3].error = f64::from(angular_error[index]);
            rows[index + 3].kind = ConstraintResidualKind::Angular;
            rows[index + 3].compliance = f64::from(self.angle_compliance);
        }

        rows
    }
}

impl PrismaticJoint {
    /// Applies motor forces to drive the joint towards the target velocity and/or position.
    #[cfg(feature = "2d")]
    fn apply_motor(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inertia1: &SolverBodyInertia,
        inertia2: &SolverBodyInertia,
        solver_data: &mut PrismaticJointSolverData,
        dt: f32,
    ) {
        let motor = &self.motor;

        if !motor.enabled {
            return;
        }

        let axis1 = body1.delta_rotation * solver_data.free_axis1;
        let world_r1 = body1.delta_rotation * solver_data.world_r1;
        let world_r2 = body2.delta_rotation * solver_data.world_r2;

        let separation = (body2.delta_position - body1.delta_position)
            + (world_r2 - world_r1)
            + solver_data.center_difference;
        let current_position = separation.dot(axis1);
        let current_velocity =
            (body2.velocity_at_point(world_r2) - body1.velocity_at_point(world_r1)).dot(axis1);

        let inv_mass1 = inertia1.effective_inv_mass();
        let inv_mass2 = inertia2.effective_inv_mass();
        let inv_angular_inertia1 = inertia1.effective_inv_angular_inertia();
        let inv_angular_inertia2 = inertia2.effective_inv_angular_inertia();

        let w1 = PositionConstraint::compute_generalized_inverse_mass(
            self,
            inv_mass1.max_element(),
            inv_angular_inertia1,
            world_r1,
            axis1,
        );
        let w2 = PositionConstraint::compute_generalized_inverse_mass(
            self,
            inv_mass2.max_element(),
            inv_angular_inertia2,
            world_r2,
            axis1,
        );

        let w_sum = w1 + w2;
        if w_sum <= 0.0 {
            return;
        }

        let velocity_error = motor.target_velocity - current_velocity;
        let position_error = motor.target_position - current_position;

        let delta_lagrange = super::motor_lagrange(
            motor.motor_model,
            position_error,
            velocity_error,
            w_sum,
            motor.max_force,
            dt,
        );

        solver_data.total_motor_lagrange += delta_lagrange;

        let impulse = delta_lagrange * axis1;
        solver_data.total_position_lagrange -= impulse;

        // Negate impulse: apply_positional_impulse convention is opposite to motor direction.
        self.apply_positional_impulse(
            body1, body2, inertia1, inertia2, -impulse, world_r1, world_r2,
        );
    }
}

impl PositionConstraint for PrismaticJoint {}

impl AngularConstraint for PrismaticJoint {}

#[cfg(all(test, feature = "3d"))]
mod tests {
    use super::*;

    #[test]
    fn offset_slider_enforces_alignment_and_hard_stop_together() {
        for axis in [Vec3::Y, Vec3::new(1.0, 2.0, 3.0).normalize()] {
            let offset = Vec3::new(2.0, -1.0, 0.5);
            let perpendicular = axis.any_orthonormal_vector();
            let mut mobile = SolverBody::DUMMY;
            mobile.delta_position = axis * 0.1 + perpendicular * 0.05;
            let mut fixed = SolverBody::DUMMY;
            let inertia = SolverBodyInertia::new(
                0.001,
                crate::math::SymmetricTensor::IDENTITY * 0.002,
                LockedAxes::default(),
                0,
                true,
            );
            let mut data = PrismaticJointSolverData {
                world_r1: offset,
                center_difference: offset,
                free_axis1: axis,
                ..default()
            };
            let mut joint = PrismaticJoint::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
                .with_limits(-0.02, 0.02);
            joint
                .solve(
                    [&mut mobile, &mut fixed],
                    [&inertia, &SolverBodyInertia::DUMMY],
                    &mut data,
                    1.0 / 480.0,
                )
                .unwrap();
            let residual = -mobile.delta_position - mobile.delta_rotation * offset + offset;
            let slide = residual.dot(axis);
            let cross_axis = (residual - slide * axis).length();
            let angle = mobile.delta_rotation.angle_between(Quat::IDENTITY);
            assert!(
                cross_axis < 0.0002 && slide.abs() <= 0.0202 && angle < 0.0002,
                "axis={axis}, cross_axis={cross_axis}, slide={slide}, angle={angle}"
            );
        }
    }

    #[test]
    fn slider_preserves_free_motion_and_uses_independent_limit_compliance() {
        let inertia = SolverBodyInertia::new(
            1.0,
            crate::math::SymmetricTensor::IDENTITY,
            LockedAxes::default(),
            0,
            true,
        );
        for (slide, compliance, expected) in [
            (0.01, 0.0, 0.01),
            (-0.01, 0.0, -0.01),
            (0.1, 0.0, 0.02),
            (-0.1, 0.0, -0.02),
            (0.1, 1.0, 0.06),
            (-0.1, 1.0, -0.06),
        ] {
            let mut fixed = SolverBody::DUMMY;
            let mut mobile = SolverBody::DUMMY;
            mobile.delta_position = Vec3::Y * slide + Vec3::X * 0.03;
            let mut data = PrismaticJointSolverData {
                free_axis1: Vec3::Y,
                ..default()
            };
            let mut joint = PrismaticJoint::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
                .with_limits(-0.02, 0.02)
                .with_limit_compliance(compliance);
            joint
                .solve(
                    [&mut fixed, &mut mobile],
                    [&SolverBodyInertia::DUMMY, &inertia],
                    &mut data,
                    1.0,
                )
                .unwrap();
            assert!(
                (mobile.delta_position.y - expected).abs() < 1.0e-6,
                "slide={slide}, compliance={compliance}, actual={:?}",
                mobile.delta_position
            );
            assert!(mobile.delta_position.x.abs() < 1.0e-6);
            assert_eq!(fixed.delta_position, Vec3::ZERO);
        }
    }
}
