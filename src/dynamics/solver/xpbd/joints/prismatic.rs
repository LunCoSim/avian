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
    #[cfg(feature = "3d")]
    #[reflect(ignore)]
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub(super) velocity_cache: super::shared::VelocityRowCache,
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
    ) {
        let [body1, body2] = bodies;

        #[cfg(feature = "2d")]
        solver_data
            .angle_constraint
            .solve([body1, body2], inertias, self.angle_compliance, dt);

        // Solve motors before limits to give limits higher priority.
        self.apply_motor(body1, body2, inertias[0], inertias[1], solver_data, dt);

        #[cfg(feature = "2d")]
        self.constrain_positions(body1, body2, inertias[0], inertias[1], solver_data, dt);
        #[cfg(feature = "3d")]
        self.constrain_frames([body1, body2], inertias, solver_data, dt);
    }
    #[cfg(feature = "3d")]
    fn prepare_velocity_projection(
        &mut self,
        bodies: [&SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut PrismaticJointSolverData,
    ) {
        let [a, b] = bodies;
        let r1 = a.delta_rotation * data.world_r1;
        let r2 = b.delta_rotation * data.world_r2;
        let separation = b.delta_position - a.delta_position + r2 - r1 + data.center_difference;
        let axis = a.delta_rotation * data.free_axis1;
        let tangent = axis.any_orthonormal_vector();
        let bitangent = axis.cross(tangent);
        let mut rows = core::array::from_fn(|_| super::shared::ConstraintRow::default());
        for (i, n) in [tangent, bitangent].into_iter().enumerate() {
            if self.align_compliance == 0.0 {
                rows[i].linear1 = -n;
                rows[i].linear2 = n;
                rows[i].angular1 = n.cross(r1 + separation);
                rows[i].angular2 = r2.cross(n);
            }
        }
        for (i, n) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
            if self.angle_compliance == 0.0 {
                rows[i + 3].angular1 = -n;
                rows[i + 3].angular2 = n;
            }
        }
        super::shared::prepare_velocity_rows(inertias, &rows, &mut data.velocity_cache);
    }
    #[cfg(feature = "3d")]
    fn project_velocities(
        &mut self,
        bodies: [&mut SolverBody; 2],
        _inertias: [&SolverBodyInertia; 2],
        data: &mut PrismaticJointSolverData,
        dt: f32,
    ) {
        let impulse = super::shared::solve_velocity_rows(bodies, &data.velocity_cache);
        let tangent = data.velocity_cache.gradients[0][1];
        let bitangent = data.velocity_cache.gradients[1][1];
        data.total_position_lagrange -= dt * (tangent * impulse[0] + bitangent * impulse[1]);
        data.angle_constraint.total_lagrange -= dt * Vec3::new(impulse[3], impulse[4], impulse[5]);
    }
}

impl PrismaticJoint {
    /// Constrains the relative positions of the bodies, only allowing translation along one free axis.
    ///
    /// Returns the force exerted by this constraint.
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
    ) {
        let [body1, body2] = bodies;
        let mut accumulated = [0.0_f64; 6];
        let mut bound = None;
        for _ in 0..3 {
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
                if slide < limits.min {
                    bound = Some((limits.min, [0.0, f64::INFINITY]));
                } else if slide > limits.max {
                    bound = Some((limits.max, [f64::NEG_INFINITY, 0.0]));
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
                rows[index].error = separation.dot(direction);
                rows[index].compliance = self.align_compliance;
                if index == 2 {
                    let (limit, bounds) = bound.unwrap();
                    rows[index].error -= limit;
                    rows[index].compliance = self.limit_compliance;
                    rows[index].bounds = bounds;
                }
            }
            for (index, direction) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
                rows[index + 3].angular1 = -direction;
                rows[index + 3].angular2 = direction;
                rows[index + 3].error = angular_error[index];
                rows[index + 3].compliance = self.angle_compliance;
            }
            if rows.iter().all(|row| row.error.abs() < 1.0e-7) {
                break;
            }
            let update = super::shared::solve_constraint_rows(
                [body1, body2],
                inertias,
                &rows,
                &mut accumulated,
                dt,
            );
            data.total_position_lagrange -=
                tangent * update[0] + bitangent * update[1] + axis * update[2];
            data.angle_constraint.total_lagrange -= Vec3::new(update[3], update[4], update[5]);
        }
    }
}

impl PrismaticJoint {
    /// Applies motor forces to drive the joint towards the target velocity and/or position.
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
            joint.solve(
                [&mut mobile, &mut fixed],
                [&inertia, &SolverBodyInertia::DUMMY],
                &mut data,
                1.0 / 480.0,
            );
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
            joint.solve(
                [&mut fixed, &mut mobile],
                [&SolverBodyInertia::DUMMY, &inertia],
                &mut data,
                1.0,
            );
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
