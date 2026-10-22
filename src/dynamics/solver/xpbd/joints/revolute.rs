use super::PointConstraintShared;
use crate::{
    dynamics::solver::{
        solver_body::{SolverBody, SolverBodyInertia},
        xpbd::*,
    },
    prelude::*,
};
use bevy::prelude::*;

use core::f32::consts::{PI, TAU};

/// Constraint data required by the XPBD constraint solver for a [`RevoluteJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct RevoluteJointSolverData {
    #[cfg(feature = "3d")]
    #[reflect(ignore)]
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub(super) velocity_cache: super::shared::VelocityRowCache,
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
    ) {
        let [body1, body2] = bodies;
        let [inertia1, inertia2] = inertias;

        // Get the effective inverse angular inertia of the bodies.
        let inv_angular_inertia1 = inertia1.effective_inv_angular_inertia();
        let inv_angular_inertia2 = inertia2.effective_inv_angular_inertia();

        #[cfg(feature = "3d")]
        {
            // Constrain the relative rotation of the bodies, only allowing rotation around one free axis
            let a1 = body1.delta_rotation * solver_data.a1;
            let a2 = body2.delta_rotation * solver_data.a2;
            let difference = a1.cross(a2);

            solver_data.total_align_lagrange += self.align_orientation(
                body1,
                body2,
                inv_angular_inertia1,
                inv_angular_inertia2,
                difference,
                0.0,
                self.align_compliance,
                dt,
            );
        }

        // Solve motors before limits to give limits higher priority.
        self.apply_motor(
            body1,
            body2,
            inv_angular_inertia1,
            inv_angular_inertia2,
            solver_data,
            dt,
        );

        // Apply angle limits when rotating around the free axis
        self.apply_angle_limits(
            body1,
            body2,
            inv_angular_inertia1,
            inv_angular_inertia2,
            solver_data,
            dt,
        );

        // Align positions
        solver_data
            .point_constraint
            .solve([body1, body2], inertias, self.point_compliance, dt);
    }
    #[cfg(feature = "3d")]
    fn prepare_velocity_projection(
        &mut self,
        bodies: [&SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut RevoluteJointSolverData,
    ) {
        let [a, b] = bodies;
        let r1 = a.delta_rotation * data.point_constraint.world_r1;
        let r2 = b.delta_rotation * data.point_constraint.world_r2;
        let axis1 = a.delta_rotation * data.a1;
        let axis2 = b.delta_rotation * data.a2;
        let tangent2 = b.delta_rotation * data.b2;
        let mut rows = core::array::from_fn(|_| super::shared::ConstraintRow::default());
        if self.point_compliance == 0.0 {
            for (i, n) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
                rows[i].linear1 = -n;
                rows[i].linear2 = n;
                rows[i].angular1 = n.cross(r1);
                rows[i].angular2 = r2.cross(n);
            }
        }
        if self.align_compliance == 0.0 {
            for (i, tangent) in [tangent2, axis2.cross(tangent2)].into_iter().enumerate() {
                let gradient = axis1.cross(tangent);
                rows[i + 3].angular1 = gradient;
                rows[i + 3].angular2 = -gradient;
            }
        }
        super::shared::prepare_velocity_rows(inertias, &rows, &mut data.velocity_cache);
    }
    #[cfg(feature = "3d")]
    fn project_velocities(
        &mut self,
        bodies: [&mut SolverBody; 2],
        _inertias: [&SolverBodyInertia; 2],
        data: &mut RevoluteJointSolverData,
        dt: f32,
    ) {
        let impulse = super::shared::solve_velocity_rows(bodies, &data.velocity_cache);
        data.point_constraint.total_lagrange -= dt * Vec3::new(impulse[0], impulse[1], impulse[2]);
        data.total_align_lagrange -= dt
            * (data.velocity_cache.gradients[3][3] * impulse[3]
                + data.velocity_cache.gradients[4][3] * impulse[4]);
    }
}

impl RevoluteJoint {
    /// Applies angle limits to limit the relative rotation of the bodies around the `hinge_axis`.
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
            #[cfg(feature = "3d")]
            {
                // [n, n1, n2] = [a1, b1, b2], where [a, b, c] are perpendicular unit axes on the bodies.
                let a1 = body1.delta_rotation * solver_data.a1;
                let b1 = body1.delta_rotation * solver_data.b1;
                let b2 = body2.delta_rotation * solver_data.b2;
                angle_limit.compute_correction(a1, b1, b2, PI)
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

        #[cfg(feature = "2d")]
        let current_angle =
            solver_data.rotation_difference + body1.delta_rotation.angle_to(body2.delta_rotation);
        #[cfg(feature = "3d")]
        let a1 = body1.delta_rotation * solver_data.a1;
        #[cfg(feature = "3d")]
        let current_angle = {
            let b1 = body1.delta_rotation * solver_data.b1;
            let b2 = body2.delta_rotation * solver_data.b2;
            let sin_angle = b1.cross(b2).dot(a1);
            let cos_angle = b1.dot(b2);
            sin_angle.atan2(cos_angle)
        };

        #[cfg(feature = "2d")]
        let relative_angular_velocity = body2.angular_velocity - body1.angular_velocity;
        #[cfg(feature = "3d")]
        let relative_angular_velocity = (body2.angular_velocity - body1.angular_velocity).dot(a1);

        #[cfg(feature = "2d")]
        let w_sum = inv_angular_inertia1 + inv_angular_inertia2;
        #[cfg(feature = "3d")]
        let w_sum =
            AngularConstraint::compute_generalized_inverse_mass(self, inv_angular_inertia1, a1)
                + AngularConstraint::compute_generalized_inverse_mass(
                    self,
                    inv_angular_inertia2,
                    a1,
                );

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

        #[cfg(feature = "2d")]
        {
            solver_data.total_motor_lagrange -= delta_lagrange;
        }
        #[cfg(feature = "3d")]
        {
            solver_data.total_motor_lagrange -= delta_lagrange * a1;
        }

        // Positive delta_lagrange increases body2's angular velocity relative to body1.
        #[cfg(feature = "2d")]
        self.apply_angular_lagrange_update(
            body1,
            body2,
            inv_angular_inertia1,
            inv_angular_inertia2,
            delta_lagrange,
        );
        #[cfg(feature = "3d")]
        self.apply_angular_lagrange_update(
            body1,
            body2,
            inv_angular_inertia1,
            inv_angular_inertia2,
            delta_lagrange,
            a1,
        );
    }
}

impl PositionConstraint for RevoluteJoint {}

impl AngularConstraint for RevoluteJoint {}
