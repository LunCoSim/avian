use super::{FixedAngleConstraintShared, PointConstraintShared};
use crate::{
    dynamics::solver::{
        solver_body::{SolverBody, SolverBodyInertia},
        xpbd::*,
    },
    prelude::*,
};
use bevy::prelude::*;

/// Solver data for the [`FixedJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct FixedJointSolverData {
    pub(super) point_constraint: PointConstraintShared,
    pub(super) angle_constraint: FixedAngleConstraintShared,
}

impl XpbdConstraintSolverData for FixedJointSolverData {
    fn clear_lagrange_multipliers(&mut self) {
        self.point_constraint.clear_lagrange_multipliers();
        self.angle_constraint.clear_lagrange_multipliers();
    }

    fn total_position_lagrange(&self) -> Vector {
        self.point_constraint.total_position_lagrange()
    }

    fn total_rotation_lagrange(&self) -> AngularVector {
        self.angle_constraint.total_rotation_lagrange()
    }
}

impl XpbdConstraint<2> for FixedJoint {
    type SolverData = FixedJointSolverData;

    fn prepare(
        &mut self,
        bodies: [&RigidBodyQueryReadOnlyItem; 2],
        solver_data: &mut FixedJointSolverData,
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
        solver_data
            .point_constraint
            .prepare(bodies, local_anchor1, local_anchor2);

        // Prepare the angular constraint.
        solver_data.angle_constraint.prepare(
            (*body1.rotation).into(),
            (*body2.rotation).into(),
            local_basis1,
            local_basis2,
        );
    }

    fn solve(
        &mut self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        solver_data: &mut FixedJointSolverData,
        dt: f32,
    ) -> Result<(), ConstraintSolveError> {
        let [body1, body2] = bodies;

        #[cfg(feature = "2d")]
        {
            solver_data
                .angle_constraint
                .solve([body1, body2], inertias, self.angle_compliance, dt);
            solver_data
                .point_constraint
                .solve([body1, body2], inertias, self.point_compliance, dt);
        }
        #[cfg(feature = "3d")]
        solve_fixed_block(
            [body1, body2],
            inertias,
            solver_data,
            self.point_compliance,
            self.angle_compliance,
            dt,
        )?;
        Ok(())
    }
}

impl PositionConstraint for FixedJoint {}

impl AngularConstraint for FixedJoint {}

// Solve translation and rotation together: an off-center point correction must
// not undo the fixed orientation. The six rows are the world-space Jacobian of
// the two coincident joint frames. Locked degrees of freedom have zero mobility.
#[cfg(feature = "3d")]
fn solve_fixed_block(
    bodies: [&mut SolverBody; 2],
    inertias: [&SolverBodyInertia; 2],
    data: &mut FixedJointSolverData,
    point_compliance: f32,
    angle_compliance: f32,
    dt: f32,
) -> Result<(), ConstraintSolveError> {
    let [body1, body2] = bodies;
    let mut accumulated = [0.0_f64; 6];
    // Newton refinement accounts for finite rotations of offset anchors.
    for _ in 0..3 {
        let r1 = body1.delta_rotation * data.point_constraint.world_r1;
        let r2 = body2.delta_rotation * data.point_constraint.world_r2;
        let separation = body2.delta_position - body1.delta_position + r2 - r1
            + data.point_constraint.center_difference;
        let angular_error = data
            .angle_constraint
            .error(body1.delta_rotation, body2.delta_rotation);
        if separation == Vec3::ZERO && angular_error == Vec3::ZERO {
            break;
        }
        let rows = frame_rows(body1, body2, data, point_compliance, angle_compliance);
        super::shared::solve_constraint_rows(
            [body1, body2],
            inertias,
            &rows,
            &mut accumulated,
            dt,
        )?;
    }
    data.point_constraint.total_lagrange -= Vec3::new(
        accumulated[0] as f32,
        accumulated[1] as f32,
        accumulated[2] as f32,
    );
    data.angle_constraint.total_lagrange -= Vec3::new(
        accumulated[3] as f32,
        accumulated[4] as f32,
        accumulated[5] as f32,
    );
    Ok(())
}

#[cfg(feature = "3d")]
pub(super) fn frame_rows(
    body1: &SolverBody,
    body2: &SolverBody,
    data: &FixedJointSolverData,
    point_compliance: f32,
    angle_compliance: f32,
) -> [super::shared::ConstraintRow; 6] {
    let r1 = body1.delta_rotation * data.point_constraint.world_r1;
    let r2 = body2.delta_rotation * data.point_constraint.world_r2;
    let separation = body2.delta_position - body1.delta_position + r2 - r1
        + data.point_constraint.center_difference;
    let angular_error = data
        .angle_constraint
        .error(body1.delta_rotation, body2.delta_rotation);
    let mut points = super::shared::point_rows(r1, r2, separation, point_compliance).into_iter();
    core::array::from_fn(|index| {
        if index < 3 {
            points.next().unwrap()
        } else {
            let axis = [Vec3::X, Vec3::Y, Vec3::Z][index - 3];
            super::shared::ConstraintRow {
                angular1: -axis,
                angular2: axis,
                error: f64::from(angular_error[index - 3]),
                kind: crate::dynamics::solver::xpbd::ConstraintResidualKind::Angular,
                compliance: f64::from(angle_compliance),
                ..default()
            }
        }
    })
}
