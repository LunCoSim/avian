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
    ) {
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
        );
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
) {
    let [body1, body2] = bodies;
    let [inertia1, inertia2] = inertias;
    let m1 = inertia1.effective_inv_mass();
    let m2 = inertia2.effective_inv_mass();
    let i1 = inertia1.effective_inv_angular_inertia();
    let i2 = inertia2.effective_inv_angular_inertia();
    let alpha = [
        f64::from(point_compliance) / f64::from(dt).powi(2),
        f64::from(angle_compliance) / f64::from(dt).powi(2),
    ];
    let matrix_i1 = Mat3::from(i1).as_dmat3();
    let matrix_i2 = Mat3::from(i2).as_dmat3();
    let mut accumulated = [0.0_f64; 6];
    // Newton refinement accounts for the finite rotations applied to anchors.
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
        let mut linear1 = [Vec3::ZERO; 6];
        let mut linear2 = [Vec3::ZERO; 6];
        let mut angular1 = [Vec3::ZERO; 6];
        let mut angular2 = [Vec3::ZERO; 6];
        let mut error = [0.0_f64; 6];
        for (index, axis) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
            linear1[index] = -axis;
            linear2[index] = axis;
            angular1[index] = axis.cross(r1);
            angular2[index] = r2.cross(axis);
            angular1[index + 3] = -axis;
            angular2[index + 3] = axis;
            error[index] = f64::from(separation[index]);
            error[index + 3] = f64::from(angular_error[index]);
        }
        let mut mass = [[0.0_f64; 6]; 6];
        for row in 0..6 {
            for col in 0..6 {
                mass[row][col] = linear1[row]
                    .as_dvec3()
                    .dot(m1.as_dvec3() * linear1[col].as_dvec3())
                    + linear2[row]
                        .as_dvec3()
                        .dot(m2.as_dvec3() * linear2[col].as_dvec3())
                    + angular1[row]
                        .as_dvec3()
                        .dot(matrix_i1 * angular1[col].as_dvec3())
                    + angular2[row]
                        .as_dvec3()
                        .dot(matrix_i2 * angular2[col].as_dvec3());
            }
            let compliance = alpha[row / 3];
            mass[row][row] += compliance;
            error[row] = -error[row] - compliance * accumulated[row];
        }
        let update = solve_mobile_rows(mass, error);
        let mut p1 = Vec3::ZERO;
        let mut p2 = Vec3::ZERO;
        let mut a1 = Vec3::ZERO;
        let mut a2 = Vec3::ZERO;
        for row in 0..6 {
            // Native solver coordinates and corrections are explicitly f32.
            let impulse = update[row] as f32;
            accumulated[row] += f64::from(impulse);
            p1 += linear1[row] * impulse;
            p2 += linear2[row] * impulse;
            a1 += angular1[row] * impulse;
            a2 += angular2[row] * impulse;
        }
        body1.delta_position += m1 * p1;
        body2.delta_position += m2 * p2;
        body1.delta_rotation = (Quat::from_scaled_axis(i1 * a1) * body1.delta_rotation).normalize();
        body2.delta_rotation = (Quat::from_scaled_axis(i2 * a2) * body2.delta_rotation).normalize();
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
}

// A rank-revealing symmetric elimination is required for valid locked axes.
// glam_matrix_extras' 6x6 LDLT divides by zero for these semidefinite systems.
// Null rows carry no mobility and therefore apply no physical correction.
#[cfg(feature = "3d")]
fn solve_mobile_rows(mut matrix: [[f64; 6]; 6], mut rhs: [f64; 6]) -> [f64; 6] {
    let mut order = [0, 1, 2, 3, 4, 5];
    let scale = (0..6).map(|i| matrix[i][i]).fold(0.0_f64, f64::max);
    let tolerance = 64.0 * f64::EPSILON * scale;
    let mut rank = 0;
    for pivot in 0..6 {
        let selected = (pivot..6)
            .max_by(|&a, &b| matrix[a][a].total_cmp(&matrix[b][b]))
            .unwrap();
        if matrix[selected][selected] <= tolerance {
            break;
        }
        matrix.swap(pivot, selected);
        for row in &mut matrix {
            row.swap(pivot, selected);
        }
        rhs.swap(pivot, selected);
        order.swap(pivot, selected);
        for row in pivot + 1..6 {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for col in pivot + 1..6 {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
            matrix[row][pivot] = 0.0;
        }
        rank += 1;
    }
    let mut result = [0.0; 6];
    for row in (0..rank).rev() {
        let tail: f64 = (row + 1..rank)
            .map(|col| matrix[row][col] * result[order[col]])
            .sum();
        result[order[row]] = (rhs[row] - tail) / matrix[row][row];
    }
    result
}

#[cfg(all(test, feature = "3d"))]
mod tests {
    use super::solve_mobile_rows;

    #[test]
    fn fixed_block_solves_mobile_rows_without_actuating_locked_directions() {
        let mut matrix = [[0.0; 6]; 6];
        matrix[0][0] = 2.0;
        matrix[2][2] = 4.0;
        matrix[3][3] = 3.0;
        matrix[0][3] = 1.0;
        matrix[3][0] = 1.0;
        let rhs = [3.0, 100.0, 8.0, 4.0, 200.0, 300.0];
        let result = solve_mobile_rows(matrix, rhs);
        assert_eq!([result[1], result[4], result[5]], [0.0; 3]);
        for row in [0, 2, 3] {
            let actual: f64 = (0..6).map(|col| matrix[row][col] * result[col]).sum();
            assert!((actual - rhs[row]).abs() < 1e-12);
        }
        let dependent = [[1.0; 6]; 6];
        let result = solve_mobile_rows(dependent, [6.0; 6]);
        assert!(result.into_iter().all(f64::is_finite));
        assert!((result.into_iter().sum::<f64>() - 6.0).abs() < 1e-12);
    }
}
