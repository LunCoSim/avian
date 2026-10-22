use crate::dynamics::solver::solver_body::{SolverBody, SolverBodyInertia};
use bevy::prelude::*;

/// One scalar world-space Jacobian row. Impulse bounds apply to the accumulated
/// multiplier in this substep, including all nonlinear refinements.
pub(crate) struct ConstraintRow {
    pub linear1: Vec3,
    pub linear2: Vec3,
    pub angular1: Vec3,
    pub angular2: Vec3,
    pub error: f32,
    pub compliance: f32,
    pub bounds: [f64; 2],
}

impl Default for ConstraintRow {
    fn default() -> Self {
        Self {
            linear1: Vec3::ZERO,
            linear2: Vec3::ZERO,
            angular1: Vec3::ZERO,
            angular2: Vec3::ZERO,
            error: 0.0,
            compliance: 0.0,
            bounds: [f64::NEG_INFINITY, f64::INFINITY],
        }
    }
}

/// Solve coupled rows with an active set for unilateral constraints.
/// Locked body degrees of freedom contribute zero mobility.
pub(crate) fn solve_constraint_rows(
    bodies: [&mut SolverBody; 2],
    inertias: [&SolverBodyInertia; 2],
    rows: &[ConstraintRow; 6],
    accumulated: &mut [f64; 6],
    dt: f32,
) -> [f32; 6] {
    let [body1, body2] = bodies;
    let [inertia1, inertia2] = inertias;
    let m1 = inertia1.effective_inv_mass();
    let m2 = inertia2.effective_inv_mass();
    let i1 = inertia1.effective_inv_angular_inertia();
    let i2 = inertia2.effective_inv_angular_inertia();
    let matrix_i1 = Mat3::from(i1).as_dmat3();
    let matrix_i2 = Mat3::from(i2).as_dmat3();
    let mut mass = [[0.0_f64; 6]; 6];
    let mut rhs = [0.0_f64; 6];
    for (index, row) in rows.iter().enumerate() {
        for (column, other) in rows.iter().enumerate() {
            mass[index][column] = row
                .linear1
                .as_dvec3()
                .dot(m1.as_dvec3() * other.linear1.as_dvec3())
                + row
                    .linear2
                    .as_dvec3()
                    .dot(m2.as_dvec3() * other.linear2.as_dvec3())
                + row
                    .angular1
                    .as_dvec3()
                    .dot(matrix_i1 * other.angular1.as_dvec3())
                + row
                    .angular2
                    .as_dvec3()
                    .dot(matrix_i2 * other.angular2.as_dvec3());
        }
        let alpha = f64::from(row.compliance) / f64::from(dt).powi(2);
        mass[index][index] += alpha;
        rhs[index] = -f64::from(row.error) - alpha * accumulated[index];
    }
    let mut clamped = [None; 6];
    let update = loop {
        let mut matrix = mass;
        let mut error = rhs;
        for (index, value) in clamped.iter().enumerate() {
            if let Some(value) = value {
                for row in 0..6 {
                    error[row] -= mass[row][index] * value;
                    matrix[row][index] = 0.0;
                    matrix[index][row] = 0.0;
                }
            }
        }
        let mut update = solve_mobile_rows(matrix, error);
        for (index, value) in clamped.iter().enumerate() {
            if let Some(value) = value {
                update[index] = *value;
            }
        }
        let mut changed = false;
        for index in 0..6 {
            let total = accumulated[index] + update[index];
            let bounded = total.clamp(rows[index].bounds[0], rows[index].bounds[1]);
            if clamped[index].is_none() && bounded != total {
                clamped[index] = Some(bounded - accumulated[index]);
                changed = true;
            }
        }
        if !changed {
            break update;
        }
    };
    let mut p1 = Vec3::ZERO;
    let mut p2 = Vec3::ZERO;
    let mut a1 = Vec3::ZERO;
    let mut a2 = Vec3::ZERO;
    let mut impulses = [0.0; 6];
    for index in 0..6 {
        // Native solver coordinates and corrections are explicitly f32.
        let impulse = update[index] as f32;
        impulses[index] = impulse;
        accumulated[index] += f64::from(impulse);
        p1 += rows[index].linear1 * impulse;
        p2 += rows[index].linear2 * impulse;
        a1 += rows[index].angular1 * impulse;
        a2 += rows[index].angular2 * impulse;
    }
    body1.delta_position += m1 * p1;
    body2.delta_position += m2 * p2;
    body1.delta_rotation = (Quat::from_scaled_axis(i1 * a1) * body1.delta_rotation).normalize();
    body2.delta_rotation = (Quat::from_scaled_axis(i2 * a2) * body2.delta_rotation).normalize();
    impulses
}

// A rank-revealing symmetric elimination is required for valid locked axes.
// glam_matrix_extras' 6x6 LDLT divides by zero for these semidefinite systems.
// Null rows carry no mobility and therefore apply no physical correction.
#[cfg(feature = "3d")]
pub(crate) fn solve_mobile_rows(mut matrix: [[f64; 6]; 6], mut rhs: [f64; 6]) -> [f64; 6] {
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
    fn block_solves_mobile_rows_without_actuating_locked_directions() {
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
