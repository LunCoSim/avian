use crate::dynamics::solver::solver_body::{SolverBody, SolverBodyInertia};
use crate::dynamics::solver::xpbd::{ConstraintResidualKind, ConstraintSolveError};
use bevy::{math::DVec3, prelude::*};

/// One scalar world-space Jacobian row. Impulse bounds apply to the accumulated
/// multiplier in this substep, including all nonlinear refinements.
#[derive(Clone, Debug)]
pub(crate) struct ConstraintRow {
    pub linear1: Vec3,
    pub linear2: Vec3,
    pub angular1: Vec3,
    pub angular2: Vec3,
    pub error: f64,
    pub compliance: f64,
    pub error_scale: f64,
    pub kind: ConstraintResidualKind,
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
            error_scale: 1.0,
            kind: ConstraintResidualKind::Linear,
            bounds: [f64::NEG_INFINITY, f64::INFINITY],
        }
    }
}

impl ConstraintRow {
    pub(crate) fn tolerance(&self) -> f64 {
        8.0 * f64::from(f32::EPSILON) * self.error_scale
    }
    pub(crate) fn residual(&self, lambda: f64, dt: f32) -> f64 {
        self.error + self.compliance * lambda / f64::from(dt).powi(2)
    }
    pub(crate) fn violation(&self, lambda: f64, dt: f32) -> f64 {
        let residual = self.residual(lambda, dt);
        if self.bounds[0] == self.bounds[1] {
            0.0
        } else if lambda == self.bounds[0] {
            (-residual).max(0.0)
        } else if lambda == self.bounds[1] {
            residual.max(0.0)
        } else {
            residual.abs()
        }
    }
    pub(crate) fn velocity(&self, bodies: [&SolverBody; 2]) -> f32 {
        self.linear1.dot(bodies[0].linear_velocity)
            + self.linear2.dot(bodies[1].linear_velocity)
            + self.angular1.dot(bodies[0].angular_velocity)
            + self.angular2.dot(bodies[1].angular_velocity)
    }
}

pub(crate) fn coordinate_scale(vectors: &[Vec3]) -> f64 {
    vectors
        .iter()
        .map(|vector| f64::from(vector.abs().max_element()))
        .sum::<f64>()
        .max(1.0)
}

pub(crate) fn jacobian_dot(
    a: (DVec3, DVec3),
    b: (DVec3, DVec3),
    inertia: &SolverBodyInertia,
) -> f64 {
    a.0.dot(inertia.effective_inv_mass().as_dvec3() * b.0)
        + a.1
            .dot(Mat3::from(inertia.effective_inv_angular_inertia()).as_dmat3() * b.1)
}
pub(crate) fn pair_inverse_mass(
    a: &ConstraintRow,
    b: &ConstraintRow,
    inertias: [&SolverBodyInertia; 2],
) -> f64 {
    jacobian_dot(
        (a.linear1.as_dvec3(), a.angular1.as_dvec3()),
        (b.linear1.as_dvec3(), b.angular1.as_dvec3()),
        inertias[0],
    ) + jacobian_dot(
        (a.linear2.as_dvec3(), a.angular2.as_dvec3()),
        (b.linear2.as_dvec3(), b.angular2.as_dvec3()),
        inertias[1],
    )
}

/// Mobility after the bilateral mechanism has transmitted an impulse.
pub(crate) fn projected_inverse_mass(
    mass: &[f64],
    coupling: &[f64],
    free_mass: f64,
) -> Result<f64, ConstraintSolveError> {
    let mut factors = mass.to_vec();
    let mut rhs = coupling.to_vec();
    let mut solution = vec![0.0; rhs.len()];
    solve_rank_revealing(&mut factors, &mut rhs, &mut solution);
    let reduction: f64 = coupling.iter().zip(&solution).map(|(a, b)| a * b).sum();
    let projected = free_mass - reduction;
    let scale = free_mass.abs()
        + coupling
            .iter()
            .zip(&solution)
            .map(|(a, b)| (a * b).abs())
            .sum::<f64>();
    let tolerance = 128.0 * f64::EPSILON * scale * (coupling.len() + 1) as f64;
    if !projected.is_finite() || projected < -tolerance {
        return Err(ConstraintSolveError::InvalidEffectiveMass);
    }
    Ok(if projected.abs() <= tolerance {
        0.0
    } else {
        projected
    })
}

pub(crate) fn pair_projected_mass(
    rows: &[ConstraintRow],
    motor: &ConstraintRow,
    inertias: [&SolverBodyInertia; 2],
    dt: f32,
) -> Result<f64, ConstraintSolveError> {
    let bilateral: Vec<_> = rows
        .iter()
        .filter(|row| row.bounds == [f64::NEG_INFINITY, f64::INFINITY])
        .collect();
    let n = bilateral.len();
    let mut mass = vec![0.0; n * n];
    let mut coupling = vec![0.0; n];
    for (i, row) in bilateral.iter().enumerate() {
        for (j, other) in bilateral.iter().enumerate() {
            mass[i * n + j] = pair_inverse_mass(row, other, inertias);
        }
        mass[i * n + i] += row.compliance / f64::from(dt).powi(2);
        coupling[i] = pair_inverse_mass(row, motor, inertias);
    }
    projected_inverse_mass(&mass, &coupling, pair_inverse_mass(motor, motor, inertias))
}

/// Solve coupled rows with an active set for unilateral constraints.
/// Locked body degrees of freedom contribute zero mobility.
pub(crate) fn solve_constraint_rows<const N: usize>(
    bodies: [&mut SolverBody; 2],
    inertias: [&SolverBodyInertia; 2],
    rows: &[ConstraintRow; N],
    accumulated: &mut [f64; N],
    dt: f32,
) -> Result<[f32; N], ConstraintSolveError> {
    let [body1, body2] = bodies;
    let [inertia1, inertia2] = inertias;
    let m1 = inertia1.effective_inv_mass();
    let m2 = inertia2.effective_inv_mass();
    let i1 = inertia1.effective_inv_angular_inertia();
    let i2 = inertia2.effective_inv_angular_inertia();
    let mut mass = [[0.0_f64; N]; N];
    let mut rhs = [0.0_f64; N];
    for (index, row) in rows.iter().enumerate() {
        for (column, other) in rows.iter().enumerate() {
            mass[index][column] = pair_inverse_mass(row, other, inertias);
        }
        let alpha = row.compliance / f64::from(dt).powi(2);
        mass[index][index] += alpha;
        rhs[index] = -row.error - alpha * accumulated[index];
    }
    let bounds: Vec<_> = rows.iter().map(|row| row.bounds).collect();
    let tolerances: Vec<_> = rows.iter().map(ConstraintRow::tolerance).collect();
    let solution =
        solve_bounded_rows(mass.as_flattened(), &rhs, &bounds, accumulated, &tolerances)?;
    let update = &solution.delta;
    if update.iter().zip(rows).any(|(&impulse, row)| {
        !impulse.is_finite()
            || impulse.abs() > f64::from(f32::MAX)
            || !(row.linear1 * impulse as f32).is_finite()
            || !(row.linear2 * impulse as f32).is_finite()
            || !(row.angular1 * impulse as f32).is_finite()
            || !(row.angular2 * impulse as f32).is_finite()
    }) {
        return Err(ConstraintSolveError::NonFiniteCorrection);
    }
    let mut p1 = Vec3::ZERO;
    let mut p2 = Vec3::ZERO;
    let mut a1 = Vec3::ZERO;
    let mut a2 = Vec3::ZERO;
    let mut impulses = [0.0; N];
    for index in 0..N {
        // Native solver coordinates and corrections are explicitly f32.
        let impulse = update[index] as f32;
        impulses[index] = impulse;
        accumulated[index] = solution.total[index];
        p1 += rows[index].linear1 * impulse;
        p2 += rows[index].linear2 * impulse;
        a1 += rows[index].angular1 * impulse;
        a2 += rows[index].angular2 * impulse;
    }
    if !(m1 * p1).is_finite()
        || !(m2 * p2).is_finite()
        || !(i1 * a1).is_finite()
        || !(i2 * a2).is_finite()
    {
        return Err(ConstraintSolveError::NonFiniteCorrection);
    }
    body1.delta_position += m1 * p1;
    body2.delta_position += m2 * p2;
    body1.delta_rotation = (Quat::from_scaled_axis(i1 * a1) * body1.delta_rotation).normalize();
    body2.delta_rotation = (Quat::from_scaled_axis(i2 * a2) * body2.delta_rotation).normalize();
    Ok(impulses)
}

#[derive(Debug, PartialEq)]
pub(crate) struct BoundedRowSolution {
    pub delta: Vec<f64>,
    pub total: Vec<f64>,
}

/// Primal feasible active-set solve. Releasing a bound with a negative dual
/// residual is essential when several contact normals share articulated mass.
pub(crate) fn solve_bounded_rows(
    mass: &[f64],
    rhs: &[f64],
    bounds: &[[f64; 2]],
    accumulated: &[f64],
    tolerances: &[f64],
) -> Result<BoundedRowSolution, ConstraintSolveError> {
    let n = rhs.len();
    if mass
        .iter()
        .chain(rhs)
        .chain(accumulated)
        .chain(tolerances)
        .any(|value| !value.is_finite())
        || bounds
            .iter()
            .any(|bound| bound[0].is_nan() || bound[1].is_nan() || bound[0] > bound[1])
    {
        return Err(ConstraintSolveError::NonFiniteSystem);
    }
    let max_iterations = (n + 1).saturating_mul(8);
    let mut iterations = 0;

    let mut current: Vec<_> = (0..n)
        .map(|i| accumulated[i].clamp(bounds[i][0], bounds[i][1]) - accumulated[i])
        .collect();
    let mut active: Vec<i8> = (0..n)
        .map(|i| {
            let total = current[i] + accumulated[i];
            if mass[i * n + i] == 0.0 {
                2
            } else if total == bounds[i][0] {
                -1
            } else if total == bounds[i][1] {
                1
            } else {
                0
            }
        })
        .collect();
    loop {
        if iterations == max_iterations {
            return Err(ConstraintSolveError::ActiveSetIterationLimit {
                dimension: n,
                iterations,
            });
        }
        iterations += 1;

        let mut matrix = mass.to_vec();
        let mut error = rhs.to_vec();
        for index in 0..n {
            if active[index] != 0 {
                for row in 0..n {
                    error[row] -= mass[row * n + index] * current[index];
                    matrix[row * n + index] = 0.0;
                    matrix[index * n + row] = 0.0;
                }
            }
        }
        let mut candidate = vec![0.0; n];
        solve_rank_revealing(&mut matrix, &mut error, &mut candidate);
        for index in 0..n {
            if active[index] != 0 {
                candidate[index] = current[index];
            }
        }
        let mut step = 1.0;
        let mut blocker = None;
        for index in 0..n {
            if active[index] != 0 {
                continue;
            }
            let direction = candidate[index] - current[index];
            let total = accumulated[index] + candidate[index];
            let limit = if total < bounds[index][0] {
                Some((bounds[index][0], -1))
            } else if total > bounds[index][1] {
                Some((bounds[index][1], 1))
            } else {
                None
            };
            if let Some((limit, side)) = limit {
                let fraction =
                    ((limit - accumulated[index] - current[index]) / direction).clamp(0.0, 1.0);
                if fraction <= step {
                    step = fraction;
                    blocker = Some((index, limit, side));
                }
            }
        }
        for index in 0..n {
            current[index] += step * (candidate[index] - current[index]);
        }
        if let Some((index, limit, side)) = blocker {
            current[index] = limit - accumulated[index];
            active[index] = side;
            continue;
        }
        let release = (0..n).find(|&index| {
            if active[index] == 0 || active[index] == 2 || bounds[index][0] == bounds[index][1] {
                return false;
            }
            let product: f64 = (0..n).map(|col| mass[index * n + col] * current[col]).sum();
            let scale = rhs[index].abs()
                + (0..n)
                    .map(|col| (mass[index * n + col] * current[col]).abs())
                    .sum::<f64>();
            let tolerance = tolerances[index].max(256.0 * f64::EPSILON * scale);
            let gradient = product - rhs[index];
            (active[index] == -1 && gradient < -tolerance)
                || (active[index] == 1 && gradient > tolerance)
        });
        if let Some(index) = release {
            active[index] = 0;
        } else {
            let total: Vec<_> = (0..n)
                .map(|index| {
                    if active[index] == -1 {
                        bounds[index][0]
                    } else if active[index] == 1 {
                        bounds[index][1]
                    } else {
                        accumulated[index] + current[index]
                    }
                })
                .collect();
            let delta = total
                .iter()
                .zip(accumulated)
                .map(|(total, old)| total - old)
                .collect();
            return Ok(BoundedRowSolution { delta, total });
        }
    }
}

/// Linearization of coincident attachment points.
pub(crate) fn point_rows(
    r1: Vec3,
    r2: Vec3,
    separation: Vec3,
    compliance: f32,
) -> [ConstraintRow; 3] {
    core::array::from_fn(|index| {
        let axis = [Vec3::X, Vec3::Y, Vec3::Z][index];
        ConstraintRow {
            linear1: -axis,
            linear2: axis,
            angular1: axis.cross(r1),
            angular2: r2.cross(axis),
            error: f64::from(separation[index]),
            compliance: f64::from(compliance),
            error_scale: coordinate_scale(&[r1, r2, separation]),
            ..Default::default()
        }
    })
}

// A rank-revealing symmetric elimination is required for valid locked axes.
// glam_matrix_extras' 6x6 LDLT divides by zero for these semidefinite systems.
// Null rows carry no mobility and therefore apply no physical correction.
#[cfg(feature = "3d")]
pub(crate) fn solve_rank_revealing(matrix: &mut [f64], rhs: &mut [f64], result: &mut [f64]) {
    let n = rhs.len();
    let mut order: smallvec::SmallVec<[usize; 6]> = (0..n).collect();
    let scale = (0..n).map(|i| matrix[i * n + i]).fold(0.0_f64, f64::max);
    let tolerance = 64.0 * f64::EPSILON * scale;
    result.fill(0.0);
    let mut rank = 0;
    for pivot in 0..n {
        let selected = (pivot..n)
            .max_by(|&a, &b| matrix[a * n + a].total_cmp(&matrix[b * n + b]))
            .unwrap();
        if matrix[selected * n + selected] <= tolerance {
            break;
        }
        for col in 0..n {
            matrix.swap(pivot * n + col, selected * n + col);
        }
        for row in 0..n {
            matrix.swap(row * n + pivot, row * n + selected);
        }
        rhs.swap(pivot, selected);
        order.swap(pivot, selected);
        for row in pivot + 1..n {
            let factor = matrix[row * n + pivot] / matrix[pivot * n + pivot];
            for col in pivot + 1..n {
                matrix[row * n + col] -= factor * matrix[pivot * n + col];
            }
            rhs[row] -= factor * rhs[pivot];
            matrix[row * n + pivot] = 0.0;
        }
        rank += 1;
    }
    for row in (0..rank).rev() {
        let tail: f64 = (row + 1..rank)
            .map(|col| matrix[row * n + col] * result[order[col]])
            .sum();
        result[order[row]] = (rhs[row] - tail) / matrix[row * n + row];
    }
}

#[cfg(all(test, feature = "3d"))]
mod tests {
    use super::{solve_bounded_rows, solve_rank_revealing};

    #[test]
    fn block_solves_mobile_rows_without_actuating_locked_directions() {
        let mut matrix = [[0.0; 6]; 6];
        matrix[0][0] = 2.0;
        matrix[2][2] = 4.0;
        matrix[3][3] = 3.0;
        matrix[0][3] = 1.0;
        matrix[3][0] = 1.0;
        let rhs = [3.0, 100.0, 8.0, 4.0, 200.0, 300.0];
        let mut factors = matrix;
        let mut work_rhs = rhs;
        let mut result = [0.0; 6];
        solve_rank_revealing(factors.as_flattened_mut(), &mut work_rhs, &mut result);
        assert_eq!([result[1], result[4], result[5]], [0.0; 3]);
        for row in [0, 2, 3] {
            let actual: f64 = (0..6).map(|col| matrix[row][col] * result[col]).sum();
            assert!((actual - rhs[row]).abs() < 1e-12);
        }
        let mut dependent = [[1.0; 6]; 6];
        let mut result = [0.0; 6];
        solve_rank_revealing(dependent.as_flattened_mut(), &mut [6.0; 6], &mut result);
        assert!(result.into_iter().all(f64::is_finite));
        assert!((result.into_iter().sum::<f64>() - 6.0).abs() < 1e-12);
    }
    #[test]
    fn unilateral_rows_release_bounds_after_coupled_contact_changes() {
        let mass = [
            0.6018144117658106,
            -0.15805984709194992,
            -0.2921046285108878,
            -0.15805984709194992,
            2.224201346667746,
            -0.3269675886694293,
            -0.2921046285108878,
            -0.3269675886694293,
            1.1857947835632103,
        ];
        let rhs = [
            -0.00006263387901439543,
            0.2467467550043163,
            -0.4124314008665573,
        ];
        let result = solve_bounded_rows(
            &mass,
            &rhs,
            &[[0.0, f64::INFINITY]; 3],
            &[0.0; 3],
            &[1e-12; 3],
        )
        .unwrap()
        .delta;
        assert!(result[0] > 0.0 && result[1] > 0.0 && result[2] == 0.0);
        for row in 0..3 {
            let gradient = (0..3)
                .map(|col| mass[row * 3 + col] * result[col])
                .sum::<f64>()
                - rhs[row];
            if result[row] > 0.0 {
                assert!(gradient.abs() < 1.0e-12);
            } else {
                assert!(gradient >= 0.0);
            }
        }
    }

    #[test]
    fn nonfinite_constraint_system_returns_a_typed_failure() {
        assert_eq!(
            solve_bounded_rows(
                &[f64::NAN],
                &[0.0],
                &[[0.0, f64::INFINITY]],
                &[0.0],
                &[1e-12]
            ),
            Err(crate::dynamics::solver::xpbd::ConstraintSolveError::NonFiniteSystem)
        );
    }
}
