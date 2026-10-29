use crate::dynamics::solver::solver_body::{SolverBody, SolverBodyInertia};
use crate::prelude::{AngularMotor, MotorModel};
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

/// Prepare hard velocity gradients and their mass-weighted responses once after
/// positional projection. Poses and inertias stay fixed during velocity relaxation.
pub(crate) fn prepare_velocity_rows(
    inertias: [&SolverBodyInertia; 2],
    rows: &[ConstraintRow; 6],
    cache: &mut VelocityRowCache,
) {
    let [ia, ib] = inertias;
    let ma = ia.effective_inv_mass();
    let mb = ib.effective_inv_mass();
    let aa = ia.effective_inv_angular_inertia();
    let ab = ib.effective_inv_angular_inertia();
    cache.motor = None;
    cache.gradients = core::array::from_fn(|i| {
        [
            rows[i].linear1,
            rows[i].linear2,
            rows[i].angular1,
            rows[i].angular2,
        ]
    });
    cache.responses = core::array::from_fn(|i| {
        [
            ma * rows[i].linear1,
            mb * rows[i].linear2,
            aa * rows[i].angular1,
            ab * rows[i].angular2,
        ]
    });
    let matrix = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            let row = cache.gradients[i];
            let response = cache.responses[j];
            f64::from(
                row[0].dot(response[0])
                    + row[1].dot(response[1])
                    + row[2].dot(response[2])
                    + row[3].dot(response[3]),
            )
        })
    });
    cache.factor = MobileMassFactor::new(matrix);
    cache.mass = matrix;
}

/// Pure velocity drives share the contact relaxation solve. Position springs
/// remain in XPBD; a velocity drive must have only one impulse owner.
pub(crate) fn velocity_drive_damping(motor: &AngularMotor) -> Option<f64> {
    if !motor.enabled {
        return None;
    }
    match motor.motor_model {
        MotorModel::ForceBased {
            stiffness: 0.0,
            damping,
        }
        | MotorModel::AccelerationBased {
            stiffness: 0.0,
            damping,
        } if damping > 0.0 => Some(f64::from(damping)),
        _ => None,
    }
}

/// The sixth row is the revolute joint's driven free axis. Its torque limit
/// bounds the accumulated substep impulse, including every contact/joint pass.
pub(crate) fn prepare_velocity_motor(
    cache: &mut VelocityRowCache,
    motor: &AngularMotor,
    dt: f32,
    initial_impulse: f64,
) {
    let Some(mut damping) = velocity_drive_damping(motor) else {
        return;
    };
    let mobility = cache.mass[5][5];
    if mobility <= 0.0 || dt <= 0.0 {
        return;
    }
    if matches!(motor.motor_model, MotorModel::AccelerationBased { .. }) {
        damping /= mobility;
    }
    let softness = 1.0 / (f64::from(dt) * damping);
    let mut hard_mass = cache.mass;
    for i in 0..6 {
        hard_mass[i][5] = 0.0;
        hard_mass[5][i] = 0.0;
    }
    let mut mass = cache.mass;
    mass[5][5] += softness;
    cache.factor = MobileMassFactor::new(mass);
    cache.motor = Some(VelocityMotor {
        target: f64::from(motor.target_velocity),
        cap: f64::from(motor.max_torque) * f64::from(dt),
        accumulated: initial_impulse,
        softness,
        hard_factor: MobileMassFactor::new(hard_mass),
    });
}

/// Project hard velocity gradients using their prepared constrained effective mass.
/// The impulse changes velocities only; it cannot repair or overwrite a pose.
pub(crate) fn solve_velocity_rows(
    bodies: [&mut SolverBody; 2],
    cache: &mut VelocityRowCache,
) -> [f32; 6] {
    let [a, b] = bodies;
    let mut rhs = core::array::from_fn(|i| {
        let row = cache.gradients[i];
        -f64::from(
            row[0].dot(a.linear_velocity)
                + row[1].dot(b.linear_velocity)
                + row[2].dot(a.angular_velocity)
                + row[3].dot(b.angular_velocity),
        )
    });
    if let Some(motor) = &cache.motor {
        rhs[5] += motor.target - motor.softness * motor.accumulated;
    }
    let mut result = cache.factor.solve(rhs);
    if let Some(motor) = &mut cache.motor {
        let bounded = result[5].clamp(
            -motor.cap - motor.accumulated,
            motor.cap - motor.accumulated,
        );
        if bounded != result[5] {
            for i in 0..6 {
                rhs[i] -= cache.mass[i][5] * bounded;
            }
            result = motor.hard_factor.solve(rhs);
            result[5] = bounded;
        }
        motor.accumulated += f64::from(result[5] as f32);
    }
    let result = result.map(|x| x as f32);
    for (response, lambda) in cache.responses.iter().zip(result) {
        a.linear_velocity += response[0] * lambda;
        b.linear_velocity += response[1] * lambda;
        a.angular_velocity += response[2] * lambda;
        b.angular_velocity += response[3] * lambda;
    }
    result
}

// A rank-revealing symmetric elimination is required for valid locked axes.
// glam_matrix_extras' 6x6 LDLT divides by zero for these semidefinite systems.
// Null rows carry no mobility and therefore apply no physical correction.
#[cfg(feature = "3d")]
pub(crate) fn solve_mobile_rows(matrix: [[f64; 6]; 6], rhs: [f64; 6]) -> [f64; 6] {
    MobileMassFactor::new(matrix).solve(rhs)
}

/// Pivoted LDLᵀ of the mobile constraint subspace; null rows exert no impulse.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct MobileMassFactor {
    matrix: [[f64; 6]; 6],
    order: [usize; 6],
    rank: usize,
}
impl MobileMassFactor {
    fn new(mut matrix: [[f64; 6]; 6]) -> Self {
        let mut order = [0, 1, 2, 3, 4, 5];
        let mut rank = 0;
        let scale = (0..6).map(|i| matrix[i][i]).fold(0.0_f64, f64::max);
        let tolerance = 64.0 * f64::EPSILON * scale;
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
            order.swap(pivot, selected);
            let diagonal = matrix[pivot][pivot];
            let mut factors = [0.0; 6];
            for row in pivot + 1..6 {
                factors[row] = matrix[row][pivot] / diagonal;
            }
            for row in pivot + 1..6 {
                for col in row..6 {
                    matrix[col][row] -= factors[row] * matrix[col][pivot];
                    matrix[row][col] = matrix[col][row];
                }
            }
            for row in pivot + 1..6 {
                matrix[row][pivot] = factors[row];
            }
            rank += 1;
        }
        Self {
            matrix,
            order,
            rank,
        }
    }
    fn solve(&self, rhs: [f64; 6]) -> [f64; 6] {
        let mut value = [0.0; 6];
        for row in 0..self.rank {
            value[row] = rhs[self.order[row]]
                - (0..row)
                    .map(|col| self.matrix[row][col] * value[col])
                    .sum::<f64>();
        }
        for (row, value) in value.iter_mut().enumerate().take(self.rank) {
            *value /= self.matrix[row][row];
        }
        for row in (0..self.rank).rev() {
            value[row] -= (row + 1..self.rank)
                .map(|col| self.matrix[col][row] * value[col])
                .sum::<f64>();
        }
        let mut result = [0.0; 6];
        for row in 0..self.rank {
            result[self.order[row]] = value[row];
        }
        result
    }
}

/// Substep-local gradients, responses and factorization for velocity relaxation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct VelocityRowCache {
    pub(crate) gradients: [[Vec3; 4]; 6],
    responses: [[Vec3; 4]; 6],
    factor: MobileMassFactor,
    mass: [[f64; 6]; 6],
    motor: Option<VelocityMotor>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct VelocityMotor {
    target: f64,
    cap: f64,
    accumulated: f64,
    softness: f64,
    hard_factor: MobileMassFactor,
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
