//! Coupled native joint frames and unilateral contact support.
//!
//! Island connectivity is owned by the native persistent island graph. Static
//! surfaces do not connect unrelated dynamic assemblies.
use super::{
    fixed,
    shared::{
        ConstraintRow, MotorConstraint, MotorInput, coordinate_scale, jacobian_dot,
        projected_inverse_mass, solve_bounded_rows,
    },
    *,
};
use crate::dynamics::{
    joints::EntityConstraint,
    solver::xpbd::{
        ConstraintResidualKind, ConstraintSolveError, ConstraintSolveFailure, XpbdSolverStatus,
    },
};
use crate::{
    dynamics::solver::{
        constraint_graph::ConstraintGraph,
        islands::BodyIslandNode,
        solver_body::{SolverBodies, SolverBody, SolverBodyIndex, SolverBodyInertia},
    },
    prelude::*,
};
use bevy::{math::DVec3, prelude::*};
use std::collections::BTreeMap;

#[derive(Clone)]
enum Source {
    Fixed(FixedJoint, FixedJointSolverData),
    Slider(PrismaticJoint, PrismaticJointSolverData),
    Hinge(RevoluteJoint, RevoluteJointSolverData),
}
struct Block {
    entity: Entity,
    bodies: [SolverBodyIndex; 2],
    source: Source,
    bound: Option<(f32, [f64; 2])>,
    motor: Option<MotorConstraint>,
}
impl Block {
    fn frame_rows(&mut self, bodies: [&SolverBody; 2]) -> [ConstraintRow; 6] {
        let [a, b] = bodies;
        match &self.source {
            Source::Fixed(j, d) => {
                fixed::frame_rows(a, b, d, j.point_compliance, j.angle_compliance)
            }
            Source::Slider(j, d) => j.frame_rows(a, b, d, &mut self.bound),
            Source::Hinge(j, d) => j.frame_rows(a, b, d, &mut self.bound),
        }
    }
    fn motor_input(&self, bodies: [&SolverBody; 2]) -> Option<MotorInput> {
        match &self.source {
            Source::Fixed(_, _) => None,
            Source::Slider(j, d) => j.motor_input(bodies[0], bodies[1], d),
            Source::Hinge(j, d) => j.motor_input(bodies[0], bodies[1], d),
        }
    }
    fn rows(&mut self, bodies: [&SolverBody; 2]) -> impl Iterator<Item = ConstraintRow> + use<> {
        let frames = self.frame_rows(bodies);
        let motor = self.motor.as_ref().map(|motor| {
            motor.row(
                self.motor_input(bodies)
                    .expect("prepared motor configuration is immutable during a solve"),
            )
        });
        frames.into_iter().chain(motor)
    }
    fn record(&mut self, rows: &[SpatialRow], update: &[f64]) {
        for (index, (row, impulse)) in rows.iter().zip(update).enumerate() {
            let impulse = *impulse as f32;
            let linear = row.row.linear2 * impulse;
            let angular = row.row.angular2 * impulse;
            if index == 6 {
                match &mut self.source {
                    Source::Fixed(_, _) => unreachable!("fixed joints have no motor row"),
                    Source::Slider(_, d) => {
                        d.total_motor_lagrange += impulse;
                        d.total_position_lagrange -= linear;
                    }
                    Source::Hinge(_, d) => d.total_motor_lagrange -= angular,
                }
                continue;
            }
            match &mut self.source {
                Source::Fixed(_, d) => {
                    if index < 3 {
                        d.point_constraint.total_lagrange -= linear;
                    } else {
                        d.angle_constraint.total_lagrange -= angular;
                    }
                }
                Source::Slider(_, d) => {
                    if index < 3 {
                        d.total_position_lagrange -= linear;
                    } else {
                        d.angle_constraint.total_lagrange -= angular;
                    }
                }
                Source::Hinge(_, d) => {
                    if index < 3 {
                        d.point_constraint.total_lagrange -= linear;
                    } else if index < 5 {
                        d.total_align_lagrange -= angular;
                    } else {
                        d.total_limit_lagrange -= angular;
                    }
                }
            }
        }
    }
}
struct SpatialRow {
    bodies: [SolverBodyIndex; 2],
    row: ConstraintRow,
    mobility: [bool; 2],
}
impl SpatialRow {
    fn terms(&self) -> [(SolverBodyIndex, DVec3, DVec3); 2] {
        let row = self;
        [
            (
                row.bodies[0],
                if row.mobility[0] {
                    row.row.linear1.as_dvec3()
                } else {
                    DVec3::ZERO
                },
                if row.mobility[0] {
                    row.row.angular1.as_dvec3()
                } else {
                    DVec3::ZERO
                },
            ),
            (
                row.bodies[1],
                if row.mobility[1] {
                    row.row.linear2.as_dvec3()
                } else {
                    DVec3::ZERO
                },
                if row.mobility[1] {
                    row.row.angular2.as_dvec3()
                } else {
                    DVec3::ZERO
                },
            ),
        ]
    }
    fn apply_dominance(&mut self, bodies: &SolverBodies) {
        let a = bodies
            .get_inertia(self.bodies[0])
            .unwrap_or(&SolverBodyInertia::DUMMY);
        let b = bodies
            .get_inertia(self.bodies[1])
            .unwrap_or(&SolverBodyInertia::DUMMY);
        self.mobility = [
            a.dominance() <= b.dominance(),
            b.dominance() <= a.dominance(),
        ];
    }
}
fn inverse_mass_entry(bodies: &SolverBodies, a: &SpatialRow, b: &SpatialRow) -> f64 {
    let mut value = 0.0;
    for (body, linear, angular) in a.terms() {
        if let Some(inertia) = bodies.get_inertia(body) {
            for (other, other_linear, other_angular) in b.terms() {
                if body == other {
                    value +=
                        jacobian_dot((linear, angular), (other_linear, other_angular), inertia);
                }
            }
        }
    }
    value
}
fn projected_mobility(
    bodies: &SolverBodies,
    rows: &[SpatialRow],
    motor: &SpatialRow,
    dt: f32,
) -> Result<f64, ConstraintSolveError> {
    let n = rows.len();
    let mut mass = vec![0.0; n * n];
    let mut coupling = vec![0.0; n];
    for (i, row) in rows.iter().enumerate() {
        for (j, other) in rows.iter().enumerate() {
            mass[i * n + j] = inverse_mass_entry(bodies, row, other);
        }
        mass[i * n + i] += row.row.compliance / f64::from(dt).powi(2);
        coupling[i] = inverse_mass_entry(bodies, row, motor);
    }
    projected_inverse_mass(&mass, &coupling, inverse_mass_entry(bodies, motor, motor))
}

#[derive(Default)]
struct Group {
    blocks: Vec<usize>,
    contacts: Vec<(usize, usize)>,
}

/// One native solve for every connected assembly, including its contact support.
pub(crate) fn solve_joint_contact_islands(
    mut bodies: ResMut<SolverBodies>,
    indices: Query<Option<&SolverBodyIndex>, (With<RigidBody>, Without<RigidBodyDisabled>)>,
    island_nodes: Query<&BodyIslandNode>,
    mut fixed: Query<
        (Entity, &FixedJoint, &mut FixedJointSolverData),
        (Without<RigidBody>, Without<JointDisabled>),
    >,
    mut sliders: Query<
        (Entity, &PrismaticJoint, &mut PrismaticJointSolverData),
        (Without<RigidBody>, Without<JointDisabled>),
    >,
    mut hinges: Query<
        (Entity, &RevoluteJoint, &mut RevoluteJointSolverData),
        (Without<RigidBody>, Without<JointDisabled>),
    >,
    mut graph: ResMut<ConstraintGraph>,
    time: Res<Time>,
    mut physics_time: ResMut<Time<Physics>>,
    mut status: ResMut<XpbdSolverStatus>,
) {
    if status.failure.is_some() {
        return;
    }
    let dt = time.delta_secs();
    let mut blocks = Vec::new();
    let mut add = |entity, endpoints, source| {
        let Ok(pair) = indices.get_many(endpoints) else {
            return;
        };
        let pair = pair.map(|i| i.copied().unwrap_or(SolverBodyIndex::INVALID));
        if pair[0] != pair[1] {
            blocks.push(Block {
                entity,
                bodies: pair,
                source,
                bound: None,
                motor: None,
            });
        }
    };
    for (entity, joint, data) in &mut fixed {
        add(
            entity,
            joint.entities(),
            Source::Fixed(joint.clone(), *data),
        );
    }
    for (entity, joint, data) in &mut sliders {
        add(
            entity,
            joint.entities(),
            Source::Slider(joint.clone(), *data),
        );
    }
    for (entity, joint, data) in &mut hinges {
        add(
            entity,
            joint.entities(),
            Source::Hinge(joint.clone(), *data),
        );
    }
    if blocks.is_empty() {
        return;
    }
    blocks.sort_by_key(|b| b.entity);
    let body_islands: Vec<u32> = bodies
        .entities()
        .iter()
        .map(|entity| {
            island_nodes
                .get(*entity)
                .expect("native XPBD assemblies require IslandPlugin")
                .island_id()
                .0
        })
        .collect();
    let group_for = |pair: [SolverBodyIndex; 2]| {
        pair.into_iter()
            .filter(|index| {
                bodies
                    .get(*index)
                    .is_some_and(|body| body.flags.is_dynamic())
            })
            .min_by_key(|index| bodies.get_inertia(*index).unwrap().dominance())
            .and_then(|index| body_islands.get(index.0 as usize).copied())
    };
    let mut groups: BTreeMap<u32, Group> = BTreeMap::new();
    for (index, block) in blocks.iter().enumerate() {
        if let Some(group) = group_for(block.bodies) {
            groups.entry(group).or_default().blocks.push(index);
        }
    }
    for (color_index, color) in graph.colors.iter().enumerate() {
        for (index, contact) in color.contact_constraints.iter().enumerate() {
            if let Some(group) = group_for([contact.body_index1, contact.body_index2])
                && let Some(group) = groups.get_mut(&group)
            {
                group.contacts.push((color_index, index));
            }
        }
    }
    for group in groups.values() {
        // Normalize drives against the admitted bilateral mechanism, including
        // loads carried through other joints, before nonlinear refinement.
        let needs_mobility = group.blocks.iter().any(|index| {
            let block = &blocks[*index];
            block
                .motor_input([
                    bodies.get(block.bodies[0]).unwrap_or(&SolverBody::DUMMY),
                    bodies.get(block.bodies[1]).unwrap_or(&SolverBody::DUMMY),
                ])
                .is_some_and(|input| input.normalized())
        });
        let mut mechanism = Vec::new();
        for index in group.blocks.iter().filter(|_| needs_mobility) {
            let block = &mut blocks[*index];
            let a = bodies.get(block.bodies[0]).unwrap_or(&SolverBody::DUMMY);
            let b = bodies.get(block.bodies[1]).unwrap_or(&SolverBody::DUMMY);
            for row in block.frame_rows([a, b]) {
                let mut row = SpatialRow {
                    bodies: block.bodies,
                    row,
                    mobility: [true; 2],
                };
                row.apply_dominance(&bodies);
                if row.row.bounds == [f64::NEG_INFINITY, f64::INFINITY] {
                    mechanism.push(row);
                }
            }
        }
        for index in &group.blocks {
            let block = &mut blocks[*index];
            let a = bodies.get(block.bodies[0]).unwrap_or(&SolverBody::DUMMY);
            let b = bodies.get(block.bodies[1]).unwrap_or(&SolverBody::DUMMY);
            if let Some(input) = block.motor_input([a, b]) {
                let mut motor_row = SpatialRow {
                    bodies: block.bodies,
                    row: input.jacobian.clone(),
                    mobility: [true; 2],
                };
                motor_row.apply_dominance(&bodies);
                let inverse_mass = if input.normalized() {
                    projected_mobility(&bodies, &mechanism, &motor_row, dt)
                } else {
                    Ok(inverse_mass_entry(&bodies, &motor_row, &motor_row))
                };
                let prepared = inverse_mass.and_then(|mass| input.prepare(mass, dt));
                match prepared {
                    Ok(motor) => block.motor = motor,
                    Err(reason) => {
                        status.failure = Some(ConstraintSolveFailure {
                            constraints: vec![block.entity],
                            primary_entity: Some(block.entity),
                            reason,
                        });
                        physics_time.pause();
                        return;
                    }
                }
            }
        }
        let count = group
            .blocks
            .iter()
            .map(|index| 6 + usize::from(blocks[*index].motor.is_some()))
            .sum::<usize>()
            + group
                .contacts
                .iter()
                .map(|i| graph.colors[i.0].contact_constraints[i.1].points.len())
                .sum::<usize>();
        let mut accumulated = vec![0.0; count];
        for iteration in 0..=4 {
            let mut rows = Vec::with_capacity(count);
            let mut ranges = Vec::with_capacity(group.blocks.len());
            for index in &group.blocks {
                let block = &mut blocks[*index];
                let a = bodies.get(block.bodies[0]).unwrap_or(&SolverBody::DUMMY);
                let b = bodies.get(block.bodies[1]).unwrap_or(&SolverBody::DUMMY);
                let start = rows.len();
                for row in block.rows([a, b]) {
                    rows.push(SpatialRow {
                        bodies: block.bodies,
                        row,
                        mobility: [true; 2],
                    });
                }
                ranges.push(start..rows.len());
            }
            for index in &group.contacts {
                let contact = &graph.colors[index.0].contact_constraints[index.1];
                let a = bodies
                    .get(contact.body_index1)
                    .unwrap_or(&SolverBody::DUMMY);
                let b = bodies
                    .get(contact.body_index2)
                    .unwrap_or(&SolverBody::DUMMY);
                for point in &contact.points {
                    let r1 = a.delta_rotation * point.anchor1;
                    let r2 = b.delta_rotation * point.anchor2;
                    let separation = (b.delta_position - a.delta_position + r2 - r1)
                        .dot(contact.normal)
                        + point.initial_separation;
                    rows.push(SpatialRow {
                        bodies: [contact.body_index1, contact.body_index2],
                        row: ConstraintRow {
                            linear1: -contact.normal,
                            linear2: contact.normal,
                            angular1: contact.normal.cross(r1),
                            angular2: r2.cross(contact.normal),
                            error: f64::from(separation),
                            error_scale: coordinate_scale(&[
                                r1,
                                r2,
                                a.delta_position,
                                b.delta_position,
                            ]) + f64::from(point.initial_separation.abs()),
                            kind: ConstraintResidualKind::ContactNormal,
                            bounds: [0.0, f64::INFINITY],
                            ..default()
                        },
                        mobility: [true; 2],
                    });
                }
            }
            for row in &mut rows {
                row.apply_dominance(&bodies);
            }
            let worst = rows.iter().zip(&accumulated).enumerate().max_by(|a, b| {
                (a.1.0.row.violation(*a.1.1, dt) / a.1.0.row.tolerance())
                    .total_cmp(&(b.1.0.row.violation(*b.1.1, dt) / b.1.0.row.tolerance()))
            });
            let converged = rows
                .iter()
                .zip(&accumulated)
                .all(|(row, &lambda)| row.row.violation(lambda, dt) <= row.row.tolerance());
            // A drive must transmit effort every substep, even when its predicted
            // coordinate change is smaller than the position residual tolerance.
            if converged
                && (iteration != 0
                    || group
                        .blocks
                        .iter()
                        .all(|index| blocks[*index].motor.is_none()))
            {
                break;
            }
            if iteration == 4 {
                let (row, (failed, &lambda)) =
                    worst.expect("a non-convergent island has constraint rows");
                let primary_entity = ranges
                    .iter()
                    .position(|range| range.contains(&row))
                    .map(|slot| blocks[group.blocks[slot]].entity)
                    .or_else(|| {
                        failed
                            .bodies
                            .iter()
                            .find_map(|index| bodies.get_entity(*index))
                    });
                status.failure = Some(ConstraintSolveFailure {
                    constraints: group
                        .blocks
                        .iter()
                        .map(|index| blocks[*index].entity)
                        .collect(),
                    primary_entity,
                    reason: ConstraintSolveError::NonlinearIterationLimit {
                        iterations: iteration,
                        row,
                        residual: failed.row.residual(lambda, dt),
                        tolerance: failed.row.tolerance(),
                        kind: failed.row.kind,
                    },
                });
                physics_time.pause();
                return;
            }
            let update = match solve_rows(&mut bodies, &rows, &mut accumulated, dt) {
                Ok(update) => update,
                Err(reason) => {
                    status.failure = Some(ConstraintSolveFailure {
                        constraints: group
                            .blocks
                            .iter()
                            .map(|index| blocks[*index].entity)
                            .collect(),
                        primary_entity: None,
                        reason,
                    });
                    physics_time.pause();
                    return;
                }
            };
            for (slot, index) in group.blocks.iter().enumerate() {
                let range = ranges[slot].clone();
                blocks[*index].record(&rows[range.clone()], &update[range]);
            }
            let mut cursor = ranges.last().map_or(0, |range| range.end);
            for index in &group.contacts {
                for point in &mut graph.colors[index.0].contact_constraints[index.1].points {
                    let impulse = update[cursor] as f32 / dt;
                    point.normal_part.impulse += impulse;
                    point.normal_part.total_impulse += impulse;
                    cursor += 1;
                }
            }
        }
    }
    for block in blocks {
        match block.source {
            Source::Fixed(_, d) => *fixed.get_mut(block.entity).unwrap().2 = d,
            Source::Slider(_, d) => *sliders.get_mut(block.entity).unwrap().2 = d,
            Source::Hinge(_, d) => *hinges.get_mut(block.entity).unwrap().2 = d,
        }
    }
}

fn solve_rows(
    bodies: &mut SolverBodies,
    rows: &[SpatialRow],
    accumulated: &mut [f64],
    dt: f32,
) -> Result<Vec<f64>, ConstraintSolveError> {
    let n = rows.len();
    let mut mass = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    for (i, row) in rows.iter().enumerate() {
        for (j, other) in rows.iter().enumerate().take(i + 1) {
            let value = inverse_mass_entry(bodies, row, other);
            mass[i * n + j] = value;
            mass[j * n + i] = value;
        }
        let alpha = f64::from(row.row.compliance) / f64::from(dt).powi(2);
        mass[i * n + i] += alpha;
        rhs[i] = -f64::from(row.row.error) - alpha * accumulated[i];
    }
    let bounds: Vec<_> = rows.iter().map(|row| row.row.bounds).collect();
    let tolerances: Vec<_> = rows.iter().map(|row| row.row.tolerance()).collect();
    let solution = solve_bounded_rows(&mass, &rhs, &bounds, accumulated, &tolerances)?;
    let update = &solution.delta;
    if update.iter().zip(rows).any(|(&impulse, row)| {
        !impulse.is_finite()
            || impulse.abs() > f64::from(f32::MAX)
            || !(row.row.linear1 * impulse as f32).is_finite()
            || !(row.row.linear2 * impulse as f32).is_finite()
            || !(row.row.angular1 * impulse as f32).is_finite()
            || !(row.row.angular2 * impulse as f32).is_finite()
    }) {
        return Err(ConstraintSolveError::NonFiniteCorrection);
    }
    let mut touched: smallvec::SmallVec<[SolverBodyIndex; 16]> = rows
        .iter()
        .flat_map(|row| row.bodies)
        .filter(|index| bodies.contains_index(*index))
        .collect();
    touched.sort_unstable_by_key(|index| index.0);
    touched.dedup();
    let mut position: smallvec::SmallVec<[Vec3; 16]> =
        core::iter::repeat_n(Vec3::ZERO, touched.len()).collect();
    let mut rotation = position.clone();
    for (index, row) in rows.iter().enumerate() {
        accumulated[index] = solution.total[index];
        for (body, linear, angular) in row.terms() {
            if let Some(inertia) = bodies.get_inertia(body) {
                let impulse = update[index];
                position[touched
                    .binary_search_by_key(&body.0, |index| index.0)
                    .unwrap()] +=
                    (inertia.effective_inv_mass().as_dvec3() * linear * impulse).as_vec3();
                rotation[touched
                    .binary_search_by_key(&body.0, |index| index.0)
                    .unwrap()] += (Mat3::from(inertia.effective_inv_angular_inertia()).as_dmat3()
                    * angular
                    * impulse)
                    .as_vec3();
            }
        }
    }
    if position
        .iter()
        .chain(rotation.iter())
        .any(|correction| !correction.is_finite())
    {
        return Err(ConstraintSolveError::NonFiniteCorrection);
    }
    for (slot, index) in touched.into_iter().enumerate() {
        let body = bodies.get_mut(index).unwrap();
        body.delta_position += position[slot];
        if rotation[slot] != Vec3::ZERO {
            body.delta_rotation =
                (Quat::from_scaled_axis(rotation[slot]) * body.delta_rotation).normalize();
        }
    }
    Ok(solution.delta)
}
