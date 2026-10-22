//! Coupled native joint frames and unilateral contact support.
//!
//! Island connectivity is owned by the native persistent island graph. Static
//! surfaces do not connect unrelated dynamic assemblies.
use super::{
    fixed,
    shared::{ConstraintRow, solve_bounded_rows},
    *,
};
use crate::dynamics::{
    joints::EntityConstraint,
    solver::xpbd::{ConstraintSolveError, ConstraintSolveFailure, XpbdSolverStatus},
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
}
impl Block {
    fn rows(&mut self, bodies: [&SolverBody; 2]) -> [ConstraintRow; 6] {
        let [a, b] = bodies;
        match &self.source {
            Source::Fixed(j, d) => {
                fixed::frame_rows(a, b, d, j.point_compliance, j.angle_compliance)
            }
            Source::Slider(j, d) => j.frame_rows(a, b, d, &mut self.bound),
            Source::Hinge(j, d) => j.frame_rows(a, b, d, &mut self.bound),
        }
    }
    fn record(&mut self, rows: &[SpatialRow], update: &[f64]) {
        for (index, (row, impulse)) in rows.iter().zip(update).enumerate() {
            let impulse = *impulse as f32;
            let linear = row.row.linear2 * impulse;
            let angular = row.row.angular2 * impulse;
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
    // Apply each actuator once. Frame/contact refinements must not reapply a motor.
    for block in &mut blocks {
        let mut dummy1 = SolverBody::DUMMY;
        let mut dummy2 = SolverBody::DUMMY;
        let access = bodies.access();
        // SAFETY: distinct valid indices were checked above; blocks run serially.
        let (a, b) = unsafe { access.get_pair_unchecked_mut(block.bodies[0], block.bodies[1]) };
        let (a, mut ia) = a.unwrap_or((&mut dummy1, &SolverBodyInertia::DUMMY));
        let (b, mut ib) = b.unwrap_or((&mut dummy2, &SolverBodyInertia::DUMMY));
        if ia.dominance() > ib.dominance() {
            ia = &SolverBodyInertia::DUMMY;
        } else if ib.dominance() > ia.dominance() {
            ib = &SolverBodyInertia::DUMMY;
        }
        match &mut block.source {
            Source::Fixed(_, _) => {}
            Source::Slider(j, d) => j.apply_motor(a, b, ia, ib, d, dt),
            Source::Hinge(j, d) => j.apply_motor(
                a,
                b,
                ia.effective_inv_angular_inertia(),
                ib.effective_inv_angular_inertia(),
                d,
                dt,
            ),
        }
    }
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
        let count = group.blocks.len() * 6
            + group
                .contacts
                .iter()
                .map(|i| graph.colors[i.0].contact_constraints[i.1].points.len())
                .sum::<usize>();
        let mut accumulated = vec![0.0; count];
        for iteration in 0..=4 {
            let mut rows = Vec::with_capacity(count);
            for index in &group.blocks {
                let block = &mut blocks[*index];
                let a = bodies.get(block.bodies[0]).unwrap_or(&SolverBody::DUMMY);
                let b = bodies.get(block.bodies[1]).unwrap_or(&SolverBody::DUMMY);
                for row in block.rows([a, b]) {
                    rows.push(SpatialRow {
                        bodies: block.bodies,
                        row,
                        mobility: [true; 2],
                    });
                }
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
                            error: separation,
                            bounds: [0.0, f64::INFINITY],
                            ..default()
                        },
                        mobility: [true; 2],
                    });
                }
            }
            // Dominance removes mobility from the dominant endpoint only.
            for row in &mut rows {
                let a = bodies
                    .get_inertia(row.bodies[0])
                    .unwrap_or(&SolverBodyInertia::DUMMY);
                let b = bodies
                    .get_inertia(row.bodies[1])
                    .unwrap_or(&SolverBodyInertia::DUMMY);
                if a.dominance() > b.dominance() {
                    row.mobility[0] = false;
                } else if b.dominance() > a.dominance() {
                    row.mobility[1] = false;
                }
            }
            // Stop nonlinear refinement once every active scalar constraint is
            // satisfied to native f32 precision. Unloaded contacts may separate.
            let tolerance = f64::from(8.0 * f32::EPSILON);
            let converged = rows.iter().zip(&accumulated).all(|(row, &lambda)| {
                let residual = f64::from(row.row.error)
                    + f64::from(row.row.compliance) * lambda / f64::from(dt).powi(2);
                if lambda == row.row.bounds[0] {
                    residual >= -tolerance
                } else if lambda == row.row.bounds[1] {
                    residual <= tolerance
                } else {
                    residual.abs() <= tolerance
                }
            });
            if converged {
                break;
            }
            if iteration == 4 {
                status.failure = Some(ConstraintSolveFailure {
                    constraints: group
                        .blocks
                        .iter()
                        .map(|index| blocks[*index].entity)
                        .collect(),
                    reason: ConstraintSolveError::NonlinearIterationLimit {
                        iterations: iteration,
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
                        reason,
                    });
                    physics_time.pause();
                    return;
                }
            };
            for (slot, index) in group.blocks.iter().enumerate() {
                blocks[*index].record(
                    &rows[slot * 6..slot * 6 + 6],
                    &update[slot * 6..slot * 6 + 6],
                );
            }
            let mut cursor = group.blocks.len() * 6;
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
    let terms = |row: &SpatialRow| {
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
    };
    for (i, row) in rows.iter().enumerate() {
        for (j, other) in rows.iter().enumerate().take(i + 1) {
            let mut value = 0.0;
            for (body, linear, angular) in terms(row) {
                if let Some(inertia) = bodies.get_inertia(body) {
                    for (other_body, other_linear, other_angular) in terms(other) {
                        if body == other_body {
                            value += linear
                                .dot(inertia.effective_inv_mass().as_dvec3() * other_linear)
                                + angular.dot(
                                    Mat3::from(inertia.effective_inv_angular_inertia()).as_dmat3()
                                        * other_angular,
                                );
                        }
                    }
                }
            }
            mass[i * n + j] = value;
            mass[j * n + i] = value;
        }
        let alpha = f64::from(row.row.compliance) / f64::from(dt).powi(2);
        mass[i * n + i] += alpha;
        rhs[i] = -f64::from(row.row.error) - alpha * accumulated[i];
    }
    let bounds: Vec<_> = rows.iter().map(|row| row.row.bounds).collect();
    let update = solve_bounded_rows(&mass, &rhs, &bounds, accumulated)?;
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
        accumulated[index] += update[index];
        for (body, linear, angular) in terms(row) {
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
    Ok(update)
}
