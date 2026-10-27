use core::cmp::Ordering;

use super::joints::*;
use crate::{
    dynamics::{
        joints::EntityConstraint,
        solver::{
            schedule::SubstepSolverSystems,
            solver_body::{SolverBodies, SolverBody, SolverBodyIndex, SolverBodyInertia},
            xpbd::{XpbdConstraint, XpbdConstraintSolverData},
        },
    },
    prelude::*,
};
use bevy::{ecs::component::Mutable, prelude::*};

/// A plugin for a joint solver using Extended Position-Based Dynamics (XPBD).
pub struct XpbdSolverPlugin;

impl Plugin for XpbdSolverPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<XpbdVelocityProjection>();
        #[cfg(feature = "3d")]
        {
            app.init_resource::<JointVelocitySolverConfig>()
                .register_type::<JointVelocitySolverConfig>();
            app.add_systems(
                SubstepSchedule,
                relax_joint_contact_velocities
                    .after(XpbdSolverSystems::VelocityProjection)
                    .before(SubstepSolverSystems::Damping)
                    .run_if(has_rigid_joint_velocity_constraints),
            );
        }

        app.register_required_components::<FixedJoint, FixedJointSolverData>();
        app.register_required_components::<RevoluteJoint, RevoluteJointSolverData>();
        #[cfg(feature = "3d")]
        app.register_required_components::<SphericalJoint, SphericalJointSolverData>();
        app.register_required_components::<PrismaticJoint, PrismaticJointSolverData>();
        app.register_required_components::<DistanceJoint, DistanceJointSolverData>();

        // Configure scheduling.
        app.configure_sets(
            SubstepSchedule,
            (
                XpbdSolverSystems::SolveConstraints,
                XpbdSolverSystems::SolveUserConstraints,
                XpbdSolverSystems::VelocityProjection,
            )
                .chain()
                .after(SubstepSolverSystems::Relax)
                .before(SubstepSolverSystems::Damping),
        );

        // Prepare joints before the substepping loop.
        app.add_systems(
            PhysicsSchedule,
            (
                prepare_xpbd_joint::<FixedJoint>,
                prepare_xpbd_joint::<RevoluteJoint>,
                #[cfg(feature = "3d")]
                prepare_xpbd_joint::<SphericalJoint>,
                prepare_xpbd_joint::<PrismaticJoint>,
                prepare_xpbd_joint::<DistanceJoint>,
            )
                .chain()
                .in_set(SolverSystems::PrepareJoints),
        );

        // Solve joints with XPBD.
        app.add_systems(
            SubstepSchedule,
            (
                store_pre_solve_deltas,
                solve_xpbd_joint::<FixedJoint>,
                solve_xpbd_joint::<RevoluteJoint>,
                #[cfg(feature = "3d")]
                solve_xpbd_joint::<SphericalJoint>,
                solve_xpbd_joint::<PrismaticJoint>,
                solve_xpbd_joint::<DistanceJoint>,
            )
                .chain()
                .in_set(XpbdSolverSystems::SolveConstraints),
        );

        // Perform XPBD velocity updates after constraint solving.
        app.add_systems(
            SubstepSchedule,
            (project_linear_velocity, project_angular_velocity)
                .chain()
                .in_set(XpbdSolverSystems::VelocityProjection),
        );

        // Write back the forces applied by the XPBD joints.
        app.add_systems(
            PhysicsSchedule,
            (
                writeback_joint_forces::<FixedJoint>,
                writeback_joint_forces::<RevoluteJoint>,
                #[cfg(feature = "3d")]
                writeback_joint_forces::<SphericalJoint>,
                writeback_joint_forces::<PrismaticJoint>,
                writeback_joint_forces::<DistanceJoint>,
            )
                .chain()
                .in_set(SolverSystems::Finalize),
        );
    }
}

/// Number of coupled joint/contact velocity relaxation passes per substep.
#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct JointVelocitySolverConfig {
    /// Fixed number of contact/joint passes, in addition to the native relaxation pass.
    pub iterations: core::num::NonZeroU8,
}
impl Default for JointVelocitySolverConfig {
    fn default() -> Self {
        Self {
            iterations: core::num::NonZeroU8::new(16).unwrap(),
        }
    }
}
#[cfg(feature = "3d")]
fn has_rigid_joint_velocity_constraints(
    joints: Query<
        (
            Option<&FixedJoint>,
            Option<&PrismaticJoint>,
            Option<&RevoluteJoint>,
        ),
        (
            Or<(With<FixedJoint>, With<PrismaticJoint>, With<RevoluteJoint>)>,
            Without<JointDisabled>,
        ),
    >,
    active: Query<(&RigidBody, &SolverBodyIndex), (Without<Sleeping>, Without<RigidBodyDisabled>)>,
) -> bool {
    joints.iter().any(|(fixed, slider, hinge)| {
        let bodies = if let Some(joint) = fixed {
            joint.entities()
        } else if let Some(joint) = slider {
            joint.entities()
        } else {
            hinge.unwrap().entities()
        };
        bodies.into_iter().any(|body| {
            active
                .get(body)
                .is_ok_and(|(body, index)| body.is_dynamic() && index.is_valid())
        })
    })
}

#[cfg(feature = "3d")]
fn relax_joint_contact_velocities(
    mut solver_bodies: ResMut<SolverBodies>,
    body_indices: Query<Option<&SolverBodyIndex>, (With<RigidBody>, Without<RigidBodyDisabled>)>,
    mut fixed: Query<
        (&mut FixedJoint, &mut FixedJointSolverData),
        (Without<RigidBody>, Without<JointDisabled>),
    >,
    mut sliders: Query<
        (&mut PrismaticJoint, &mut PrismaticJointSolverData),
        (Without<RigidBody>, Without<JointDisabled>),
    >,
    mut hinges: Query<
        (&mut RevoluteJoint, &mut RevoluteJointSolverData),
        (Without<RigidBody>, Without<JointDisabled>),
    >,
    mut graph: ResMut<crate::dynamics::solver::constraint_graph::ConstraintGraph>,
    islands: Res<crate::dynamics::solver::islands::PhysicsIslands>,
    body_islands: Query<&crate::dynamics::solver::islands::BodyIslandNode>,
    config: Res<crate::dynamics::solver::SolverConfig>,
    iterations: Res<JointVelocitySolverConfig>,
    length_unit: Res<PhysicsLengthUnit>,
    time: Res<Time>,
    mut diagnostics: ResMut<crate::dynamics::solver::SolverDiagnostics>,
) {
    let dt = time.delta_secs();
    let mut coupled_islands: smallvec::SmallVec<[_; 8]> = fixed
        .iter()
        .map(|(joint, _)| joint.entities())
        .chain(sliders.iter().map(|(joint, _)| joint.entities()))
        .chain(hinges.iter().map(|(joint, _)| joint.entities()))
        .flatten()
        .filter_map(|body| body_islands.get(body).ok().map(|node| node.island_id()))
        .collect();
    coupled_islands.sort_unstable();
    coupled_islands.dedup();
    // All relaxation passes share one component borrow and one joint order.
    // No structural changes can occur until these query items are released.
    let mut fixed: Vec<_> = fixed
        .iter_mut()
        .sort_unstable_by_key::<Entity, _>(|entity| entity.index())
        .collect();
    let mut sliders: Vec<_> = sliders
        .iter_mut()
        .sort_unstable_by_key::<Entity, _>(|entity| entity.index())
        .collect();
    let mut hinges: Vec<_> = hinges
        .iter_mut()
        .sort_unstable_by_key::<Entity, _>(|entity| entity.index())
        .collect();
    apply_xpbd_joint_pass::<FixedJoint>(
        &mut solver_bodies,
        &body_indices,
        &mut fixed,
        dt,
        JointSolverPhase::PrepareVelocities,
    );
    apply_xpbd_joint_pass::<PrismaticJoint>(
        &mut solver_bodies,
        &body_indices,
        &mut sliders,
        dt,
        JointSolverPhase::PrepareVelocities,
    );
    apply_xpbd_joint_pass::<RevoluteJoint>(
        &mut solver_bodies,
        &body_indices,
        &mut hinges,
        dt,
        JointSolverPhase::PrepareVelocities,
    );
    apply_xpbd_joint_pass::<FixedJoint>(
        &mut solver_bodies,
        &body_indices,
        &mut fixed,
        dt,
        JointSolverPhase::ProjectVelocities,
    );
    apply_xpbd_joint_pass::<PrismaticJoint>(
        &mut solver_bodies,
        &body_indices,
        &mut sliders,
        dt,
        JointSolverPhase::ProjectVelocities,
    );
    apply_xpbd_joint_pass::<RevoluteJoint>(
        &mut solver_bodies,
        &body_indices,
        &mut hinges,
        dt,
        JointSolverPhase::ProjectVelocities,
    );
    for _ in 0..iterations.iterations.get() {
        let start = crate::utils::Instant::now();
        crate::dynamics::solver::plugin::solve_contacts_graph::<false>(
            &mut solver_bodies,
            &mut graph,
            &config,
            length_unit.0,
            dt,
            |contact| {
                islands
                    .contact_node(contact.contact_id)
                    .is_some_and(|node| coupled_islands.binary_search(&node.island_id()).is_ok())
            },
        );
        diagnostics.relax_velocities += start.elapsed();
        apply_xpbd_joint_pass::<FixedJoint>(
            &mut solver_bodies,
            &body_indices,
            &mut fixed,
            dt,
            JointSolverPhase::ProjectVelocities,
        );
        apply_xpbd_joint_pass::<PrismaticJoint>(
            &mut solver_bodies,
            &body_indices,
            &mut sliders,
            dt,
            JointSolverPhase::ProjectVelocities,
        );
        apply_xpbd_joint_pass::<RevoluteJoint>(
            &mut solver_bodies,
            &body_indices,
            &mut hinges,
            dt,
            JointSolverPhase::ProjectVelocities,
        );
    }
}

/// System sets for the XPBD constraint solver in the [`SubstepSchedule`].
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum XpbdSolverSystems {
    /// Solves constraints using Extended Position-Based Dynamics (XPBD).
    SolveConstraints,
    /// A system set for user constraints.
    SolveUserConstraints,
    /// Performs velocity updates after XPBD constraint solving.
    VelocityProjection,
}

/// A marker component for [rigid bodies](RigidBody) whose velocities should be projected
/// from the position corrections applied by XPBD constraints.
///
/// The XPBD solver only stores pre-solve deltas and projects velocities for bodies with this
/// component, so bodies that no XPBD constraint touches are skipped entirely.
///
/// This is inserted and removed automatically for the bodies of joints in the [`JointGraph`].
/// If you implement a [custom XPBD constraint](crate::dynamics::solver::xpbd#custom-constraints)
/// that is not registered in the joint graph, insert this component on the participating bodies
/// yourself, or the constraint will have no effect on their velocities.
///
/// [`JointGraph`]: crate::dynamics::joints::joint_graph::JointGraph
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
#[reflect(Component, Debug, Default, PartialEq)]
pub struct XpbdVelocityProjection;

/// Iterates through the XPBD joints of a given type and solves them.
pub fn prepare_xpbd_joint<
    C: Component<Mutability = Mutable> + EntityConstraint<2> + XpbdConstraint<2>,
>(
    bodies: Query<RigidBodyQueryReadOnly, Without<RigidBodyDisabled>>,
    mut joints: Query<(&mut C, &mut C::SolverData), (Without<RigidBody>, Without<JointDisabled>)>,
) where
    C::SolverData: Component<Mutability = Mutable>,
{
    for (mut joint, mut solver_data) in &mut joints {
        // Clear the Lagrange multipliers.
        solver_data.clear_lagrange_multipliers();

        // Get components for entities
        if let Ok([body1, body2]) = bodies.get_many(joint.entities()) {
            joint.prepare([&body1, &body2], &mut solver_data);
        }
    }
}

/// Iterates through the XPBD joints of a given type and solves them.
///
/// Corrections execute in ascending live entity-index order, independently of
/// archetype table packing. Deterministic replay still requires deterministic
/// entity creation and topology changes by the application.
pub fn solve_xpbd_joint<
    C: Component<Mutability = Mutable> + EntityConstraint<2> + XpbdConstraint<2>,
>(
    mut solver_bodies: ResMut<SolverBodies>,
    body_indices: Query<Option<&SolverBodyIndex>, (With<RigidBody>, Without<RigidBodyDisabled>)>,
    mut joints: Query<(&mut C, &mut C::SolverData), (Without<RigidBody>, Without<JointDisabled>)>,
    time: Res<Time>,
) where
    C::SolverData: Component<Mutability = Mutable>,
{
    // Joint corrections share body state. Order by live entity slot index,
    // independently of ECS table packing and Entity's opaque bit encoding.
    let mut joints: Vec<_> = joints
        .iter_mut()
        .sort_unstable_by_key::<Entity, _>(|entity| entity.index())
        .collect();
    apply_xpbd_joint_pass::<C>(
        &mut solver_bodies,
        &body_indices,
        &mut joints,
        time.delta_secs(),
        JointSolverPhase::Position,
    );
}

#[derive(Clone, Copy)]
enum JointSolverPhase {
    Position,
    PrepareVelocities,
    ProjectVelocities,
}

fn apply_xpbd_joint_pass<
    C: Component<Mutability = Mutable> + EntityConstraint<2> + XpbdConstraint<2>,
>(
    solver_bodies: &mut SolverBodies,
    body_indices: &Query<Option<&SolverBodyIndex>, (With<RigidBody>, Without<RigidBodyDisabled>)>,
    joints: &mut [(Mut<'_, C>, Mut<'_, C::SolverData>)],
    delta_secs: f32,
    phase: JointSolverPhase,
) where
    C::SolverData: Component<Mutability = Mutable>,
{
    let access = solver_bodies.access();

    let mut dummy_body1 = SolverBody::default();
    let mut dummy_body2 = SolverBody::default();

    for (joint, solver_data) in joints {
        let [entity1, entity2] = joint.entities();

        let Ok([index1, index2]) = body_indices.get_many([entity1, entity2]) else {
            continue;
        };
        let index1 = index1.copied().unwrap_or(SolverBodyIndex::INVALID);
        let index2 = index2.copied().unwrap_or(SolverBodyIndex::INVALID);

        if index1 == index2 {
            continue;
        }

        let (mut body1, mut inertia1) = (&mut dummy_body1, &SolverBodyInertia::DUMMY);
        let (mut body2, mut inertia2) = (&mut dummy_body2, &SolverBodyInertia::DUMMY);

        // Get the solver bodies for the two jointed bodies.
        //
        // SAFETY: The two jointed bodies are distinct, and joints are processed serially here.
        let (b1, b2) = unsafe { access.get_pair_unchecked_mut(index1, index2) };
        if let Some((body, inertia)) = b1 {
            body1 = body;
            inertia1 = inertia;
        }
        if let Some((body, inertia)) = b2 {
            body2 = body;
            inertia2 = inertia;
        }

        // If a body has a higher dominance, it is treated as a static or kinematic body.
        match (inertia1.dominance() - inertia2.dominance()).cmp(&0) {
            Ordering::Greater => inertia1 = &SolverBodyInertia::DUMMY,
            Ordering::Less => inertia2 = &SolverBodyInertia::DUMMY,
            _ => {}
        }

        match phase {
            JointSolverPhase::PrepareVelocities => {
                joint.prepare_velocity_projection([body1, body2], [inertia1, inertia2], solver_data)
            }
            JointSolverPhase::ProjectVelocities => joint.project_velocities(
                [body1, body2],
                [inertia1, inertia2],
                solver_data,
                delta_secs,
            ),
            JointSolverPhase::Position => joint.solve(
                [body1, body2],
                [inertia1, inertia2],
                solver_data,
                delta_secs,
            ),
        }
    }
}

/// Stores the delta position and rotation of each body before XPBD constraints are solved.
fn store_pre_solve_deltas(
    solver_bodies: Res<SolverBodies>,
    mut query: Query<
        (
            &SolverBodyIndex,
            &mut PreSolveDeltaPosition,
            &mut PreSolveDeltaRotation,
        ),
        (With<XpbdVelocityProjection>, Without<RigidBodyDisabled>),
    >,
) {
    for (index, mut pre_solve_delta_position, mut pre_solve_delta_rotation) in &mut query {
        let Some(body) = solver_bodies.get(*index) else {
            continue;
        };
        pre_solve_delta_position.0 = body.delta_position;
        pre_solve_delta_rotation.0 = body.delta_rotation;
    }
}

/// Updates the linear velocity of all dynamic bodies based on the change in position from the XPBD solver.
fn project_linear_velocity(
    mut solver_bodies: ResMut<SolverBodies>,
    bodies: Query<
        (&SolverBodyIndex, &PreSolveDeltaPosition),
        (With<XpbdVelocityProjection>, RigidBodyActiveFilter),
    >,
    time: Res<Time>,
) {
    let delta_secs = time.delta_secs();

    let access = solver_bodies.access();

    for (index, pre_solve_delta_pos) in &bodies {
        // SAFETY: Each entity has a unique solver body index, so the accessed bodies are disjoint.
        let body = unsafe { access.body_unchecked_mut(*index) };
        // v = (x - x_prev) / h
        let new_lin_vel = (body.delta_position - pre_solve_delta_pos.0) / delta_secs;
        body.linear_velocity += new_lin_vel;
    }
}

/// Updates the angular velocity of all dynamic bodies based on the change in rotation from the XPBD solver.
#[cfg(feature = "2d")]
fn project_angular_velocity(
    mut solver_bodies: ResMut<SolverBodies>,
    bodies: Query<
        (&SolverBodyIndex, &PreSolveDeltaRotation),
        (With<XpbdVelocityProjection>, RigidBodyActiveFilter),
    >,
    time: Res<Time>,
) {
    let delta_secs = time.delta_secs();

    let access = solver_bodies.access();

    for (index, pre_solve_delta_rot) in &bodies {
        // SAFETY: Each entity has a unique solver body index, so the accessed bodies are disjoint.
        let body = unsafe { access.body_unchecked_mut(*index) };
        let new_ang_vel = pre_solve_delta_rot.angle_to(body.delta_rotation) / delta_secs;
        body.angular_velocity += new_ang_vel;
    }
}

/// Updates the angular velocity of all dynamic bodies based on the change in rotation from the XPBD solver.
#[cfg(feature = "3d")]
fn project_angular_velocity(
    mut solver_bodies: ResMut<SolverBodies>,
    bodies: Query<
        (&SolverBodyIndex, &PreSolveDeltaRotation),
        (With<XpbdVelocityProjection>, RigidBodyActiveFilter),
    >,
    time: Res<Time>,
) {
    let delta_secs = time.delta_secs();

    let access = solver_bodies.access();

    for (index, pre_solve_delta_rot) in &bodies {
        // SAFETY: Each entity has a unique solver body index, so the accessed bodies are disjoint.
        let body = unsafe { access.body_unchecked_mut(*index) };
        let delta_rot = body.delta_rotation.mul_quat(pre_solve_delta_rot.inverse());

        let mut new_ang_vel = 2.0 * delta_rot.xyz() / delta_secs;

        if delta_rot.w < 0.0 {
            new_ang_vel = -new_ang_vel;
        }

        body.angular_velocity += new_ang_vel;
    }
}

fn writeback_joint_forces<C: Component + EntityConstraint<2> + XpbdConstraint<2>>(
    mut joints: Query<(&C::SolverData, &mut JointForces), Without<JointDisabled>>,
    time: Res<Time>,
    substep_count: Res<SubstepCount>,
) where
    C::SolverData: Component<Mutability = Mutable>,
{
    let delta_secs = time.delta_secs();

    // Detailed Rigid Body Simulation with Extended Position Based Dynamics by Müller et al.
    // states that  `f = λ * n / h²`. However, with substepping, it seems that we need to accumulate
    // Lagrange multipliers across substeps, and use the formula `f = λ * n / dt^2 * substep_count`.
    let rhs = (delta_secs * delta_secs).recip_or_zero() * substep_count.0 as f32;

    for (solver_data, mut forces) in &mut joints {
        let force = forces.force() - solver_data.total_position_lagrange() * rhs;
        let torque = forces.torque() - solver_data.total_rotation_lagrange() * rhs;
        forces.set_force(force);
        forces.set_torque(torque);
        forces.set_motor_force(solver_data.total_motor_lagrange() * rhs);
    }
}
