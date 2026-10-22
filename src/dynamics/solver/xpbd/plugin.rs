use super::{ConstraintSolveFailure, XpbdSolverStatus};
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
        app.init_resource::<XpbdSolverStatus>()
            .register_type::<XpbdSolverStatus>();
        for set in [
            SolverSystems::PostSubstep,
            SolverSystems::Restitution,
            SolverSystems::ContinuousCollision,
            SolverSystems::Finalize,
            SolverSystems::StoreContactImpulses,
        ] {
            app.configure_sets(PhysicsSchedule, set.run_if(super::xpbd_solver_healthy));
        }
        app.configure_sets(
            SubstepSchedule,
            XpbdSolverSystems::SolveUserConstraints.run_if(super::xpbd_solver_healthy),
        );
        #[cfg(feature = "3d")]
        app.register_required_components::<RigidBody, XpbdVelocityProjection>();

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
                #[cfg(feature = "2d")]
                solve_xpbd_joint::<FixedJoint>,
                #[cfg(feature = "2d")]
                solve_xpbd_joint::<RevoluteJoint>,
                #[cfg(feature = "3d")]
                solve_xpbd_joint::<SphericalJoint>,
                #[cfg(feature = "2d")]
                solve_xpbd_joint::<PrismaticJoint>,
                solve_xpbd_joint::<DistanceJoint>,
                #[cfg(feature = "3d")]
                super::joints::solve_joint_contact_islands,
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

    #[cfg(feature = "3d")]
    fn finish(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<crate::dynamics::solver::islands::IslandPlugin>(),
            "native 3D XPBD assemblies require IslandPlugin"
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
pub fn solve_xpbd_joint<
    C: Component<Mutability = Mutable> + EntityConstraint<2> + XpbdConstraint<2>,
>(
    mut solver_bodies: ResMut<SolverBodies>,
    body_indices: Query<Option<&SolverBodyIndex>, (With<RigidBody>, Without<RigidBodyDisabled>)>,
    mut joints: Query<
        (Entity, &mut C, &mut C::SolverData),
        (Without<RigidBody>, Without<JointDisabled>),
    >,
    time: Res<Time>,
    mut physics_time: ResMut<Time<Physics>>,
    mut status: ResMut<XpbdSolverStatus>,
) where
    C::SolverData: Component<Mutability = Mutable>,
{
    if status.failure.is_some() {
        return;
    }
    let delta_secs = time.delta_secs();

    let access = solver_bodies.access();

    let mut dummy_body1 = SolverBody::default();
    let mut dummy_body2 = SolverBody::default();

    for (entity, mut joint, mut solver_data) in &mut joints {
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

        if let Err(reason) = joint.solve(
            [body1, body2],
            [inertia1, inertia2],
            &mut solver_data,
            delta_secs,
        ) {
            status.failure = Some(ConstraintSolveFailure {
                constraints: vec![entity],
                reason,
            });
            physics_time.pause();
            return;
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
    status: Res<XpbdSolverStatus>,
) {
    if status.failure.is_some() {
        return;
    }
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
    status: Res<XpbdSolverStatus>,
) {
    if status.failure.is_some() {
        return;
    }
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
    status: Res<XpbdSolverStatus>,
) {
    if status.failure.is_some() {
        return;
    }
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
