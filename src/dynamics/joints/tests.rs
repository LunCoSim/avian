use core::time::Duration;

use approx::assert_relative_eq;
use bevy::{mesh::MeshPlugin, prelude::*, time::TimeUpdateStrategy};

use crate::prelude::*;

const TIMESTEP: f32 = 1.0 / 64.0;

#[derive(Resource)]
struct ExpectedConstraintFailure;

fn create_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        PhysicsPlugins::default(),
        TransformPlugin,
        AssetPlugin::default(),
        bevy::diagnostic::DiagnosticsPlugin,
        #[cfg(feature = "bevy_scene")]
        bevy::scene::ScenePlugin,
        MeshPlugin,
    ));

    app.insert_resource(SubstepCount(20));

    app.insert_resource(Gravity(Vector::ZERO));

    app.insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f32(
        TIMESTEP,
    )));
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        TIMESTEP,
    )));

    app.add_systems(
        Last,
        |status: Res<crate::dynamics::solver::xpbd::XpbdSolverStatus>,
         expected: Option<Res<ExpectedConstraintFailure>>| {
            assert!(
                expected.is_some() || status.failure.is_none(),
                "unexpected native constraint failure: {:?}",
                status.failure
            );
        },
    );

    app
}

#[cfg(feature = "3d")]
#[test]
fn loaded_slider_impact_preserves_contact_and_travel() {
    use crate::math::ToF32Precision;
    for (ground_kind, ground_speed) in [(RigidBody::Static, 0.0), (RigidBody::Kinematic, 0.2)] {
        let mut app = create_app();
        app.insert_resource(SubstepCount(8));
        app.insert_resource(Gravity(Vec3::NEG_Y * 1.62));
        app.finish();
        let ground = app
            .world_mut()
            .spawn((
                ground_kind,
                Position(RVector::NEG_Y * 0.5),
                Collider::cuboid(8.0, 1.0, 8.0),
                LinearVelocity(Vec3::Y * ground_speed),
            ))
            .id();
        let hull = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Position(RVector::Y * 2.3),
                Mass(4000.0),
                AngularInertia::new(Vec3::splat(7000.0)),
                LinearVelocity(Vec3::NEG_Y * 6.0),
            ))
            .id();
        let foot = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Position(RVector::Y * 0.3),
                Mass(180.0),
                AngularInertia::new(Vec3::splat(10.0)),
                Collider::cuboid(0.6, 0.2, 0.6),
                LinearVelocity(Vec3::NEG_Y * 6.0),
                SpeculativeCcd::MAX,
                SweptCcd::default().with_filter(CcdFilter::ALL),
            ))
            .id();
        app.world_mut().spawn(
            PrismaticJoint::new(hull, foot)
                .with_slider_axis(Vec3::Y)
                .with_local_anchor1(Vec3::NEG_Y * 2.0)
                .with_limits(-0.05, 0.02)
                .with_motor(LinearMotor {
                    max_force: 12000.0,
                    motor_model: MotorModel::ForceBased {
                        stiffness: 85000.0,
                        damping: 19000.0,
                    },
                    ..default()
                }),
        );
        let mut penetration: f64 = 0.0;
        let mut travel_error: f64 = 0.0;
        let mut worst = None;
        for tick in 0..128 {
            app.update();
            let hull_position = app.world().get::<Position>(hull).unwrap().0;
            let foot_position = app.world().get::<Position>(foot).unwrap().0;
            let hull_rotation = app.world().get::<Rotation>(hull).unwrap().0;
            let foot_rotation = app.world().get::<Rotation>(foot).unwrap().0;
            let relative =
                (foot_position - hull_position).f32() - hull_rotation * (Vec3::NEG_Y * 2.0);
            let slide = f64::from(relative.dot(hull_rotation * Vec3::Y));
            for x in [-0.3, 0.3] {
                for y in [-0.1, 0.1] {
                    for z in [-0.3, 0.3] {
                        let corner_y = foot_position.y as f64
                            + f64::from((foot_rotation * Vec3::new(x, y, z)).y);
                        let ground_y =
                            app.world().get::<Position>(ground).unwrap().0.y as f64 + 0.5;
                        penetration = penetration.max(ground_y - corner_y);
                    }
                }
            }
            let error = (-0.05 - slide).max(slide - 0.02);
            if error > travel_error {
                travel_error = error;
                let index = *app
                    .world()
                    .get::<crate::dynamics::solver::solver_body::SolverBodyIndex>(foot)
                    .unwrap();
                worst = Some((
                    tick,
                    slide,
                    app.world()
                        .resource::<crate::dynamics::solver::solver_body::SolverBodies>()
                        .get(index)
                        .unwrap()
                        .flags,
                ));
            }
        }
        assert!(
            penetration <= 0.001 && travel_error <= 0.001,
            "ground={ground_kind:?}, penetration={penetration}, travel_error={travel_error}, worst={worst:?}"
        );
        assert_eq!(
            app.world().get::<LinearVelocity>(ground).unwrap().0,
            Vec3::Y * ground_speed
        );
    }
}

/// Tests that an angular motor on a revolute joint spins the attached body.
#[test]
fn revolute_motor_spins_body() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 2.0,
            max_torque: 100.0,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 0.0,
                damping: 10.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() > 1.0,
            "Angular velocity should be significant"
        );
        assert_relative_eq!(angular_velocity.0, 2.0, epsilon = 0.5);
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(speed > 1.0, "Angular velocity should be significant");
    }
}

/// Tests that a linear motor on a prismatic joint moves the attached body.
#[test]
fn prismatic_motor_moves_body() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                target_velocity: 1.0,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    // Initialize the app.
    app.update();

    let initial_x = app.world().entity(dynamic).get::<Position>().unwrap().0.x;

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_x = body_ref.get::<Position>().unwrap().0.x;

    let displacement = final_x - initial_x;
    assert!(
        displacement > 0.5,
        "Body should have moved: {}",
        displacement
    );
}

/// Tests that an angular motor with max torque limit respects the limit.
#[test]
fn revolute_motor_respects_max_torque() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(100.0), // Heavy body to test torque limiting
            #[cfg(feature = "2d")]
            AngularInertia(100.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(100.0)),
        ))
        .id();

    // Create a revolute joint with a very limited motor torque.
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 10.0,
            max_torque: 0.1,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 0.0,
                damping: 1.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() < 5.0,
            "Velocity should be limited by max torque"
        );
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(speed < 5.0, "Velocity should be limited by max torque");
    }
}

/// Tests that a position-targeting motor moves the joint towards the target position.
#[test]
fn revolute_motor_position_target() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Create a revolute joint with a position-targeting motor.
    let target_angle = 1.0;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_position: target_angle,
            max_torque: f32::MAX,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 50.0,
                damping: 20.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 3 seconds to let it settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated towards the target angle (allow some tolerance).
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle.abs() > 0.3,
            "Motor should have rotated the body: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle.abs() > 0.3,
            "Motor should have rotated the body: {}",
            signed_angle
        );
    }
}

/// Tests that a linear position-targeting motor moves the joint towards the target position.
#[test]
fn prismatic_motor_position_target() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Create a prismatic joint with a position-targeting motor.
    // Use AccelerationBased model for stable position targeting.
    let target_position = 1.0; // Target is 1 meter along the slider axis
    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                target_position,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 10.0,
                    damping: 5.0,
                },
                ..default()
            }),
    );

    // Initialize the app.
    app.update();

    let initial_pos = app.world().entity(dynamic).get::<Position>().unwrap().0;

    // Run simulation for 3 seconds to let it settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_pos = body_ref.get::<Position>().unwrap().0;

    assert!(!final_pos.x.is_nan(), "Final position should not be NaN");
    assert!(!final_pos.y.is_nan(), "Final position should not be NaN");

    let displacement = final_pos.x - initial_pos.x;

    assert!(
        displacement.abs() > 0.1 || final_pos.x.abs() > 0.1,
        "Body should have moved: displacement={}, final_x={}",
        displacement,
        final_pos.x
    );
}

/// Tests that a velocity motor on a revolute joint respects angle limits.
///
/// The motor drives with constant velocity, but the joint should stop
/// when it reaches the angle limit.
#[test]
fn revolute_motor_respects_angle_limits() {
    use core::f32::consts::PI;

    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    let angle_limit = PI / 4.0;

    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic)
            .with_angle_limits(-angle_limit, angle_limit)
            .with_motor(AngularMotor {
                target_velocity: 5.0,
                max_torque: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    app.update();

    // Run for 2 seconds - enough time for motor to hit the limit.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle <= angle_limit + 0.1,
            "Angle {} should not exceed limit {}",
            angle,
            angle_limit
        );
        assert!(
            angle > angle_limit - 0.3,
            "Angle {} should be near the limit {}",
            angle,
            angle_limit
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle <= angle_limit + 0.1,
            "Angle {} should not exceed limit {}",
            signed_angle,
            angle_limit
        );
        assert!(
            signed_angle > angle_limit - 0.3,
            "Angle {} should be near the limit {}",
            signed_angle,
            angle_limit
        );
    }
}

/// Tests that a velocity motor on a prismatic joint respects distance limits.
///
/// The motor drives with constant velocity, but the joint should stop
/// when it reaches the distance limit.
#[test]
fn prismatic_motor_respects_limits() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    // Start at origin so we can measure displacement clearly.
    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Limit translation to [0, 1] meters along the slide axis.
    let distance_limit = 1.0;

    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_limits(0.0, distance_limit)
            .with_motor(LinearMotor {
                target_velocity: 5.0, // High velocity to ensure we hit the limit
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    app.update();

    // Make sure the motor is not near the limit from the start.
    {
        let body_ref = app.world().entity(dynamic);
        let position = body_ref.get::<Position>().unwrap();
        assert!(
            (position.0.x - distance_limit.real()).abs() > 0.1,
            "Displacement {} should not be near the limit {} at the start of the test",
            position.0.x,
            distance_limit
        );
    }

    // Run for 2 seconds - enough time for motor to hit the limit.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let position = body_ref.get::<Position>().unwrap();

    // The displacement along the slide axis (X) should be at or near the limit.
    let displacement = position.x.f32();
    assert!(
        displacement <= distance_limit + 0.001,
        "Displacement {} should not exceed limit {}",
        displacement,
        distance_limit
    );
    assert!(
        (displacement - distance_limit).abs() < 0.1,
        "Displacement {} should be near the limit {}",
        displacement,
        distance_limit
    );
}

/// Tests that `ForceBased` motor model works for revolute joints.
///
/// This is the physically accurate motor model that takes mass into account.
#[test]
fn revolute_motor_force_based() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Use ForceBased motor model with velocity control.
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 2.0,
            max_torque: 100.0,
            motor_model: MotorModel::ForceBased {
                stiffness: 0.0,
                damping: 10.0,
            },
            ..default()
        }),
    );

    app.update();

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();
    #[cfg(feature = "2d")]
    let initial_speed = angular_velocity.0.abs();
    #[cfg(feature = "3d")]
    let initial_speed = angular_velocity.0.length();

    assert!(
        initial_speed.abs() < 0.001,
        "ForceBased motor should be initiall still, speed: {}",
        initial_speed
    );

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    // The body should have gained angular velocity.
    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() > 0.5,
            "ForceBased motor should spin the body: {}",
            angular_velocity.0
        );
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(
            speed > 0.5,
            "ForceBased motor should spin the body: {}",
            speed
        );
    }
}

/// Tests that the default `SpringDamper` motor model works.
///
/// `SpringDamper` is unconditionally stable and uses `frequency`/`damping_ratio`.
#[test]
fn revolute_motor_spring_damper() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    let target_angle = 1.0;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_position: target_angle,
            max_torque: f32::MAX,
            motor_model: MotorModel::SpringDamper {
                frequency: 2.0,
                damping_ratio: 1.0,
            },
            ..default()
        }),
    );

    app.update();

    // TODO Assert the body is initially not near the target.

    // Run simulation for 3 seconds to let the spring settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated towards the target.
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle.abs() > 0.3,
            "SpringDamper motor should rotate towards target: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle.abs() > 0.3,
            "SpringDamper motor should rotate towards target: {}",
            signed_angle
        );
    }
}

/// Tests that a motor with both velocity and position targeting works.
///
/// Combined spring-damper behavior: position targeting with velocity damping.
#[test]
fn revolute_motor_combined_position_velocity() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Motor with both position and velocity targeting.
    // The velocity adds a constant offset to the spring behavior.
    let target_angle = 0.5;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            enabled: true,
            target_position: target_angle,
            target_velocity: 0.5,
            max_torque: 100.0,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 30.0,
                damping: 15.0,
            },
        }),
    );

    app.update();

    // Run for 2 seconds.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated in the positive direction.
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle > 0.2,
            "Combined motor should rotate body positively: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle > 0.2,
            "Combined motor should rotate body positively: {}",
            signed_angle
        );
    }
}

/// Tests that a prismatic motor with both velocity and position targeting works.
#[test]
fn prismatic_motor_combined_position_velocity() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Motor with both position and velocity targeting.
    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                enabled: true,
                target_position: 1.0,
                target_velocity: 0.5,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 20.0,
                    damping: 10.0,
                },
            }),
    );

    app.update();

    let initial_x = app.world().entity(dynamic).get::<Position>().unwrap().0.x;

    // Run for 2 seconds.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_x = body_ref.get::<Position>().unwrap().0.x;

    // The body should have moved.
    let displacement = final_x - initial_x;
    assert!(
        displacement.abs() > 0.1,
        "Combined motor should move the body: {}",
        displacement
    );
}

#[cfg(feature = "2d")]
fn z_spin(angular_velocity: &AngularVelocity) -> f32 {
    angular_velocity.0
}

#[cfg(feature = "3d")]
fn z_spin(angular_velocity: &AngularVelocity) -> f32 {
    angular_velocity.z
}

/// Tests that angular joint damping between bodies with unequal inertia conserves angular momentum.
#[test]
fn joint_damping_conserves_angular_momentum() {
    let mut app = create_app();
    app.finish();

    let light = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
            #[cfg(feature = "2d")]
            AngularVelocity(10.0),
            #[cfg(feature = "3d")]
            AngularVelocity(Vector::Z * 10.0),
        ))
        .id();

    let heavy = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(10.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(10.0)),
        ))
        .id();

    app.world_mut().spawn((
        RevoluteJoint::new(light, heavy),
        JointDamping {
            linear: 0.0,
            angular: 20.0,
        },
    ));

    app.update();

    let steps = (1.0 / TIMESTEP) as usize;
    for _ in 0..steps {
        app.update();
    }

    let light_spin = z_spin(app.world().get::<AngularVelocity>(light).unwrap());
    let heavy_spin = z_spin(app.world().get::<AngularVelocity>(heavy).unwrap());

    assert_relative_eq!(light_spin, heavy_spin, epsilon = 1e-3);
    assert_relative_eq!(1.0 * light_spin + 10.0 * heavy_spin, 10.0, epsilon = 1e-3);
}

/// Joints a moving kinematic body to a resting dynamic body and returns the kinematic body's
/// velocities after one second.
fn kinematic_velocity_after_damping<J: Component>(
    joint: fn(Entity, Entity) -> J,
    initial_velocity: (LinearVelocity, AngularVelocity),
    damping: JointDamping,
) -> (LinearVelocity, AngularVelocity) {
    let mut app = create_app();
    app.finish();

    let kinematic = app
        .world_mut()
        .spawn((
            RigidBody::Kinematic,
            Position(RVector::ZERO),
            // Kinematic bodies normally get mass from their colliders.
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
            initial_velocity,
        ))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn((joint(kinematic, dynamic), damping));

    app.update();

    let steps = (1.0 / TIMESTEP) as usize;
    for _ in 0..steps {
        app.update();
    }

    let body = app.world().entity(kinematic);
    (
        *body.get::<LinearVelocity>().unwrap(),
        *body.get::<AngularVelocity>().unwrap(),
    )
}

/// Tests that angular joint damping does not change the angular velocity of a kinematic body.
#[test]
fn joint_damping_does_not_change_kinematic_angular_velocity() {
    #[cfg(feature = "2d")]
    let spin = AngularVelocity(2.0);
    #[cfg(feature = "3d")]
    let spin = AngularVelocity(Vector::Z * 2.0);

    let (_, undamped) = kinematic_velocity_after_damping(
        RevoluteJoint::new,
        (LinearVelocity::ZERO, spin),
        JointDamping::default(),
    );
    assert_eq!(undamped, spin, "control: joint alone changed it");

    let (_, damped) = kinematic_velocity_after_damping(
        RevoluteJoint::new,
        (LinearVelocity::ZERO, spin),
        JointDamping {
            linear: 0.0,
            angular: 1.0,
        },
    );
    assert_eq!(damped, spin);
}

/// Tests that linear joint damping does not change the linear velocity of a kinematic body.
#[test]
fn joint_damping_does_not_change_kinematic_linear_velocity() {
    let velocity = LinearVelocity(Vector::X * 2.0);

    let (undamped, _) = kinematic_velocity_after_damping(
        PrismaticJoint::new,
        (velocity, AngularVelocity::ZERO),
        JointDamping::default(),
    );
    assert_eq!(undamped, velocity, "control: joint alone changed it");

    let (damped, _) = kinematic_velocity_after_damping(
        PrismaticJoint::new,
        (velocity, AngularVelocity::ZERO),
        JointDamping {
            linear: 1.0,
            angular: 0.0,
        },
    );
    assert_eq!(damped, velocity);
}

/// Tests that angular joint damping still damps the free axis of a body with locked rotation axes.
#[cfg(feature = "3d")]
#[test]
fn joint_damping_damps_free_axis_of_rotation_locked_body() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let body = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            AngularInertia::new(Vec3::splat(1.0)),
            LockedAxes::new().lock_rotation_x().lock_rotation_z(),
            AngularVelocity(Vector::Y * 2.0),
        ))
        .id();

    app.world_mut().spawn((
        SphericalJoint::new(anchor, body),
        JointDamping {
            linear: 0.0,
            angular: 5.0,
        },
    ));

    app.update();

    let steps = (1.0 / TIMESTEP) as usize;
    for _ in 0..steps {
        app.update();
    }

    let spin = app.world().get::<AngularVelocity>(body).unwrap().y;
    assert!(spin.abs() < 0.05, "free axis was not damped: {spin}");
}

/// A zero actuator cap must stay zero across frames and both joint kernels.
#[cfg(feature = "3d")]
#[test]
fn motor_zero_caps_do_not_actuate_either_joint_kind() {
    let mut app = create_app();
    app.finish();
    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();
    let bodies = [0, 1].map(|_| {
        app.world_mut()
            .spawn((
                RigidBody::Dynamic,
                Position(RVector::ZERO),
                Mass(1.0),
                AngularInertia::new(Vec3::ONE),
            ))
            .id()
    });
    let model = MotorModel::ForceBased {
        stiffness: 1000.0,
        damping: 100.0,
    };
    let linear = app
        .world_mut()
        .spawn((
            PrismaticJoint::new(anchor, bodies[0]).with_motor(
                LinearMotor::new(model)
                    .with_target_position(0.1)
                    .with_max_force(0.0),
            ),
            JointForces::default(),
        ))
        .id();
    let angular = app
        .world_mut()
        .spawn((
            RevoluteJoint::new(anchor, bodies[1]).with_motor(
                AngularMotor::new(model)
                    .with_target_position(0.1)
                    .with_max_torque(0.0),
            ),
            JointForces::default(),
        ))
        .id();
    for _ in 0..65 {
        app.update();
    }
    for body in bodies {
        assert_eq!(
            app.world().get::<LinearVelocity>(body).unwrap().0,
            Vec3::ZERO
        );
        assert_eq!(
            app.world().get::<AngularVelocity>(body).unwrap().0,
            Vec3::ZERO
        );
    }
    for joint in [linear, angular] {
        let forces = app.world().get::<JointForces>(joint).unwrap();
        assert_eq!(forces.force(), Vec3::ZERO);
        assert_eq!(forces.torque(), Vec3::ZERO);
        assert_eq!(forces.motor_force(), 0.0);
    }
}

/// Dimensional force gains must remain stable on low-mass endpoints.
#[cfg(feature = "3d")]
#[test]
fn motor_force_drives_remain_finite_with_small_mass_and_inertia() {
    let mut app = create_app();
    app.finish();
    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();
    let bodies = [0, 1].map(|_| {
        app.world_mut()
            .spawn((
                RigidBody::Dynamic,
                Position(RVector::ZERO),
                Mass(0.001),
                AngularInertia::new(Vec3::splat(1.0e-6)),
                SleepingDisabled,
            ))
            .id()
    });
    let model = MotorModel::ForceBased {
        stiffness: 1000.0,
        damping: 100.0,
    };
    app.world_mut().spawn(
        PrismaticJoint::new(anchor, bodies[0])
            .with_slider_axis(Vec3::Y)
            .with_motor(LinearMotor::new(model).with_target_position(0.1)),
    );
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, bodies[1])
            .with_motor(AngularMotor::new(model).with_target_position(0.1)),
    );
    for _ in 0..65 {
        app.update();
        for body in bodies {
            assert!(
                app.world()
                    .get::<LinearVelocity>(body)
                    .unwrap()
                    .0
                    .is_finite()
            );
            assert!(
                app.world()
                    .get::<AngularVelocity>(body)
                    .unwrap()
                    .0
                    .is_finite()
            );
        }
    }
    assert_relative_eq!(
        app.world().get::<Position>(bodies[0]).unwrap().0.y,
        0.1,
        epsilon = 1.0e-4
    );
    assert_relative_eq!(
        app.world()
            .get::<Rotation>(bodies[1])
            .unwrap()
            .0
            .to_scaled_axis()
            .z,
        0.1,
        epsilon = 1.0e-4
    );
}

/// Disabling a joint stops its damping and retires its completed reaction.
#[cfg(feature = "3d")]
#[test]
fn joint_disabled_clears_completed_reactions_and_damping() {
    let mut app = create_app();
    app.finish();
    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();
    let body = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            AngularInertia::new(Vec3::ONE),
            AngularVelocity(Vec3::Z * 2.0),
            SleepingDisabled,
        ))
        .id();
    let joint = app
        .world_mut()
        .spawn((
            RevoluteJoint::new(anchor, body),
            JointDamping {
                linear: 0.0,
                angular: 20.0,
            },
            JointForces::default(),
        ))
        .id();
    app.update();
    app.update();
    let before = app.world().get::<AngularVelocity>(body).unwrap().0;
    assert!(app.world().get::<JointForces>(joint).unwrap().torque().z < 0.0);
    app.world_mut().entity_mut(joint).insert(JointDisabled);
    app.update();
    assert_eq!(app.world().get::<AngularVelocity>(body).unwrap().0, before);
    assert_eq!(
        app.world().get::<JointForces>(joint).unwrap().torque(),
        Vec3::ZERO
    );
}

#[cfg(feature = "3d")]
#[test]
fn rotating_fixed_joint_preserves_offset_and_orientation() {
    let mut app = create_app();
    app.insert_resource(SubstepCount(8));
    app.finish();
    let bodies = [
        (RVector::ZERO, Vec3::Y * (30.0 * core::f32::consts::PI)),
        (RVector::X, Vec3::ZERO),
    ]
    .map(|(position, omega)| {
        app.world_mut()
            .spawn((
                RigidBody::Dynamic,
                Position(position),
                Rotation::IDENTITY,
                Mass(1.0),
                AngularInertia::new(Vec3::splat(0.004)),
                AngularVelocity(omega),
                SleepingDisabled,
            ))
            .id()
    });
    app.world_mut()
        .spawn(FixedJoint::new(bodies[0], bodies[1]).with_local_anchor1(Vec3::X));
    for _ in 0..64 {
        app.update();
        let p1 = app.world().get::<Position>(bodies[0]).unwrap().0;
        let p2 = app.world().get::<Position>(bodies[1]).unwrap().0;
        let q1 = app.world().get::<Rotation>(bodies[0]).unwrap().0;
        let q2 = app.world().get::<Rotation>(bodies[1]).unwrap().0;
        let separation = (p1 + (q1 * Vec3::X).real() - p2).length();
        let angle = q1.angle_between(q2);
        assert!(
            separation < 0.001 && angle < 0.001,
            "fixed joint residual: separation={separation}, angle={angle}"
        );
    }
}

#[cfg(feature = "3d")]
#[test]
fn disabled_body_does_not_become_a_fixed_joint_anchor_or_damping_sink() {
    for angular in [false, true] {
        let mut app = create_app();
        app.finish();
        let anchor = app
            .world_mut()
            .spawn((
                RigidBody::Static,
                RigidBodyDisabled,
                Position(RVector::ZERO),
            ))
            .id();
        let velocity = if angular { Vec3::ZERO } else { Vec3::X * 2.0 };
        let spin = if angular { Vec3::Z * 2.0 } else { Vec3::ZERO };
        let body = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Position(RVector::X),
                Mass(1.0),
                AngularInertia::new(Vec3::ONE),
                LinearVelocity(velocity),
                AngularVelocity(spin),
                SleepingDisabled,
            ))
            .id();
        let joint = if angular {
            app.world_mut()
                .spawn((
                    RevoluteJoint::new(anchor, body),
                    JointDamping {
                        linear: 0.0,
                        angular: 20.0,
                    },
                    JointForces::default(),
                ))
                .id()
        } else {
            app.world_mut()
                .spawn((
                    FixedJoint::new(anchor, body).with_local_anchor1(Vec3::X),
                    JointForces::default(),
                ))
                .id()
        };
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(app.world().get::<LinearVelocity>(body).unwrap().0, velocity);
        assert_eq!(app.world().get::<AngularVelocity>(body).unwrap().0, spin);
        let forces = app.world().get::<JointForces>(joint).unwrap();
        assert_eq!(forces.force(), Vec3::ZERO);
        assert_eq!(forces.torque(), Vec3::ZERO);
    }
}

#[cfg(feature = "3d")]
#[test]
fn rejected_constraint_does_not_publish_partial_native_integration() {
    use crate::dynamics::{
        joints::EntityConstraint,
        solver::{
            solver_body::{SolverBody, SolverBodyInertia},
            xpbd::{
                ConstraintSolveError, XpbdConstraint, XpbdConstraintSolverData, XpbdSolverStatus,
                XpbdSolverSystems, solve_xpbd_joint,
            },
        },
    };
    use bevy::ecs::entity::{EntityMapper, MapEntities};
    #[derive(Component)]
    struct RejectingConstraint([Entity; 2]);
    impl MapEntities for RejectingConstraint {
        fn map_entities<M: EntityMapper>(&mut self, mapper: &mut M) {
            self.0 = self.0.map(|entity| mapper.get_mapped(entity));
        }
    }
    impl EntityConstraint<2> for RejectingConstraint {
        fn entities(&self) -> [Entity; 2] {
            self.0
        }
    }
    #[derive(Component, Default)]
    struct Data;
    impl XpbdConstraintSolverData for Data {}
    impl XpbdConstraint<2> for RejectingConstraint {
        type SolverData = Data;
        fn prepare(&mut self, _: [&RigidBodyQueryReadOnlyItem; 2], _: &mut Data) {}
        fn solve(
            &mut self,
            bodies: [&mut SolverBody; 2],
            _: [&SolverBodyInertia; 2],
            _: &mut Data,
            _: f32,
        ) -> Result<(), ConstraintSolveError> {
            bodies[0].delta_position = Vec3::splat(42.0);
            Err(ConstraintSolveError::NonFiniteSystem)
        }
    }
    let mut app = create_app();
    app.insert_resource(ExpectedConstraintFailure);
    app.add_systems(
        SubstepSchedule,
        solve_xpbd_joint::<RejectingConstraint>.in_set(XpbdSolverSystems::SolveUserConstraints),
    );
    app.finish();
    let mobile = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::Y),
            LinearVelocity(Vec3::X),
            Mass(1.0),
            AngularInertia::new(Vec3::ONE),
        ))
        .id();
    let anchor = app.world_mut().spawn(RigidBody::Static).id();
    let constraint = app
        .world_mut()
        .spawn((RejectingConstraint([mobile, anchor]), Data))
        .id();
    for _ in 0..4 {
        app.update();
    }
    let failure = app
        .world()
        .resource::<XpbdSolverStatus>()
        .failure
        .as_ref()
        .unwrap();
    assert_eq!(failure.constraints, vec![constraint]);
    assert_eq!(failure.reason, ConstraintSolveError::NonFiniteSystem);
    assert!(app.world().resource::<Time<Physics>>().is_paused());
    assert_eq!(app.world().get::<Position>(mobile).unwrap().0, RVector::Y);
    assert_eq!(
        app.world().get::<LinearVelocity>(mobile).unwrap().0,
        Vec3::X
    );
}

#[cfg(feature = "3d")]
#[test]
fn motor_stiffness_carries_load_transmitted_through_a_fixed_frame() {
    for angular in [false, true] {
        let mechanism_mass: f32 = if angular { 3.05 } else { 30.0 };
        let omega = (40000.0 / mechanism_mass).sqrt();
        for model in [
            MotorModel::ForceBased {
                stiffness: 40000.0,
                damping: 4000.0,
            },
            MotorModel::AccelerationBased {
                stiffness: 40000.0 / mechanism_mass,
                damping: 4000.0 / mechanism_mass,
            },
            MotorModel::SpringDamper {
                frequency: omega / core::f32::consts::TAU,
                damping_ratio: 4000.0 / (2.0 * mechanism_mass * omega),
            },
        ] {
            let mut app = create_app();
            app.insert_resource(SubstepCount(8));
            app.finish();
            let anchor = app.world_mut().spawn(RigidBody::Static).id();
            // Keep the mechanism awake to measure the drive's equilibrium,
            // independently of the native island's idle-motion threshold.
            let hub = app
                .world_mut()
                .spawn((
                    RigidBody::Dynamic,
                    SleepingDisabled,
                    Position(RVector::ZERO),
                    Mass(6.0),
                    AngularInertia::new(Vec3::splat(0.05)),
                ))
                .id();
            let load = app
                .world_mut()
                .spawn((
                    RigidBody::Dynamic,
                    SleepingDisabled,
                    Position(RVector::ZERO),
                    Mass(24.0),
                    AngularInertia::new(Vec3::splat(3.0)),
                ))
                .id();
            app.world_mut().spawn(FixedJoint::new(hub, load));
            if angular {
                app.world_mut()
                    .entity_mut(load)
                    .insert(ConstantTorque(Vec3::NEG_Y * 100.0));
                app.world_mut().spawn(
                    RevoluteJoint::new(anchor, hub)
                        .with_hinge_axis(Vec3::Y)
                        .with_angle_limits(-1.0, 1.0)
                        .with_motor(AngularMotor {
                            target_position: 0.06,
                            max_torque: 1000.0,
                            motor_model: model,
                            ..default()
                        }),
                );
            } else {
                app.world_mut()
                    .entity_mut(load)
                    .insert(ConstantForce(Vec3::NEG_Y * 100.0));
                app.world_mut().spawn(
                    PrismaticJoint::new(anchor, hub)
                        .with_slider_axis(Vec3::Y)
                        .with_limits(-1.0, 1.0)
                        .with_motor(LinearMotor {
                            target_position: 0.06,
                            max_force: 1000.0,
                            motor_model: model,
                            ..default()
                        }),
                );
            }
            for _ in 0..128 {
                app.update();
            }
            let coordinate = if angular {
                app.world()
                    .get::<Rotation>(hub)
                    .unwrap()
                    .0
                    .to_euler(EulerRot::YXZ)
                    .0
            } else {
                app.world().get::<Position>(hub).unwrap().y as f32
            };
            let expected = 0.06 - 100.0 / 40000.0;
            assert!(
                (coordinate - expected).abs() < 0.0001,
                "drive must retain physical stiffness across the attached load: angular={angular}, model={model:?}, measured={coordinate}, expected={expected}"
            );
        }
    }
}

#[cfg(feature = "3d")]
#[test]
fn joint_topology_changes_preserve_contact_velocity_projection() {
    let mut app = create_app();
    app.finish();
    let bodies: [_; 3] = core::array::from_fn(|_| app.world_mut().spawn(RigidBody::Dynamic).id());
    let joint = app
        .world_mut()
        .spawn(FixedJoint::new(bodies[0], bodies[1]))
        .id();
    app.update();
    app.world_mut()
        .entity_mut(joint)
        .insert(FixedJoint::new(bodies[0], bodies[2]));
    app.update();
    for body in bodies {
        assert!(app.world().get::<XpbdVelocityProjection>(body).is_some());
    }
    app.world_mut().entity_mut(joint).insert(JointDisabled);
    app.update();
    for body in bodies {
        assert!(app.world().get::<XpbdVelocityProjection>(body).is_some());
    }
    app.world_mut().despawn(joint);
    app.update();
    for body in bodies {
        assert!(app.world().get::<XpbdVelocityProjection>(body).is_some());
    }
}
