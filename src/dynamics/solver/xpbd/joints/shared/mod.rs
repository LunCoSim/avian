mod fixed_angle_constraint;
mod point_constraint;

pub use fixed_angle_constraint::FixedAngleConstraintShared;
pub use point_constraint::PointConstraintShared;

#[cfg(feature = "3d")]
mod constraint_rows;
#[cfg(feature = "3d")]
pub(super) use constraint_rows::{
    ConstraintRow, coordinate_scale, jacobian_dot, point_rows, projected_inverse_mass,
    solve_bounded_rows, solve_constraint_rows,
};

#[cfg(feature = "3d")]
mod motor_constraint;
#[cfg(feature = "3d")]
pub(super) use motor_constraint::{
    MotorConstraint, MotorInput, prepare_pair_motor, rows_with_motor,
};
