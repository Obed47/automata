use nalgebra::{DMatrix, DVector};
use std::f64::EPSILON;

/// Sample quadratic function: f(x) = 0.5 * xᵀAx - bᵀx
fn objective_function(x: &DVector<f64>) -> f64 {
    let a = DMatrix::<f64>::from_vec(2, 2, vec![3.0, 2.0, 2.0, 6.0]); // symmetric positive definite
    let b = DVector::<f64>::from_vec(vec![2.0, -8.0]);
    (0.5 * x.transpose() * &a * x - b.transpose() * x)[(0, 0)]
}


fn gradient(x: &DVector<f64>) -> DVector<f64> {
    let a = DMatrix::<f64>::from_vec(2, 2, vec![3.0, 2.0, 2.0, 6.0]);
    let b = DVector::<f64>::from_vec(vec![2.0, -8.0]);
    &a * x - b
}

fn sr1_quasi_newton(
    initial_guess: DVector<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> DVector<f64> {
    let mut current_point = initial_guess.clone();
    let mut inverse_hessian_approx = DMatrix::<f64>::identity(current_point.len(), current_point.len());

    for iteration in 0..max_iterations {
        let current_gradient = gradient(&current_point);

        if current_gradient.norm() < tolerance {
            println!("Converged in {} iterations.", iteration);
            return current_point;
        }

        let search_direction = -&inverse_hessian_approx * &current_gradient;

        let step_size = 1.0;
        let next_point = &current_point + step_size * &search_direction;

        let s = &next_point - &current_point;
        let y = &gradient(&next_point) - &current_gradient;
        let diff = &y - &inverse_hessian_approx * &s;

        let denominator = diff.transpose() * &s;
        if denominator[(0, 0)].abs() > EPSILON {
            let update = (&diff * diff.transpose()) / denominator[(0, 0)];
            inverse_hessian_approx += update;
        }

        current_point = next_point;
    }

    println!("Reached max iterations.");
    current_point
}

fn main() {
    let initial_guess = DVector::<f64>::from_vec(vec![0.0, 0.0]);
    let tolerance = 1e-6;
    let max_iterations = 100;

    let optimal_point = sr1_quasi_newton(initial_guess, tolerance, max_iterations);

    println!("Optimal point: {}", optimal_point);
    println!("Objective function value: {}", objective_function(&optimal_point));
}
