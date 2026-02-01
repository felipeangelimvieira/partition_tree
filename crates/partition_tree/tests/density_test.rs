//! Integration tests for partition_tree::density
//!
//! These tests focus on:
//! - ConstantDensity for f64 and usize inputs (eval, integrate_rule over rules)
//! - HalfNormalDensity pdf/cdf relationships and integration over intervals
//! - mean() semantics for both densities

use std::collections::HashSet;
use std::sync::Arc;

use partition_tree::density::{ConstantDensity, ConstantF64, DensityFunction, HalfNormalDensity};
use partition_tree::rules::{BelongsTo, ContinuousInterval, Rule};

#[test]
fn constant_density_eval_and_integrate_continuous() {
    let c = 3.5;
    let d = ConstantF64::new(c);
    assert_eq!(d.eval(&0.0), c);
    assert_eq!(d.eval(&-123.456), c);
    assert!(d.mean().is_none());

    let rule = ContinuousInterval::new(-2.0, 2.0, true, false, Some((-10.0, 10.0)), false);
    // integrate = c * interval length
    assert!((d.integrate_rule(&rule) - c * rule.volume()).abs() < 1e-12);
}

#[test]
fn constant_density_eval_and_integrate_categorical() {
    let domain: Arc<Vec<usize>> = Arc::new((0..6).collect());
    let names: Arc<Vec<String>> = Arc::new(
        ["a", "b", "c", "d", "e", "f"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );

    let rule = BelongsTo::new(
        [1usize, 3, 5].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        true,
    );
    let d = ConstantDensity::<usize>::new(2.0);
    assert_eq!(d.eval(&0usize), 2.0);
    assert!(d.mean().is_none());
    assert!((d.integrate_rule(&rule) - 2.0 * rule.volume()).abs() < 1e-12);
}

#[test]
fn halfnormal_eval_symmetry_and_mean() {
    let sigma = 1.5;
    let offset = 3.0;
    let side = 1.0; // right side (x >= offset)
    let measure = 1.0;
    let d = HalfNormalDensity {
        scale: sigma,
        offset,
        side,
        measure,
    };

    // eval should be 0 below offset when side=+1
    assert_eq!(d.eval(&(offset - 1.0)), 0.0);
    // Positive for values above offset
    assert!(d.eval(&(offset + 0.1)) > 0.0);

    // mean = offset + side * sigma * sqrt(2/pi), scaled by measure
    let expected_mean = offset + side * sigma * (2.0 / std::f64::consts::PI).sqrt();
    let got = d.mean().unwrap();
    assert!((got[0] - expected_mean).abs() < 1e-12);
}

#[test]
fn halfnormal_integrate_known_values() {
    let sigma = 2.0;
    let offset = 10.0;
    let side = 1.0;
    let measure = 1.0;
    let d = HalfNormalDensity {
        scale: sigma,
        offset,
        side,
        measure,
    };

    // Integrate from offset to offset + sigma*sqrt(2) --> erf(1)
    let a = offset;
    let b = offset + sigma * 2.0_f64.sqrt();
    let rule = ContinuousInterval::new(a, b, true, true, None, false);
    let got = d.integrate_rule(&rule);
    let expected = statrs::function::erf::erf(1.0);
    assert!((got - expected).abs() < 1e-12);

    // Integrate from offset to infinity -> 1.0
    let rule_inf = ContinuousInterval::new(offset, f64::INFINITY, true, true, None, false);
    let got_inf = d.integrate_rule(&rule_inf);
    assert!((got_inf - 1.0).abs() < 1e-12);

    // Left side variant: side = -1.0, support x <= offset
    let d_left = HalfNormalDensity {
        scale: sigma,
        offset,
        side: -1.0,
        measure,
    };
    let rule_left = ContinuousInterval::new(f64::NEG_INFINITY, offset, true, true, None, false);
    let got_left = d_left.integrate_rule(&rule_left);
    assert!((got_left - 1.0).abs() < 1e-12);
}
