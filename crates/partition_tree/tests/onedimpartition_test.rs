//! Integration tests for partition_tree::onedimpartition
//!
//! These tests cover:
//! - Delegation of evaluate/volume/relative_volume to the rule
//! - mean() and integrate() delegating to the density
//! - split() behavior with and without custom densities
//! - Support for both continuous and categorical rules

use std::collections::HashSet;
use std::sync::Arc;

use partition_tree::density::{ConstantDensity, ConstantF64, HalfNormalDensity};
use partition_tree::onedimpartition::OneDimPartition;
use partition_tree::rules::{BelongsTo, ContinuousInterval, Rule};

// ---------------------------
// Continuous rule + ConstantF64 density
// ---------------------------

#[test]
fn onedimpartition_continuous_delegation_and_integrate() {
    // Rule: [0, 10] (closed) on domain [0, 10]
    let rule = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), false);
    let density = ConstantF64::new(2.5);
    let part = OneDimPartition::new(rule.clone(), density.clone());

    // evaluate() delegates to the rule
    let data = vec![
        Some(-1.0),
        Some(0.0),
        Some(5.0),
        Some(10.0),
        Some(10.1),
        None,
    ];
    assert_eq!(part.evaluate(&data), rule.evaluate(&data));

    // volume/relative_volume come from the rule
    assert_eq!(part.volume(), rule.volume());
    assert_eq!(part.relative_volume(), rule.relative_volume());

    // integrate() = c * volume
    assert!((part.integrate() - density.c * rule.volume()).abs() < f64::EPSILON);
}

#[test]
fn onedimpartition_continuous_split_with_custom_densities() {
    let rule = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), true);
    let base_density = ConstantF64::new(1.0);
    let part = OneDimPartition::new(rule, base_density);

    // Provide custom densities for left/right
    let left_d = ConstantF64::new(2.0);
    let right_d = ConstantF64::new(3.0);
    let (left, right) = part.split(4.0, Some(true), Some(left_d.clone()), Some(right_d.clone()));

    // Check rule bounds around split
    assert_eq!(left.rule.low, 0.0);
    assert_eq!(left.rule.high, 4.0);
    assert_eq!(left.rule.upper_closed, false);
    assert_eq!(left.rule.accept_none, true);

    assert_eq!(right.rule.low, 4.0);
    assert_eq!(right.rule.high, 10.0);
    assert_eq!(right.rule.lower_closed, true);
    assert_eq!(right.rule.accept_none, false);

    // Check densities carried over
    assert_eq!(left.density.c, 2.0);
    assert_eq!(right.density.c, 3.0);

    // Integrals match c * volume
    assert!((left.integrate() - left.density.c * left.rule.volume()).abs() < 1e-12);
    assert!((right.integrate() - right.density.c * right.rule.volume()).abs() < 1e-12);
}

#[test]
fn onedimpartition_continuous_split_clone_density() {
    let rule = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), false);
    let base_density = ConstantF64::new(5.0);
    let part = OneDimPartition::new(rule, base_density.clone());

    // Without providing densities, both sides should clone the base density
    let (left, right) = part.split(6.0, None, None, None);
    assert_eq!(left.density.c, base_density.c);
    assert_eq!(right.density.c, base_density.c);
}

// ---------------------------
// Categorical rule + ConstantDensity<usize>
// ---------------------------

#[test]
fn onedimpartition_categorical_constant_integrate() {
    let domain: Arc<Vec<usize>> = Arc::new((0..5).collect());
    let names: Arc<Vec<String>> = Arc::new(
        ["a", "b", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );

    // Values deduped, ordered by insertion order
    let rule = BelongsTo::new(
        [1usize, 4, 4, 2].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        false,
    );
    let density = ConstantDensity::<usize>::new(7.0);
    let part = OneDimPartition::new(rule.clone(), density.clone());

    // integrate = c * number of active categories
    assert!((part.integrate() - density.c * rule.volume()).abs() < f64::EPSILON);

    // Delegated evaluation sanity check
    let data = vec![Some(1usize), Some(0usize), Some(2usize), None];
    assert_eq!(part.evaluate(&data), rule.evaluate(&data));
}

// ---------------------------
// Continuous rule + HalfNormalDensity through OneDimPartition
// ---------------------------

#[test]
fn onedimpartition_halfnormal_integrate_segment() {
    let sigma = 2.0;
    let offset = 10.0;
    let side = 1.0; // support x >= offset
    let measure = 1.0;
    let density = HalfNormalDensity {
        scale: sigma,
        offset,
        side,
        measure,
    };

    // Integrate over [offset, offset + sigma*sqrt(2)] -> erf(1)
    let length = sigma * 2.0_f64.sqrt();
    let rule = ContinuousInterval::new(offset, offset + length, true, true, None, false);
    let part = OneDimPartition::new(rule, density);

    let got = part.integrate();
    let expected = statrs::function::erf::erf(1.0);
    assert!((got - expected).abs() < 1e-12);
}
