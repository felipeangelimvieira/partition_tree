//! Integration tests for partition_tree::rules
//!
//! These tests exercise the public API of rules:
//! - ContinuousInterval (continuous rule)
//! - BelongsTo (categorical rule)
//! - RuleType (enum wrapper)
//! - Rule trait default behavior (vector evaluation)
//!
//! The tests focus on correctness for evaluation semantics, handling of None,
//! set/interval algebra (split and bitand), as well as numeric characteristics
//! (volume, relative_volume, means) and Display/Into conversions.

use std::collections::HashSet;
use std::sync::Arc;

use partition_tree::rules::{BelongsTo, ContinuousInterval, Rule, RuleType};

// ---------------------------
// ContinuousInterval tests
// ---------------------------

/// Verify inclusive/exclusive bound checks and vector evaluation.
#[test]
fn continuous_interval_evaluate_inclusive_exclusive() {
    let ci = ContinuousInterval::new(0.0, 10.0, true, false, Some((0.0, 10.0)), false);

    let data = vec![
        Some(-0.1), // below lower -> false
        Some(0.0),  // equals lower, lower_closed=true -> true
        Some(5.0),  // strictly inside -> true
        Some(10.0), // equals upper, upper_closed=false -> false
        None,       // None, accept_none=false -> false via trait default
    ];

    let got = ci.evaluate(&data);
    assert_eq!(got, vec![false, true, true, false, false]);
}

/// Verify handling of None when accept_none is toggled.
#[test]
fn continuous_interval_none_handling() {
    let ci_false = ContinuousInterval::new(0.0, 1.0, true, true, None, false);
    let ci_true = ContinuousInterval::new(0.0, 1.0, true, true, None, true);

    assert_eq!(ci_false.evaluate(&[None]), vec![false]);
    assert_eq!(ci_true.evaluate(&[None]), vec![true]);
}

/// Check numeric characteristics: volume and relative volume across domains.
#[test]
fn continuous_interval_volume_and_relative() {
    // Full-domain interval [0,10] on domain [0,10] -> relative=1
    let ci = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), false);
    assert_eq!(ci.volume(), 10.0);
    assert_eq!(ci.relative_volume(), 1.0);

    // Infinite domain -> relative_volume returns 0.0 per implementation
    let ci_inf = ContinuousInterval::new(-5.0, 5.0, true, true, None, false);
    assert_eq!(ci_inf.relative_volume(), 0.0);

    // Zero-sized domain -> relative_volume returns 0.0
    let ci_zero_domain = ContinuousInterval::new(2.0, 3.0, true, true, Some((5.0, 5.0)), false);
    assert_eq!(ci_zero_domain.relative_volume(), 0.0);
}

/// Validate mean and Into<Vec<f64>> conversion (yields the interval midpoint).
#[test]
fn continuous_interval_mean_and_into_vec() {
    let ci = ContinuousInterval::new(2.0, 6.0, true, true, Some((0.0, 10.0)), false);
    let m = (2.0 + 6.0) / 2.0;
    assert_eq!(ci.mean(), vec![m]);

    let vec_mid: Vec<f64> = ci.clone().into();
    assert_eq!(vec_mid, vec![m]);
}

/// Verify split semantics and None routing.
#[test]
fn continuous_interval_split() {
    // Original interval [0,10] (both closed)
    let ci = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), true);

    // Split at 5.0, route None to left
    let (left, right) = ci.split(5.0, Some(true));

    // Left should be [0,5) (upper_closed=false)
    assert_eq!(left.low, 0.0);
    assert_eq!(left.high, 5.0);
    assert_eq!(left.lower_closed, true);
    assert_eq!(left.upper_closed, false);
    assert_eq!(left.accept_none, true);

    // Right should be [5,10] (lower_closed=true from split, upper_closed preserved)
    assert_eq!(right.low, 5.0);
    assert_eq!(right.high, 10.0);
    assert_eq!(right.lower_closed, true);
    assert_eq!(right.upper_closed, true);
    assert_eq!(right.accept_none, false);

    // Evaluate boundary behavior around the split point
    use std::f64::EPSILON;
    let below = 5.0 - 10.0 * EPSILON;
    let at = 5.0;
    let above = 5.0 + 10.0 * EPSILON;

    assert_eq!(
        left.evaluate(&[Some(0.0), Some(below), Some(at), Some(10.0)]),
        vec![true, true, false, false]
    );
    assert_eq!(
        right.evaluate(&[Some(below), Some(at), Some(above), Some(10.0)]),
        vec![false, true, true, true]
    );
}

/// Intersection of intervals using BitAnd preserves domain from the left operand
/// and combines closure and accept_none via logical AND.
#[test]
fn continuous_interval_bitand_intersection() {
    let a = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), true);
    let b = ContinuousInterval::new(5.0, 15.0, false, true, Some((-100.0, 100.0)), false);

    let c = a.clone() & b.clone();

    assert_eq!(c.low, 5.0);
    assert_eq!(c.high, 10.0);
    assert_eq!(c.lower_closed, false); // true && false
    assert_eq!(c.upper_closed, true); // true && true
    assert_eq!(c.accept_none, false); // true && false
    // Domain should come from left operand
    assert_eq!(c.domain, a.domain);

    // Sanity check evaluation
    assert_eq!(
        c.evaluate(&[Some(4.999999), Some(5.0), Some(10.0), Some(10.000001)]),
        vec![false, false, true, false]
    );
}

// ---------------------------
// BelongsTo tests
// ---------------------------

/// Construction preserves insertion order and de-duplicates values.
#[test]
fn belongs_to_new_order_and_dedup() {
    let domain: Arc<Vec<usize>> = Arc::new((0..4).collect());
    let names: Arc<Vec<String>> =
        Arc::new(["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect());

    let b = BelongsTo::new(
        [2usize, 2, 0, 3].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        false,
    );

    // Values are stored as a set (order-insensitive), duplicates removed
    assert_eq!(
        b.values,
        [2usize, 0, 3].into_iter().collect::<HashSet<usize>>()
    );
    let vals_set: HashSet<_> = b.values_as_usize().into_iter().collect();
    assert_eq!(
        vals_set,
        [Some(0usize), Some(2usize), Some(3usize)]
            .into_iter()
            .collect::<HashSet<Option<usize>>>()
    );
    let domain_set: HashSet<_> = b.domain_as_usize().into_iter().collect();
    assert_eq!(domain_set, (0..4).collect::<HashSet<usize>>());
}

/// Membership evaluation and None handling.
#[test]
fn belongs_to_evaluate_and_none() {
    let domain: Arc<Vec<usize>> = Arc::new((0..4).collect());
    let names: Arc<Vec<String>> =
        Arc::new(["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect());

    let b_false = BelongsTo::new(
        [2usize, 0, 3].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        false,
    );
    let b_true = BelongsTo::new(
        [2usize, 0, 3].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        true,
    );

    let data = vec![Some(2usize), None, Some(1usize), Some(0usize)];
    assert_eq!(b_false.evaluate(&data), vec![true, false, false, true]);
    assert_eq!(b_true.evaluate(&data), vec![true, true, false, true]);

    assert_eq!(b_false.volume(), 3.0);
    assert!((b_false.relative_volume() - 0.75).abs() < f64::EPSILON);

    // Mean returns the values themselves (as a set, order-insensitive)
    let mean_set: Vec<f64> = b_false.mean().into_iter().collect();
    assert_eq!(mean_set, [1.0, 0.0, 1.0, 1.0]);
}

/// Split keeps only the split point on the left (when present) and the rest on the right.
#[test]
fn belongs_to_split_point_present() {
    let domain: Arc<Vec<usize>> = Arc::new((0..5).collect());
    let names: Arc<Vec<String>> = Arc::new(
        ["a", "b", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );

    let b = BelongsTo::new(
        [4usize, 1, 3].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        true,
    );

    let (left, right) = b.split(1usize, Some(false)); // route None to right

    assert_eq!(left.values.len(), 1);
    assert!(left.values.contains(&1));
    assert_eq!(left.accept_none, false);

    assert_eq!(
        right.values,
        [4usize, 3].into_iter().collect::<HashSet<usize>>()
    );
    assert_eq!(right.accept_none, true);

    // Evaluate membership
    assert_eq!(
        left.evaluate(&[Some(1), Some(4), None]),
        vec![true, false, false]
    );
    assert_eq!(
        right.evaluate(&[Some(1), Some(4), None]),
        vec![false, true, true]
    );
}

/// BitAnd returns ordered intersection according to the domain order and
/// requires the same domain Arc for both operands.
#[test]
fn belongs_to_bitand_and_order() {
    let domain: Arc<Vec<usize>> = Arc::new((0..6).collect());
    let names: Arc<Vec<String>> = Arc::new(
        ["a", "b", "c", "d", "e", "f"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );

    // Use the same Arc clones to satisfy the internal assert
    let a = BelongsTo::new(
        [2usize, 0, 5].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        false,
    );
    let b = BelongsTo::new(
        [0usize, 1, 2, 3].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        true,
    );

    let inter = a & b;
    // Intersection should be {0, 2} (order-insensitive)
    assert_eq!(
        inter.values,
        [0usize, 2].into_iter().collect::<HashSet<usize>>()
    );
    assert_eq!(inter.accept_none, false); // false && true

    // Into<Vec<f64>> flags membership across the full domain
    let flags: Vec<f64> = inter.clone().into();
    // Order is domain-iteration dependent; just validate length and count of ones
    assert_eq!(flags.len(), domain.len());
    assert_eq!(flags.iter().filter(|x| **x == 1.0).count(), 2);

    // Display uses domain names corresponding to the selected indices
    let s = format!("{}", inter);
    // Display order is not guaranteed; compare as sets
    let disp = s
        .strip_prefix("BelongsTo(")
        .unwrap()
        .strip_suffix(")")
        .unwrap();
    let parts: HashSet<_> = disp.split(", ").map(|s| s.to_string()).collect();
    assert_eq!(
        parts,
        ["a".to_string(), "c".to_string()].into_iter().collect()
    );
}

// ---------------------------
// RuleType and Display tests
// ---------------------------

#[test]
fn rule_type_and_display() {
    // ContinuousInterval Display
    let ci = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), false);
    let s = format!("{}", ci);
    assert_eq!(s, "ContinuousInterval(0, 10, true, true, (0, 10))");

    // RuleType::is_categorical
    let domain: Arc<Vec<usize>> = Arc::new((0..3).collect());
    let names: Arc<Vec<String>> = Arc::new(["x", "y", "z"].iter().map(|s| s.to_string()).collect());
    let b = BelongsTo::new(
        [0usize, 2].into_iter().collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        false,
    );

    let rt_cont = RuleType::Continuous(ci);
    let rt_cat = RuleType::BelongsTo(b);

    assert_eq!(rt_cont.is_categorical(), false);
    assert_eq!(rt_cat.is_categorical(), true);
}
