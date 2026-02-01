//! Dynamic (type-erased) API tests for OneDimPartition
//!
//! These tests verify the behavior of `DynOneDimPartition` including:
//! - evaluate_dyn / eval_one_dyn on correct and incorrect input types
//! - integrate_dyn / volume_dyn / relative_volume_dyn
//! - mean_dyn downcasting when available
//! - rule_any / density_any downcasting
//! - split_dyn with and without custom densities, and none routing

use std::any::Any;
use std::collections::HashSet;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use partition_tree::density::{ConstantDensity, ConstantF64, HalfNormalDensity};
use partition_tree::onedimpartition::{DynOneDimPartition, OneDimPartition};
use partition_tree::rules::{BelongsTo, ContinuousInterval, Rule};

fn as_any<T: 'static>(t: &T) -> &dyn Any {
    t
}

#[test]
fn dyn_continuous_evaluate_and_integrate() {
    let rule = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), true);
    let density = ConstantF64::new(2.0);
    let part = OneDimPartition::new(rule.clone(), density.clone());
    let dp: Box<dyn DynOneDimPartition> = Box::new(part.clone());

    // evaluate_dyn supports Vec<Option<f64>>
    let data_vec = vec![
        Some(-1.0),
        Some(0.0),
        Some(5.0),
        Some(10.0),
        Some(10.1),
        None,
    ];
    let got = dp.evaluate_dyn(as_any(&data_vec)).expect("correct type");
    let exp = rule.evaluate(&data_vec);
    assert_eq!(got, exp);

    // evaluate_dyn supports Box<[Option<f64>]> as well
    let data_box: Box<[Option<f64>]> = vec![Some(1.0), None, Some(11.0)].into_boxed_slice();
    let got2 = dp
        .evaluate_dyn(as_any(&data_box))
        .expect("boxed slice type");
    let exp2 = rule.evaluate(&data_box);
    assert_eq!(got2, exp2);

    // integrate/volume/relative_volume delegated correctly
    assert!((dp.integrate_dyn() - part.integrate()).abs() < 1e-12);
    assert!((dp.volume_dyn() - part.volume()).abs() < 1e-12);
    assert!((dp.relative_volume_dyn() - part.relative_volume()).abs() < 1e-12);
}

#[test]
fn dyn_continuous_eval_one_and_type_errors() {
    let rule = ContinuousInterval::new(-5.0, 5.0, true, false, None, false);
    let density = ConstantF64::new(1.0);
    let dp: Box<dyn DynOneDimPartition> = Box::new(OneDimPartition::new(rule, density));

    // Some(x) inside interval
    let res_true = dp.eval_one_dyn(as_any(&Some(0.0))).expect("correct type");
    assert!(res_true);

    // None should be false since accept_none = false
    let res_none = dp.eval_one_dyn(as_any(&None::<f64>)).expect("correct type");
    assert!(!res_none);

    // Wrong input type -> Err (not panic)
    let err = dp
        .evaluate_dyn(as_any(&vec![Some(1i32), Some(2i32)]))
        .expect_err("type mismatch should return Err");
    assert!(err.contains("Type mismatch"));

    let err2 = dp
        .eval_one_dyn(as_any(&Some(123i32)))
        .expect_err("type mismatch should return Err");
    assert!(err2.contains("Type mismatch"));
}

#[test]
fn dyn_continuous_rule_and_density_downcast() {
    let rule = ContinuousInterval::new(1.0, 2.0, true, true, None, true);
    let density = ConstantF64::new(3.0);
    let dp: Box<dyn DynOneDimPartition> =
        Box::new(OneDimPartition::new(rule.clone(), density.clone()));

    // Downcast rule_any
    let r_any = dp.rule_any();
    let r = r_any
        .downcast_ref::<ContinuousInterval>()
        .expect("rule downcast");
    assert_eq!(r.low, 1.0);
    assert_eq!(r.high, 2.0);

    // Downcast density_any
    let d_any = dp.density_any();
    let d = d_any
        .downcast_ref::<ConstantF64>()
        .expect("density downcast");
    assert_eq!(d.c, 3.0);
}

#[test]
fn dyn_continuous_split_with_custom_densities_and_none_routing() {
    let rule = ContinuousInterval::new(0.0, 10.0, true, true, Some((0.0, 10.0)), true);
    let base = ConstantF64::new(1.0);
    let dp: Box<dyn DynOneDimPartition> = Box::new(OneDimPartition::new(rule.clone(), base));

    let left_d = ConstantF64::new(2.0);
    let right_d = ConstantF64::new(5.0);

    // Split at 4.0; route None to left
    let (l, r) = dp.split_dyn(
        as_any(&4.0),
        Some(true),
        Some(as_any(&left_d)),
        Some(as_any(&right_d)),
    );

    // Inspect left rule/density
    let l_rule = l.rule_any().downcast_ref::<ContinuousInterval>().unwrap();
    assert_eq!(l_rule.low, 0.0);
    assert_eq!(l_rule.high, 4.0);
    assert!(l_rule.lower_closed);
    assert!(!l_rule.upper_closed);
    assert!(l_rule.accept_none);

    let l_den = l.density_any().downcast_ref::<ConstantF64>().unwrap();
    assert_eq!(l_den.c, 2.0);

    // Inspect right rule/density
    let r_rule = r.rule_any().downcast_ref::<ContinuousInterval>().unwrap();
    assert_eq!(r_rule.low, 4.0);
    assert_eq!(r_rule.high, 10.0);
    assert!(r_rule.lower_closed);
    assert!(r_rule.upper_closed);
    assert!(!r_rule.accept_none);

    let r_den = r.density_any().downcast_ref::<ConstantF64>().unwrap();
    assert_eq!(r_den.c, 5.0);
}

#[test]
fn dyn_continuous_split_clone_density_when_none_provided() {
    let rule = ContinuousInterval::new(0.0, 10.0, true, true, None, false);
    let base = ConstantF64::new(7.0);
    let dp: Box<dyn DynOneDimPartition> = Box::new(OneDimPartition::new(rule, base.clone()));

    let (l, r) = dp.split_dyn(as_any(&6.0), None, None, None);
    let l_den = l.density_any().downcast_ref::<ConstantF64>().unwrap();
    let r_den = r.density_any().downcast_ref::<ConstantF64>().unwrap();
    assert_eq!(l_den.c, base.c);
    assert_eq!(r_den.c, base.c);
}

#[test]
fn dyn_mismatch_panics_in_split_dyn() {
    let rule = ContinuousInterval::new(0.0, 1.0, true, true, None, false);
    let base = ConstantF64::new(1.0);
    let dp: Box<dyn DynOneDimPartition> = Box::new(OneDimPartition::new(rule, base));

    // Wrong point type should panic
    let res = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = dp.split_dyn(as_any(&1usize), None, None, None);
    }));
    assert!(res.is_err());

    // Wrong density type should panic
    let wrong_left = ConstantDensity::<usize>::new(2.0);
    let res2 = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = dp.split_dyn(as_any(&0.5f64), None, Some(as_any(&wrong_left)), None);
    }));
    assert!(res2.is_err());
}

#[test]
fn dyn_categorical_basic() {
    let domain: Arc<Vec<usize>> = Arc::new((0..6).collect());
    let names: Arc<Vec<String>> = Arc::new(
        ["a", "b", "c", "d", "e", "f"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );
    let rule = BelongsTo::new(
        [1usize, 3usize, 5usize]
            .into_iter()
            .collect::<HashSet<usize>>(),
        Arc::clone(&domain),
        Arc::clone(&names),
        true,
    );
    let density = ConstantDensity::<usize>::new(4.0);

    let part = OneDimPartition::new(rule.clone(), density.clone());
    let dp: Box<dyn DynOneDimPartition> = Box::new(part.clone());

    // evaluate_dyn
    let data = vec![Some(0usize), Some(1usize), None, Some(4usize), Some(5usize)];
    let got = dp.evaluate_dyn(as_any(&data)).expect("correct type");
    let exp = rule.evaluate(&data);
    assert_eq!(got, exp);

    // integrate = c * count(values)
    assert!((dp.integrate_dyn() - density.c * rule.volume()).abs() < 1e-12);

    // volume/relative
    assert_eq!(dp.volume_dyn(), part.volume());
    assert_eq!(dp.relative_volume_dyn(), part.relative_volume());

    // split_dyn at value 3: left keeps 3, right drops it
    let (l, r) = dp.split_dyn(as_any(&3usize), Some(false), None, None);

    let l_rule = l.rule_any().downcast_ref::<BelongsTo>().unwrap();
    assert_eq!(l_rule.values_as_usize(), vec![Some(3)]);
    assert!(!l_rule.values_as_usize().is_empty());
    assert!(l_rule.accept_none == false); // none routed to right

    let r_rule = r.rule_any().downcast_ref::<BelongsTo>().unwrap();
    // values_as_usize returns values as Vec<Option<usize>>; order not guaranteed
    let r_set: HashSet<_> = r_rule.values_as_usize().into_iter().collect();
    let exp_set: HashSet<_> = vec![Some(1usize), Some(5usize)].into_iter().collect();
    assert_eq!(r_set, exp_set);
    assert!(r_rule.accept_none);
}
