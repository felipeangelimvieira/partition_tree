//! A cell contains a set of one-dimensional partitions, one for each feature dimension.
//! It is responsible for:
//! * Evaluating a dataframe row-wise against its partitions

use crate::conf::*;
use crate::density::*;
use crate::dtype_adapter::*;
use crate::onedimpartition::*;
use crate::rules::*;
use core::panic;
use polars::prelude::*;
use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone)]
pub struct Cell {
    pub partitions: HashMap<String, Box<dyn DynOneDimPartition>>,
}

impl Cell {
    pub fn new() -> Self {
        Self {
            partitions: HashMap::new(),
        }
    }

    pub fn from_partitions(parts: HashMap<String, Box<dyn DynOneDimPartition>>) -> Self {
        Self { partitions: parts }
    }

    pub fn insert<S, R, D>(&mut self, name: S, p: OneDimPartition<R, D>)
    where
        S: Into<String>,
        R: Rule<D::Input> + 'static + Send + Sync + Clone,
        D: DensityFunction + 'static + Send + Sync + Clone,
        D::Input: 'static + Clone,
    {
        self.partitions.insert(name.into(), Box::new(p));
    }

    pub fn split(
        &self,
        name: &str,
        point: &dyn std::any::Any,
        none_to_left: Option<bool>,
        left_density: Option<&dyn std::any::Any>,
        right_density: Option<&dyn std::any::Any>,
    ) -> (Self, Self) {
        let part = self
            .partitions
            .get(name)
            .unwrap_or_else(|| panic!("Partition with name {name} not found"));

        let (left_p, right_p) = part.split_dyn(point, none_to_left, left_density, right_density);

        let mut left_cell = self.clone();
        let mut right_cell = self.clone();
        // replace (not insert-duplicate) the chosen partition
        left_cell.partitions.insert(name.to_string(), left_p);
        right_cell.partitions.insert(name.to_string(), right_p);
        (left_cell, right_cell)
    }

    /// Split a categorical partition by a subset of category codes.
    /// Categories in the subset go to the left child, others go to the right.
    pub fn split_categorical_subset(
        &self,
        name: &str,
        subset: &std::collections::HashSet<u32>,
        none_to_left: Option<bool>,
    ) -> (Self, Self) {
        let part = self
            .partitions
            .get(name)
            .unwrap_or_else(|| panic!("Partition with name {name} not found"));

        let (left_p, right_p) = part
            .split_subset_dyn(subset, none_to_left, None, None)
            .unwrap_or_else(|| {
                panic!("Partition {name} does not support subset splitting (not categorical)")
            });

        let mut left_cell = self.clone();
        let mut right_cell = self.clone();
        left_cell.partitions.insert(name.to_string(), left_p);
        right_cell.partitions.insert(name.to_string(), right_p);
        (left_cell, right_cell)
    }

    /// Evaluate a single dataframe **Series** against the stored partition for that column name.
    /// Returns a row mask (Vec<bool>) of matches.
    pub fn match_series_return_mask(&self, name: &str, s: &Series) -> Result<Vec<bool>, String> {
        let part = self
            .partitions
            .get(name)
            .ok_or_else(|| format!("No partition registered for column '{name}'"))?;

        let dtype_adapter = DtypeAdapter::new_from_dtype(s.dtype());
        dtype_adapter.evaluate_partition_for_series(s, part)
    }

    pub fn match_dataframe(
        &self,
        df: &DataFrame,
        subset: &UInt32Chunked,
        conf: TargetBehaviour,
    ) -> Result<UInt32Chunked, String> {
        if subset.is_empty() {
            return Ok(UInt32Chunked::from_iter_values(
                PlSmallStr::from_static("idx"),
                std::iter::empty(),
            ));
        }

        let mask = self.match_dataframe_return_mask(df, subset, conf)?;

        if mask.len() != subset.len() {
            return Err(format!(
                "Mask length ({}) does not match subset length ({})",
                mask.len(),
                subset.len()
            ));
        }

        let mut out: Vec<u32> = Vec::with_capacity(subset.len());

        for (pos, opt_idx) in subset.into_iter().enumerate() {
            if mask.get(pos).copied().unwrap_or(false) {
                if let Some(i) = opt_idx {
                    out.push(i);
                }
            }
        }

        Ok(UInt32Chunked::from_iter_values(
            PlSmallStr::from_static("idx"),
            out.into_iter(),
        ))
    }

    /// Build a boolean mask for the provided subset according to the configured partitions.
    pub fn match_dataframe_return_mask(
        &self,
        df: &DataFrame,
        subset: &UInt32Chunked,
        conf: TargetBehaviour,
    ) -> Result<Vec<bool>, String> {
        // New semantics: if the provided subset is empty, return an empty result.
        if subset.is_empty() {
            return Ok(Vec::new());
        }

        // Build a combined boolean mask across all included partitions (logical AND)
        // If no partitions are included by conf, treat mask as all-true.
        let mut combined_mask: Option<Vec<bool>> = None;

        for (name, _) in &self.partitions {
            if !conf.includes(name) {
                continue;
            }

            let column = df
                .column(name)
                .map_err(|_| format!("Column '{}' not found in DataFrame", name))?;

            let s_filtered = match column {
                Column::Scalar(scalar_column) => scalar_column.to_series(),
                _ => column
                    .as_series()
                    .ok_or_else(|| format!("Column '{}' has no Series representation", name))?
                    .clone()
                    .take(subset)
                    .map_err(|e| e.to_string())?,
            };
            let col_mask = self.match_series_return_mask(name, &s_filtered)?;

            match &mut combined_mask {
                None => combined_mask = Some(col_mask),
                Some(acc) => {
                    if acc.len() != col_mask.len() {
                        return Err(format!(
                            "Mask length mismatch for column '{}': {} vs {}",
                            name,
                            acc.len(),
                            col_mask.len()
                        ));
                    }
                    for (a, b) in acc.iter_mut().zip(col_mask.iter()) {
                        *a = *a && *b;
                    }
                }
            }
        }

        let height = subset.len();

        Ok(match combined_mask {
            Some(mask) => mask,
            None => vec![true; height],
        })
    }

    pub fn match_dataframe_xy(
        &self,
        df: &DataFrame,
        subset: &UInt32Chunked,
    ) -> Result<UInt32Chunked, String> {
        self.match_dataframe(df, subset, TargetBehaviour::Include)
    }

    pub fn match_dataframe_x(
        &self,
        df: &DataFrame,
        subset: &UInt32Chunked,
    ) -> Result<UInt32Chunked, String> {
        self.match_dataframe(df, subset, TargetBehaviour::Exclude)
    }

    pub fn match_dataframe_y(
        &self,
        df: &DataFrame,
        subset: &UInt32Chunked,
    ) -> Result<UInt32Chunked, String> {
        self.match_dataframe(df, subset, TargetBehaviour::Only)
    }
    pub fn match_hashmap(
        &self,
        row: &HashMap<String, Box<dyn Any>>,
        conf: TargetBehaviour,
    ) -> bool {
        let mut result = true;
        for (name, part) in &self.partitions {
            if !conf.includes(name) {
                continue;
            }

            let value_any = row
                .get(name)
                .unwrap_or_else(|| panic!("Column '{}' not found in input row", name));

            let value_ref = value_any.as_ref();
            let matches_res = if let Some(string_val) = Self::any_to_str(value_ref) {
                Self::evaluate_string_value(part.as_ref(), string_val)
            } else {
                part.eval_one_dyn(value_ref)
            };

            let matches = matches_res
                .map_err(|e| panic!("Error evaluating partition for column '{}': {}", name, e))
                .unwrap_or(false);

            result &= matches;
            if !result {
                break;
            }
        }
        result
    }

    fn any_to_str<'a>(value: &'a dyn Any) -> Option<&'a str> {
        if let Some(s) = value.downcast_ref::<String>() {
            Some(s.as_str())
        } else if let Some(s) = value.downcast_ref::<Cow<'static, str>>() {
            Some(s.as_ref())
        } else if let Some(s) = value.downcast_ref::<&'static str>() {
            Some(*s)
        } else {
            None
        }
    }

    fn evaluate_string_value(
        part: &dyn DynOneDimPartition,
        string_val: &str,
    ) -> Result<bool, String> {
        let rule_any = part.rule_any();

        macro_rules! try_eval_numeric {
            ($ty:ty) => {
                if let Some(rule) = rule_any.downcast_ref::<BelongsToGeneric<$ty>>() {
                    if let Some(idx) = rule.domain_names.iter().position(|name| name == string_val)
                    {
                        if let Some(value) = rule.domain.get(idx) {
                            return part.eval_one_dyn(value);
                        }
                    }
                    return Ok(false);
                }
            };
        }

        try_eval_numeric!(u8);
        try_eval_numeric!(u16);
        try_eval_numeric!(u32);
        try_eval_numeric!(u64);
        try_eval_numeric!(i8);
        try_eval_numeric!(i16);
        try_eval_numeric!(i32);
        try_eval_numeric!(i64);
        try_eval_numeric!(usize);

        if rule_any
            .downcast_ref::<BelongsToGeneric<String>>()
            .is_some()
        {
            let owned = string_val.to_string();
            return part.eval_one_dyn(&owned);
        }

        Ok(false)
    }
    pub fn get_partition(&self, name: &str) -> Option<&Box<dyn DynOneDimPartition>> {
        self.partitions.get(name)
    }

    pub fn get_rule(&self, name: &str) -> Option<&dyn Any> {
        self.partitions.get(name).map(|part| part.rule_any())
    }

    pub fn get_rule_as_continuous_interval(&self, name: &str) -> &ContinuousInterval {
        self.get_rule(name)
            .and_then(|r| r.downcast_ref::<ContinuousInterval>())
            .unwrap()
    }

    pub fn get_rule_as_belongs_to<T: 'static + RuleValue>(
        &self,
        name: &str,
    ) -> &BelongsToGeneric<T> {
        self.get_rule(name)
            .and_then(|r| r.downcast_ref::<BelongsToGeneric<T>>())
            .unwrap()
    }

    pub fn get_rule_as_belongs_to_u32(&self, name: &str) -> &BelongsToGeneric<u32> {
        self.get_rule_as_belongs_to::<u32>(name)
    }

    pub fn target_volume(&self) -> f64 {
        self.partitions
            .iter()
            .filter(|(name, _)| name.starts_with(TARGET_PREFIX))
            .map(|(_, part)| part.volume_dyn())
            .product()
    }

    pub fn target_relative_volume(&self) -> f64 {
        self.partitions
            .iter()
            .filter(|(name, _)| name.starts_with(TARGET_PREFIX))
            .map(|(_, part)| part.relative_volume_dyn())
            .product()
    }

    pub fn mean(&self) -> HashMap<String, Vec<f64>> {
        let mut means = HashMap::new();
        for (name, part) in &self.partitions {
            means.insert(name.clone(), part.mean_dyn());
        }
        means
    }

    pub fn project_to_target_cells(&self) -> Self {
        let target_parts: HashMap<String, Box<dyn DynOneDimPartition>> = self
            .partitions
            .iter()
            .filter(|(name, _)| name.starts_with(TARGET_PREFIX))
            .map(|(name, part)| (name.clone(), part.clone_box()))
            .collect();
        Cell::from_partitions(target_parts)
    }

    pub fn categorical_features(&self) -> Vec<String> {
        self.partitions
            .iter()
            .filter_map(|(name, part)| {
                if part.rule_any().is::<BelongsTo>() {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Non-target axis lengths using partition-provided metrics.
    /// Returns (name, length_metric, is_continuous).
    pub fn axis_lengths_non_target(&self) -> Vec<(String, f64, bool)> {
        self.partitions
            .iter()
            .filter(|(name, _)| !name.starts_with(TARGET_PREFIX))
            .map(|(name, part)| {
                let is_continuous = part.rule_any().is::<ContinuousInterval>();
                let len = part.phi_volume();
                (name.clone(), len, is_continuous)
            })
            .collect()
    }

    pub fn inverse_one_hot(&self, row: HashMap<String, Vec<f64>>) -> HashMap<String, Box<dyn Any>> {
        let mut result: HashMap<String, Box<dyn Any>> = HashMap::new();
        for (name, vec) in row {
            if let Some(part) = self.partitions.get(&name) {
                let rule_any = part.rule_any();
                if let Some(rule) = rule_any.downcast_ref::<BelongsTo>() {
                    let inv = rule.inverse_one_hot(&vec);
                    result.insert(name, Box::new(inv));
                } else if let Some(rule) = rule_any.downcast_ref::<BelongsToGeneric<String>>() {
                    let inv = rule.inverse_one_hot(&vec);
                    result.insert(name, Box::new(inv));
                } else if let Some(rule) = rule_any.downcast_ref::<ContinuousInterval>() {
                    let inv = rule.inverse_one_hot(&vec);
                    result.insert(name, Box::new(inv));
                } else {
                    result.insert(name, Box::new(vec));
                }
            } else {
                result.insert(name, Box::new(vec));
            }
        }
        result
    }

    pub fn get_domain(&self, column_name: &str) -> (f64, f64) {
        let rule = self
            .partitions
            .get(column_name)
            .and_then(|part| part.rule_any().downcast_ref::<ContinuousInterval>())
            .unwrap_or_else(|| {
                panic!(
                    "Partition for column '{}' is not a ContinuousInterval",
                    column_name
                )
            });
        rule.domain
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deterministic ordering for display
        let mut keys: Vec<&String> = self.partitions.keys().collect();
        keys.sort();

        write!(f, "Cell{{ ")?;

        let mut first = true;
        for key in keys {
            let part = &self.partitions[key];
            let rule_any = part.rule_any();

            // Helper to render BelongsToGeneric for concrete T that implements Display
            fn bt_str<T: RuleValue + fmt::Display>(any: &dyn Any) -> Option<String> {
                if let Some(r) = any.downcast_ref::<BelongsToGeneric<T>>() {
                    let vals: Vec<String> =
                        r.values.iter().cloned().map(|v| format!("{}", v)).collect();
                    return Some(format!("BelongsTo({})", vals.join(", ")));
                }
                None
            }

            let rule_str = if let Some(r) = rule_any.downcast_ref::<ContinuousInterval>() {
                format!("{}", r)
            } else if let Some(r) = rule_any.downcast_ref::<BelongsTo>() {
                // usize-coded variant has a specialized Display showing names
                format!("{}", r)
            } else if let Some(s) = bt_str::<u8>(rule_any)
                .or_else(|| bt_str::<u16>(rule_any))
                .or_else(|| bt_str::<u32>(rule_any))
                .or_else(|| bt_str::<u64>(rule_any))
                .or_else(|| bt_str::<i8>(rule_any))
                .or_else(|| bt_str::<i16>(rule_any))
                .or_else(|| bt_str::<i32>(rule_any))
                .or_else(|| bt_str::<i64>(rule_any))
                .or_else(|| bt_str::<bool>(rule_any))
                .or_else(|| bt_str::<String>(rule_any))
                .or_else(|| bt_str::<usize>(rule_any))
            {
                s
            } else {
                // Fallback when rule type is not recognized
                "<rule>".to_string()
            };

            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", key, rule_str)?;
            first = false;
        }

        write!(f, " }}")
    }
}
