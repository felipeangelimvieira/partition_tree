use crate::conf::TARGET_PREFIX;
use crate::dtype_adapter::*;
use crate::node::Node;
use core::fmt;
use core::panic;
use itertools::izip;
use polars::prelude::*;
use rand::prelude::*;
use rand::rng;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

#[derive(Debug, Clone, PartialEq)]
pub enum SplitResult {
    ContinuousSplit(f64, f64, Option<bool>),
    /// Categorical split: (subset of category codes, gain, none_to_left)
    /// The subset indicates which categories go to the LEFT child.
    CategoricalSplit(Vec<u32>, f64, Option<bool>),
    InvalidSplit(String),
}

impl SplitResult {
    pub fn gain(&self) -> f64 {
        match self {
            SplitResult::ContinuousSplit(_, gain, _) => *gain,
            SplitResult::CategoricalSplit(_, gain, _) => *gain,
            SplitResult::InvalidSplit(_) => -f64::INFINITY,
        }
    }

    /// Returns the subset of category codes for categorical splits
    pub fn categorical_subset(&self) -> Option<&Vec<u32>> {
        match self {
            SplitResult::CategoricalSplit(subset, _, _) => Some(subset),
            _ => None,
        }
    }

    pub fn loss(&self) -> f64 {
        -self.gain()
    }

    /// Returns the split value for continuous splits.
    /// For categorical splits, this returns the number of categories in the left subset.
    pub fn split_value(&self) -> f64 {
        match self {
            SplitResult::ContinuousSplit(value, _, _) => *value,
            SplitResult::CategoricalSplit(subset, _, _) => subset.len() as f64,
            SplitResult::InvalidSplit(msg) => {
                panic!("Cannot get split value from InvalidSplit: {}", msg)
            }
        }
    }

    pub fn is_valid(&self) -> bool {
        match self {
            SplitResult::InvalidSplit(_) => false,
            _ => true,
        }
    }
}

impl fmt::Display for SplitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitResult::ContinuousSplit(value, gain, none_to_left) => write!(
                f,
                "ContinuousSplit(value: {:.6}, gain: {:.6}, none_to_left: {:?})",
                value, gain, none_to_left
            ),
            SplitResult::CategoricalSplit(subset, gain, none_to_left) => write!(
                f,
                "CategoricalSplit(subset: {:?}, gain: {:.6}, none_to_left: {:?})",
                subset, gain, none_to_left
            ),
            SplitResult::InvalidSplit(msg) => write!(f, "InvalidSplit(msg: {})", msg),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SplitRestrictions {
    pub min_samples_split: usize,
    pub min_samples_leaf_y: usize,
    pub min_samples_leaf_x: usize,
    pub min_samples_leaf: usize,
    pub max_depth: usize,
    pub min_target_volume: f64,
    pub min_split_gain: f64,
    pub total_target_volume: f64,
    pub min_density_value: f64,
    pub max_density_value: f64,
    pub max_measure_value: f64,
    pub dataset_size: f64,
}

impl Default for SplitRestrictions {
    fn default() -> Self {
        Self {
            min_samples_split: 2,
            min_samples_leaf_y: 1,
            min_samples_leaf_x: 1,
            min_samples_leaf: 1,
            max_depth: usize::MAX,
            min_target_volume: 0.0,
            min_split_gain: 0.0_f64,
            total_target_volume: 1.0,
            min_density_value: 0.0,
            max_density_value: f64::INFINITY,
            max_measure_value: f64::INFINITY,
            dataset_size: 0.0,
        }
    }
}

impl SplitRestrictions {
    pub fn is_node_valid(&self, node: &Node) -> (bool, String) {
        // We no longer have partition on Node; approximate using x/y/xy counts only
        let n_xy = node.indices_xy.len();
        let n_x = node.indices_x.len();
        let n_y = node.indices_y.len();
        if node.depth >= self.max_depth {
            return (
                false,
                format!("depth ({}) >= max_depth ({})", node.depth, self.max_depth),
            );
        }

        if n_xy < self.min_samples_leaf {
            return (
                false,
                format!(
                    "xy_count ({}) < min_samples_leaf ({})",
                    n_xy, self.min_samples_leaf
                ),
            );
        }

        if n_x < self.min_samples_leaf_x {
            return (
                false,
                format!(
                    "x_count ({}) < min_samples_leaf_x ({})",
                    n_x, self.min_samples_leaf_x
                ),
            );
        }

        if n_y < self.min_samples_leaf_y {
            return (
                false,
                format!(
                    "y_count ({}) < min_samples_leaf_y ({})",
                    n_y, self.min_samples_leaf_y
                ),
            );
        }

        (true, "ok".to_string())
        // target volume constraint will be evaluated per-split since we don't have it here
    }

    #[allow(clippy::too_many_arguments)]
    pub fn is_point_valid(
        &self,
        xy_l: &f64,
        x_l: &f64,
        y_l: &f64,
        xy_r: &f64,
        x_r: &f64,
        y_r: &f64,
        target_volume: &f64,
        target_volume_left: &f64,
        target_volume_right: &f64,
        gain: &f64,
    ) -> (bool, String) {
        let cond_min_split = (xy_l + xy_r) >= self.min_samples_split as f64;
        let cond_right_xy_leaf = *xy_r >= self.min_samples_leaf as f64;
        let cond_left_xy_leaf = *xy_l >= self.min_samples_leaf as f64;
        let cond_left_x_leaf = *x_l >= self.min_samples_leaf_x as f64;
        let cond_right_x_leaf = *x_r >= self.min_samples_leaf_x as f64;
        let cond_left_y_leaf = *y_l >= self.min_samples_leaf_y as f64;
        let cond_right_y_leaf = *y_r >= self.min_samples_leaf_y as f64;
        let cond_target_volume = (*target_volume
            >= self.min_target_volume * self.total_target_volume)
            && (*target_volume_left >= self.min_target_volume * self.total_target_volume)
            && (*target_volume_right >= self.min_target_volume * self.total_target_volume);
        let cond_gain = *gain >= self.min_split_gain;

        if !cond_min_split {
            return (
                false,
                format!(
                    "total_xy ({}) < min_samples_split ({})",
                    (xy_l + xy_r),
                    self.min_samples_split
                ),
            );
        }

        if !cond_right_xy_leaf {
            return (
                false,
                format!(
                    "right_xy_count ({}) < min_samples_leaf ({})",
                    xy_r, self.min_samples_leaf
                ),
            );
        }

        if !cond_left_xy_leaf {
            return (
                false,
                format!(
                    "left_xy_count ({}) < min_samples_leaf ({})",
                    *xy_l, self.min_samples_leaf
                ),
            );
        }

        if !cond_left_x_leaf {
            return (
                false,
                format!(
                    "left_x_count ({}) < min_samples_leaf_x ({})",
                    *x_l, self.min_samples_leaf_x
                ),
            );
        }

        if !cond_right_x_leaf {
            return (
                false,
                format!(
                    "right_x_count ({}) < min_samples_leaf_x ({})",
                    *x_r, self.min_samples_leaf_x
                ),
            );
        }

        if !cond_left_y_leaf {
            return (
                false,
                format!(
                    "left_y_count ({}) < min_samples_leaf_y ({})",
                    *y_l, self.min_samples_leaf_y
                ),
            );
        }

        if !cond_right_y_leaf {
            return (
                false,
                format!(
                    "right_y_count ({}) < min_samples_leaf_y ({})",
                    *y_r, self.min_samples_leaf_y
                ),
            );
        }

        if !cond_target_volume {
            if *target_volume < self.min_target_volume {
                return (
                    false,
                    format!(
                        "target_volume ({}) < min_target_volume ({})",
                        *target_volume, self.min_target_volume
                    ),
                );
            }
            if *target_volume_left < self.min_target_volume {
                return (
                    false,
                    format!(
                        "target_volume_left ({}) < min_target_volume ({})",
                        *target_volume_left, self.min_target_volume
                    ),
                );
            }
            return (
                false,
                format!(
                    "target_volume_right ({}) < min_target_volume ({})",
                    *target_volume_right, self.min_target_volume
                ),
            );
        }

        if !cond_gain {
            return (
                false,
                format!(
                    "gain ({}) < min_split_gain ({})",
                    *gain, self.min_split_gain
                ),
            );
        }

        // Check density constraints for both children
        let left_density = if *xy_l > 0.0 && *target_volume_left > 0.0 {
            *xy_l / (*x_l * *target_volume_left)
        } else {
            0.0
        };
        let right_density = if *x_r > 0.0 && *target_volume_right > 0.0 {
            *xy_r / (*x_r * *target_volume_right)
        } else {
            0.0
        };

        if left_density < self.min_density_value {
            return (
                false,
                format!(
                    "left_density ({}) < min_density_value ({})",
                    left_density, self.min_density_value
                ),
            );
        }
        if right_density < self.min_density_value {
            return (
                false,
                format!(
                    "right_density ({}) < min_density_value ({})",
                    right_density, self.min_density_value
                ),
            );
        }
        if left_density > self.max_density_value {
            return (
                false,
                format!(
                    "left_density ({}) > max_density_value ({})",
                    left_density, self.max_density_value
                ),
            );
        }
        if right_density > self.max_density_value {
            return (
                false,
                format!(
                    "right_density ({}) > max_density_value ({})",
                    right_density, self.max_density_value
                ),
            );
        }

        (true, "ok".to_string())
    }
}

pub fn _subsample_columns(
    df: &DataFrame,
    max_features: Option<usize>,
    rng: &mut StdRng,
) -> Vec<PlSmallStr> {
    // Separate feature columns from target columns
    let all_cols: Vec<PlSmallStr> = df.get_column_names().into_iter().cloned().collect();

    let mut feature_cols: Vec<PlSmallStr> = all_cols
        .iter()
        .filter(|name| !name.starts_with(TARGET_PREFIX))
        .cloned()
        .collect();

    let target_cols: Vec<PlSmallStr> = all_cols
        .iter()
        .filter(|name| name.starts_with(TARGET_PREFIX))
        .cloned()
        .collect();

    // Shuffle only feature columns
    feature_cols.shuffle(rng);

    // If max_features is set, subsample only the feature columns
    let selected_features: Vec<PlSmallStr> = if let Some(maxf) = max_features {
        feature_cols.into_iter().take(maxf).collect()
    } else {
        feature_cols
    };

    // Always include all target columns along with the selected features
    let mut result = selected_features;
    result.extend(target_cols);
    result
}
pub fn find_best_split_column(
    node: &Node,
    df: &DataFrame,
    restrictions: &SplitRestrictions,
    max_features: Option<usize>,
    rng: &mut StdRng,
    sample_weights: &Float64Chunked,
    exclude_targets: bool,
) -> (String, SplitResult) {
    // Iterate over all columns
    // Separate target_* columns

    let mut col_names: Vec<PlSmallStr> = if max_features.is_some() {
        _subsample_columns(df, max_features, rng)
    } else {
        df.get_column_names()
            .into_iter()
            .cloned()
            .collect::<Vec<PlSmallStr>>()
    };

    // Filter out target columns if in forced-feature phase
    if exclude_targets {
        col_names.retain(|name| !name.starts_with(TARGET_PREFIX));
    }

    col_names.shuffle(rng);
    //println!("Considering cols {:?}", col_names);

    col_names
        .par_iter()
        .map(|col| {
            let split = search_split_column(node, df, col, restrictions, sample_weights);
            (col.to_string(), split)
        })
        .max_by(|a, b| {
            a.1.gain()
                .partial_cmp(&b.1.gain())
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or_else(|| {
            (
                String::new(),
                SplitResult::InvalidSplit("No valid split found".into()),
            )
        })
}

pub fn search_split_column(
    node: &Node,
    df: &DataFrame,
    column_name: &str,
    restrictions: &SplitRestrictions,
    sample_weights: &Float64Chunked,
) -> SplitResult {
    let (is_valid, reason) = restrictions.is_node_valid(node);
    if !is_valid {
        return SplitResult::InvalidSplit(format!("Node is not valid: {}", reason));
    }

    let column = match df.column(column_name) {
        Ok(s) => s,
        Err(_) => panic!("Column '{}' not found in DataFrame", column_name),
    };

    let dtype_adapter = DtypeAdapter::new_from_dtype(column.dtype());
    dtype_adapter.find_best_split_for_dtype(node, column, restrictions, sample_weights)
}

pub fn _search_sorted_right(arr1: &[f64], arr2: &[f64]) -> Vec<usize> {
    arr1.iter()
        .map(|v| arr2.partition_point(|x: &f64| *x <= *v))
        .collect()
}

pub fn _evaluate_split_points(
    count_xy_left: &Vec<f64>,
    count_x_left: &Vec<f64>,
    count_y_left: &Vec<f64>,
    count_xy_right: &Vec<f64>,
    count_x_right: &Vec<f64>,
    count_y_right: &Vec<f64>,
    target_volume: &f64,
    target_volume_children: &Vec<(f64, f64)>,
    restrictions: &SplitRestrictions,
) -> (Option<usize>, f64, Option<String>) {
    let mut max_gain = f64::NEG_INFINITY;
    let mut max_index: Option<usize> = None;
    let mut last_invalid_reason: Option<String> = None;
    for (i, (xy_l, x_l, y_l, xy_r, x_r, y_r, (left_vol, right_vol))) in izip!(
        count_xy_left.iter(),
        count_x_left.iter(),
        count_y_left.iter(),
        count_xy_right.iter(),
        count_x_right.iter(),
        count_y_right.iter(),
        target_volume_children.iter(),
    )
    .enumerate()
    {
        let gain = logistic_gain(
            &xy_l,
            &x_l,
            &y_l,
            &xy_r,
            &x_r,
            &y_r,
            target_volume,
            &left_vol,
            &right_vol,
            &restrictions.dataset_size,
        );
        let (is_valid, reason) = restrictions.is_point_valid(
            xy_l.into(),
            x_l.into(),
            y_l.into(),
            xy_r.into(),
            x_r.into(),
            y_r.into(),
            target_volume,
            &left_vol,
            &right_vol,
            &gain,
        );

        if !is_valid {
            last_invalid_reason = Some(reason);
        }

        if gain > max_gain && is_valid {
            max_gain = gain;
            max_index = Some(i);
        }
    }

    (max_index, max_gain, last_invalid_reason)
}

pub fn logistic_gain(
    xy_l: &f64,
    x_l: &f64,
    y_l: &f64,
    xy_r: &f64,
    x_r: &f64,
    y_r: &f64,
    target_volume: &f64,
    target_volume_left: &f64,
    target_volume_right: &f64,
    dataset_size: &f64,
) -> f64 {
    let xy_l = *xy_l as f64;
    let xy_r = *xy_r as f64;
    let x_l = *x_l as f64;
    let x_r = *x_r as f64;
    let _y_l = *y_l as f64;
    let _y_r = *y_r as f64;
    let total_xy = xy_l + xy_r;
    let total_x = x_l + x_r;

    // Guard against division by zero or invalid volumes
    if total_xy <= 0.0 || total_x <= 0.0 || *target_volume <= 0.0 || *dataset_size <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let left_log = if xy_l > 0.0 && x_l > 0.0 && *target_volume_left > 0.0 {
        (xy_l / (x_l * *target_volume_left)).ln() * (xy_l / total_xy)
    } else if xy_l > 0.0 {
        // xy_l > 0 but denominator is 0 or negative -> invalid split
        return f64::NEG_INFINITY;
    } else {
        0.0
    };

    let right_log = if xy_r > 0.0 && x_r > 0.0 && *target_volume_right > 0.0 {
        (xy_r / (x_r * *target_volume_right)).ln() * (xy_r / total_xy)
    } else if xy_r > 0.0 {
        // xy_r > 0 but denominator is 0 or negative -> invalid split
        return f64::NEG_INFINITY;
    } else {
        0.0
    };

    let parent_log = (total_xy / (total_x * *target_volume)).ln();

    let children_log = left_log + right_log;

    let gain = (children_log - parent_log) * total_xy / *dataset_size;

    gain
}

pub fn add_none_count(count: &Vec<f64>, none_count: &f64) -> Vec<f64> {
    count.iter().map(|&c| c + none_count).collect()
}
