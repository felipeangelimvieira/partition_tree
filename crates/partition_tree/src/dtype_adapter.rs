use crate::cell::*;
use crate::conf::*;
use crate::dataframe::*;
use crate::density::*;
use crate::node::*;
use crate::onedimpartition::*;
use crate::rules::*;
use crate::split::*;
use itertools::sorted;
use polars::prelude::*;
use rayon::Yield;
use rayon::iter::Split;
use statrs::statistics::Data;
use std::any::Any;
use std::char::MAX;
use std::collections::{HashMap, HashSet};
pub struct DtypeAdapter {
    pub data_type: DataType,
}

impl DtypeAdapter {
    pub fn new_from_dtype(data_type: &DataType) -> Self {
        Self {
            data_type: data_type.clone(),
        }
    }
    pub fn new(series: &Series) -> Self {
        Self {
            data_type: series.dtype().clone(),
        }
    }
}

impl DtypeAdapter {
    // ---- Default partition for a given dtype ----
    pub fn default_partition(
        &self,
        s: &Series,
        boundaries_expansion_factor: f64,
    ) -> Box<dyn DynOneDimPartition> {
        match s.dtype() {
            DataType::Float32 | DataType::Float64 => {
                Box::new(self._default_continuous_interval(s, boundaries_expansion_factor))
            }
            DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                Box::new(self._default_belongs_to_enum(s))
            }
            DataType::String => Box::new(self._default_belongs_to_string(s)),
            dt => panic!("default_partition not implemented for this dtype {:?}", dt),
        }
    }

    fn _default_continuous_interval(
        &self,
        s: &Series,
        boundaries_expansion_factor: f64,
    ) -> OneDimPartition<ContinuousInterval, ConstantF64> {
        let low: f64;
        let high: f64;

        if s.name().starts_with(TARGET_PREFIX) {
            let low_raw = s.min().expect("min value").unwrap_or(f64::NEG_INFINITY);
            let high_raw = s.max().expect("max value").unwrap_or(f64::INFINITY);

            let range = high_raw - low_raw;
            let mean = (high_raw + low_raw) / 2.0;
            let expansion = range * (1.0 + boundaries_expansion_factor);
            low = mean - expansion / 2.0;
            high = mean + expansion / 2.0;
        } else {
            low = f64::NEG_INFINITY;
            high = f64::INFINITY;
        }

        let rule = ContinuousInterval::new(low, high, true, true, Some((low, high)), true);
        let density = ConstantF64::new(1.0 / rule.volume());
        OneDimPartition::new(rule, density)
    }

    fn _default_belongs_to_enum(&self, s: &Series) -> OneDimPartition<BelongsToU32, ConstantU32> {
        match s.dtype() {
            DataType::Enum(_, categorical_mapping)
            | DataType::Categorical(_, categorical_mapping) => {
                // Extract the physical representation (UInt32 codes) from the Enum column
                let phys = s.to_physical_repr();
                let phys_u32 = phys
                    .cast(&DataType::UInt32)
                    .expect("Cast Enum physical to UInt32");
                let ca = phys_u32.u32().expect("Convert Enum physical to u32");
                let mut seen: HashSet<u32> = HashSet::new();
                let mut domain: Vec<u32> = Vec::new();
                for value in ca.into_no_null_iter() {
                    if seen.insert(value) {
                        domain.push(value);
                    }
                }

                let names: Vec<String> = domain
                    .iter()
                    .map(|v| categorical_mapping.cat_to_str(*v).unwrap().to_string())
                    .collect();
                let values: HashSet<u32> = domain.iter().copied().collect();

                let rule =
                    BelongsToU32::new(values, Arc::new(domain.clone()), Arc::new(names), true);
                let density = ConstantU32::new(1.0 / rule.volume());
                OneDimPartition::new(rule, density)
            }

            _ => panic!("Not implemented for this dtype"),
        }
    }

    fn _default_belongs_to_string(
        &self,
        s: &Series,
    ) -> OneDimPartition<BelongsToString, ConstantDensity<String>> {
        match s.dtype() {
            DataType::String => {
                let ca = s.str().expect("String series expected");
                let mut seen: HashSet<String> = HashSet::new();
                let mut domain: Vec<String> = Vec::new();

                for opt_value in ca.into_iter() {
                    if let Some(value) = opt_value {
                        if !seen.contains(value) {
                            let owned = value.to_owned();
                            seen.insert(owned.clone());
                            domain.push(owned);
                        }
                    }
                }

                let values: HashSet<String> = domain.iter().cloned().collect();
                let rule = BelongsToString::new(
                    values,
                    Arc::new(domain.clone()),
                    Arc::new(domain.clone()),
                    true,
                );

                let density_scale = if rule.volume() == 0.0 {
                    0.0
                } else {
                    1.0 / rule.volume()
                };
                let density = ConstantDensity::<String>::new(density_scale);

                OneDimPartition::new(rule, density)
            }
            dt => panic!(
                "default_belongs_to_string not implemented for this dtype {:?}",
                dt
            ),
        }
    }

    //  ------------- Find best split -------------
    pub fn find_best_split_for_dtype(
        &self,
        node: &Node,
        column: &Column,
        restrictions: &SplitRestrictions,
        sample_weights: &Float64Chunked,
    ) -> SplitResult {
        match self.data_type {
            DataType::Float64 => find_best_continuous_split_series(
                node,
                column.as_series().unwrap(),
                restrictions,
                sample_weights,
            ),
            DataType::Int32 => {
                let s_f = column.cast(&DataType::Float64).unwrap();
                find_best_continuous_split_series(
                    node,
                    s_f.as_series().unwrap(),
                    restrictions,
                    sample_weights,
                )
            }

            DataType::Int64 => {
                let s_f = column.cast(&DataType::Float64).unwrap();
                find_best_continuous_split_series(
                    node,
                    s_f.as_series().unwrap(),
                    restrictions,
                    sample_weights,
                )
            }
            DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                find_best_categorical_split_series(
                    node,
                    column.as_series().unwrap(),
                    restrictions,
                    sample_weights,
                )
            }
            _ => panic!("Unsupported dtype for splitting: {:?}", column.dtype()),
        }
    }

    pub fn predict_mean_for_dtype(
        &self,
        root_cell: &Cell,
        col_name: &PlSmallStr,
        means: Vec<Vec<f64>>,
    ) -> Column {
        match &self.data_type {
            DataType::Float64 => {
                let means_for_col = means
                    .iter()
                    .map(|v| {
                        root_cell
                            .get_rule_as_continuous_interval(col_name)
                            .inverse_one_hot(v)
                    })
                    .collect::<Vec<f64>>();
                Column::new(col_name.clone(), means_for_col)
            }
            DataType::Enum(_, categorical_mapping)
            | DataType::Categorical(_, categorical_mapping) => {
                let means_for_col = means
                    .iter()
                    .map(|v| {
                        root_cell
                            .get_rule_as_belongs_to_u32(col_name)
                            .inverse_one_hot(v)
                    })
                    .collect::<Vec<u32>>();

                let classes = means_for_col
                    .iter()
                    .map(|&code| categorical_mapping.cat_to_str(code))
                    .collect::<Vec<Option<&str>>>();
                Column::new(col_name.clone(), classes)
            }
            _ => panic!("Not implemented"),
        }
    }

    pub fn evaluate_partition_for_series(
        &self,
        s: &Series,
        part: &Box<dyn DynOneDimPartition>,
    ) -> Result<Vec<bool>, String> {
        match &self.data_type {
            DataType::Float64 => {
                let ca = s.f64().map_err(|e| e.to_string())?;
                let v: Vec<Option<f64>> = ca.into_iter().collect();
                part.evaluate_dyn(&v)
            }
            DataType::Int32 => {
                let ca = s.i32().map_err(|e| e.to_string())?;
                let v: Vec<Option<i32>> = ca.into_iter().collect();
                part.evaluate_dyn(&v)
            }
            DataType::Boolean => {
                let ca = s.bool().map_err(|e| e.to_string())?;
                let v: Vec<Option<bool>> = ca.into_iter().collect();
                part.evaluate_dyn(&v)
            }
            DataType::String => {
                let ca = s.str().map_err(|e| e.to_string())?;
                let v: Vec<Option<String>> = ca
                    .into_iter()
                    .map(|opt| opt.map(|value| value.to_string()))
                    .collect();
                part.evaluate_dyn(&v)
            }
            // Categorical/Enum columns have a UInt32 physical backing in Polars
            DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                // Access the physical representation and cast to UInt32 to cover u8/u16/u32
                let phys = s.to_physical_repr();
                let phys_u32 = phys.cast(&DataType::UInt32).map_err(|e| e.to_string())?;
                let ca = phys_u32.u32().map_err(|e| e.to_string())?;
                let v: Vec<Option<u32>> = ca.into_iter().collect();
                part.evaluate_dyn(&v)
            }
            // Allow plain UInt32 too, if your pipeline uses it
            DataType::UInt32 => {
                let ca = s.u32().map_err(|e| e.to_string())?;
                let v: Vec<Option<u32>> = ca.into_iter().collect();
                part.evaluate_dyn(&v)
            }
            dt => Err(format!(
                "Unsupported dtype for column '{:?}' of dtype '{:?}'",
                s.name(),
                dt
            )),
        }
    }
}

fn find_best_continuous_split_series(
    node: &Node,
    series: &Series,
    restrictions: &SplitRestrictions,
    sample_weights: &Float64Chunked,
) -> SplitResult {
    let res_xy = match series.sorted_numeric_values::<Float64Type>(
        &node.indices_xy,
        true,
        &sample_weights,
    ) {
        Ok(res) => res,
        Err(_) => return SplitResult::InvalidSplit("Error sorting numeric xy values".into()),
    };

    let res_x =
        match series.sorted_numeric_values::<Float64Type>(&node.indices_x, true, &sample_weights) {
            Ok(v) => v,
            Err(_) => return SplitResult::InvalidSplit("Error sorting numeric x values".into()),
        };
    let res_y =
        match series.sorted_numeric_values::<Float64Type>(&node.indices_y, true, &sample_weights) {
            Ok(v) => v,
            Err(_) => return SplitResult::InvalidSplit("Error sorting numeric y values".into()),
        };

    let NumericSortResult {
        sorted_values: sorted_values_xy,
        unique_values: unique_values_xy,
        sorted_weights: sorted_weights_xy,
        none_count: none_count_xy,
        none_weight: none_weight_xy,
    } = res_xy;

    let is_target_column = series.name().starts_with(TARGET_PREFIX);
    // Maybe add conditions at the borders?

    let mut edges = match is_target_column {
        true => res_y.unique_values.clone(),
        false => res_x.unique_values.clone(),
    };
    //if edges.len() == 1 && is_target_column {
    //    add_domain_to_edges_if_finite(&mut edges, node.cell.get_domain(series.name()));
    //}

    let mut candidate_split_points: Vec<f64>;
    if edges.len() <= 1 {
        candidate_split_points = vec![];
    } else {
        candidate_split_points = edges.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
    }

    if candidate_split_points.is_empty() {
        return SplitResult::InvalidSplit(
            format!(
                "candidate_split_points is empty. Unique_x is {:?}, parent node.indices_x.len() is {}, unique_y is {:?}, parent node.indices_y.len() is {}. ",
                res_x.unique_values, node.indices_x.len(), res_y.unique_values, node.indices_y.len()
            )
            .into(),
        );
    }

    if candidate_split_points.len() > MAX_CANDIDATE_SPLIT_POINTS {
        // Select MAX_CANDIDATE_SPLIT_POINTS evenly spaced points
        let step = candidate_split_points.len() / MAX_CANDIDATE_SPLIT_POINTS;
        candidate_split_points = candidate_split_points
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % step == 0)
            .map(|(_, v)| v)
            .collect();
    }

    let sorted_values_x = res_x.sorted_values;
    let sorted_weights_x = res_x.sorted_weights;
    let none_weight_x: f64 = res_x.none_weight;
    let sorted_values_y = res_y.sorted_values;
    let sorted_weights_y = res_y.sorted_weights;
    let none_weight_y = res_y.none_weight;

    let cum_weights_xy: Vec<f64> = sorted_weights_xy
        .iter()
        .scan(0.0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();
    let cum_weights_x: Vec<f64> = sorted_weights_x
        .iter()
        .scan(0.0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();
    let cum_weights_y: Vec<f64> = sorted_weights_y
        .iter()
        .scan(0.0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();

    let total_weight_xy: f64 = cum_weights_xy.last().copied().unwrap_or(0.0);
    let total_weight_x: f64 = cum_weights_x.last().copied().unwrap_or(0.0);
    let total_weight_y: f64 = cum_weights_y.last().copied().unwrap_or(0.0);

    // Find index and map to n_{xy}(A_l) and n_{xy}(A_r)
    // v is the number of elements <= split_point
    // If v = 0, no elements are to the left, so weight_left = 0
    // If v >= len, all elements are to the left, so weight_left = total
    // Otherwise, weight_left = cum_weights[v-1] (sum of first v elements)
    let weight_xy_left: Vec<f64> = _search_sorted_right(&candidate_split_points, &sorted_values_xy)
        .into_iter()
        .map(|v| {
            if v == 0 {
                0.0 // No samples to the left of split point
            } else if v >= cum_weights_xy.len() {
                total_weight_xy
            } else {
                cum_weights_xy[v - 1] // Sum of first v elements
            }
        })
        .collect();
    let weight_xy_right: Vec<f64> = weight_xy_left
        .iter()
        .map(|&v| (total_weight_xy - v))
        .collect();
    let weight_x_left: Vec<f64>;
    let weight_y_left: Vec<f64>;
    let weight_x_right: Vec<f64>;
    let weight_y_right: Vec<f64>;

    let target_volume = node.cell.target_volume();
    let target_volume_children: Vec<(f64, f64)>;

    if series.name().starts_with(TARGET_PREFIX) {
        weight_x_left = vec![total_weight_x; candidate_split_points.len()];
        weight_x_right = weight_x_left.clone();
        weight_y_left = _search_sorted_right(&candidate_split_points, &sorted_values_y)
            .into_iter()
            .map(|v| {
                if v == 0 {
                    0.0
                } else if v >= cum_weights_y.len() {
                    total_weight_y
                } else {
                    cum_weights_y[v - 1]
                }
            })
            .collect();
        weight_y_right = weight_y_left
            .iter()
            .map(|&v| (total_weight_y) - v)
            .collect();

        target_volume_children = candidate_split_points
            .iter()
            .map(|split_value: &f64| {
                let (left_cell, right_cell) =
                    node.cell
                        .split(series.name(), split_value, None, None, None);
                let left_volume = left_cell.target_volume();
                let right_volume = right_cell.target_volume();

                (left_volume, right_volume)
            })
            .collect::<Vec<(f64, f64)>>();
    } else {
        weight_x_left = _search_sorted_right(&candidate_split_points, &sorted_values_x)
            .into_iter()
            .map(|v| {
                if v == 0 {
                    0.0
                } else if v >= cum_weights_x.len() {
                    total_weight_x
                } else {
                    cum_weights_x[v - 1]
                }
            })
            .collect();
        weight_x_right = weight_x_left
            .iter()
            .map(|&v| (total_weight_x) - v)
            .collect();

        weight_y_left = vec![total_weight_y; candidate_split_points.len()];
        weight_y_right = weight_y_left.clone();

        target_volume_children = vec![(target_volume, target_volume); candidate_split_points.len()];
    }

    let target_volume = &node.cell.target_volume();

    let (max_index_none_left, gain_left, _) = _evaluate_split_points(
        &add_none_count(&weight_xy_left, &none_weight_xy),
        &add_none_count(&weight_x_left, &none_weight_x),
        &add_none_count(&weight_y_left, &none_weight_y),
        &weight_xy_right,
        &weight_x_right,
        &weight_y_right,
        &target_volume,
        &target_volume_children,
        restrictions,
    );

    let (max_index_none_right, gain_right, last_invalid_reason) = _evaluate_split_points(
        &weight_xy_left,
        &weight_x_left,
        &weight_y_left,
        &add_none_count(&weight_xy_right, &none_weight_xy),
        &add_none_count(&weight_x_right, &none_weight_x),
        &add_none_count(&weight_y_right, &none_weight_y),
        &target_volume,
        &target_volume_children,
        restrictions,
    );

    let is_left_greater = gain_left > gain_right;
    let (max_index, gain) = if is_left_greater {
        (max_index_none_left, gain_left)
    } else {
        (max_index_none_right, gain_right)
    };

    if max_index.is_none() || gain.is_infinite() || gain.is_nan() {
        return SplitResult::InvalidSplit(format!(
            "Max index is None. No valid split found. Last invalid reason: {:?}",
            last_invalid_reason
        ));
    }

    let split_value = candidate_split_points
        .get(max_index.unwrap())
        .copied()
        .unwrap();

    // If gains are identic, then none can go either side
    let none_to_left = if gain_left != gain_right {
        Some(is_left_greater)
    } else {
        None
    };

    SplitResult::ContinuousSplit(split_value, gain, none_to_left)
}

fn find_best_categorical_split_series(
    node: &Node,
    series: &Series,
    restrictions: &SplitRestrictions,
    sample_weights: &Float64Chunked,
) -> SplitResult {
    let is_target_column = series.name().starts_with(TARGET_PREFIX);

    // For target columns, we use the original binary split logic
    // since the algorithm is designed for X feature splits (keeping Y volume unchanged)

    // Get category counts for x and xy indices
    let res_xy = match series.count_categories(&node.indices_xy, sample_weights) {
        Ok(v) => v,
        Err(_) => return SplitResult::InvalidSplit("Error counting categories for xy".into()),
    };
    let res_x = match series.count_categories(&node.indices_x, sample_weights) {
        Ok(v) => v,
        Err(_) => return SplitResult::InvalidSplit("Error counting categories for x".into()),
    };
    let res_y = match series.count_categories(&node.indices_y, sample_weights) {
        Ok(v) => v,
        Err(_) => return SplitResult::InvalidSplit("Error counting categories for y".into()),
    };

    let CategoryCountResult {
        counts: count_xy_raw,
        unique_categories: unique_xy,
        none_count: _,
        none_weight: none_weight_xy,
    } = res_xy;

    let CategoryCountResult {
        counts: count_x_raw,
        unique_categories: unique_x,
        none_count: _,
        none_weight: none_weight_x,
    } = res_x;

    let CategoryCountResult {
        counts: count_y_raw,
        unique_categories: unique_y,
        none_count: _,
        none_weight: none_weight_y,
    } = res_y;

    let xy_weights_map: HashMap<u32, f64> = unique_xy
        .iter()
        .cloned()
        .zip(count_xy_raw.iter().cloned())
        .collect();

    let x_weights_map: HashMap<u32, f64> = unique_x
        .iter()
        .cloned()
        .zip(count_x_raw.iter().cloned())
        .collect();

    let y_weights_map: HashMap<u32, f64> = unique_y
        .iter()
        .cloned()
        .zip(count_y_raw.iter().cloned())
        .collect();

    // Build count maps: a[c] for xy counts, b[c] for x counts
    let a_map: HashMap<u32, f64> = unique_xy
        .iter()
        .cloned()
        .zip(count_xy_raw.iter().cloned())
        .collect();

    let b_map: HashMap<u32, f64> = if is_target_column {
        unique_x
            .iter()
            .cloned()
            .zip(count_x_raw.iter().cloned().map(|_| 1.0))
            .collect()
    } else {
        unique_x
            .iter()
            .cloned()
            .zip(count_x_raw.iter().cloned())
            .collect()
    };
    // Get categories with b[c] > 0 (categories present in x indices)

    let unique_reference = if is_target_column {
        &unique_y
    } else {
        &unique_x
    };
    let categories: Vec<u32> = unique_reference
        .iter()
        .filter(|&&c| *b_map.get(&c).unwrap_or(&0.0) > 0.0)
        .cloned()
        .collect();

    if categories.len() <= 1 {
        return SplitResult::InvalidSplit("Less than 2 categories with positive x count".into());
    }

    // Compute totals
    let total_xy: f64 = count_xy_raw.iter().sum();
    let total_x: f64 = count_x_raw.iter().sum();
    let total_y: f64 = count_y_raw.iter().sum();

    if total_x == 0.0 {
        return SplitResult::InvalidSplit("Total x count is zero".into());
    }

    // Compute ordering scores p[c] = a[c] / b[c] and sort
    let mut scored_categories: Vec<(u32, f64)> = categories
        .iter()
        .map(|&c| {
            let a_c = *a_map.get(&c).unwrap_or(&0.0);
            let b_c = *b_map.get(&c).unwrap_or(&0.0);
            let p_c = if b_c > 0.0 { a_c / b_c } else { 0.0 };
            (c, p_c)
        })
        .collect();

    // Sort by p[c] ascending
    scored_categories.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_categories: Vec<u32> = scored_categories.iter().map(|(c, _)| *c).collect();
    let m = sorted_categories.len();

    // Target volume (Y volume stays unchanged for X splits)
    let target_volume = node.cell.target_volume();

    // Sort count_xy and count_x according to sorted_categories
    // to prepare for prefix scan

    // Prefix scan to find best split using the ordering from p[c]
    // but computing the standard logistic gain for each candidate
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_m: Option<usize> = None;
    let mut best_none_to_left: Option<bool> = None;
    let mut last_invalid_reason: Option<String> = None;

    // Accumulate prefix sums for xy and x counts
    let mut xy_l: f64 = 0.0_f64;
    let mut x_l: f64;
    let mut x_r: f64;
    let mut y_l: f64;
    let mut y_r: f64;

    let mut target_volume_l: f64;
    let mut target_volume_r: f64;

    if is_target_column {
        x_l = total_x;
        x_r = total_x;
        target_volume_l = 0.0_f64;
        target_volume_r = target_volume.clone();
        y_l = 0.0_f64;
        y_r = total_y;
    } else {
        x_l = 0.0_f64;
        x_r = total_x;
        target_volume_l = target_volume.clone();
        target_volume_r = target_volume.clone();
        y_l = total_y;
        y_r = total_y;
    }

    for split_m in 1..m {
        let c = sorted_categories[split_m - 1];
        xy_l += xy_weights_map.get(&c).copied().unwrap_or(0.0);
        let xy_r = total_xy - xy_l;

        if is_target_column {
            target_volume_l += 1.0;
            target_volume_r -= 1.0;
            y_l += y_weights_map.get(&c).copied().unwrap_or(0.0);
            y_r = total_y - y_l;
        } else {
            // For X feature splits, y counts and volumes don't change between children
            // y_l = y_r = total_y (all y values are in both children's domain)

            x_l += x_weights_map.get(&c).copied().unwrap_or(0.0);
            x_r = total_x - x_l;
        };

        // Evaluate with none to left
        let gain_none_left = logistic_gain(
            &(xy_l + none_weight_xy),
            &(x_l + none_weight_x),
            &(y_l + none_weight_y),
            &xy_r,
            &x_r,
            &y_r,
            &target_volume,
            &target_volume_l,
            &target_volume_r,
            &restrictions.dataset_size,
        );
        let (is_valid_none_left, reason_left) = restrictions.is_point_valid(
            &(xy_l + none_weight_xy),
            &(x_l + none_weight_x),
            &(y_l + none_weight_y),
            &xy_r,
            &x_r,
            &y_r,
            &target_volume,
            &target_volume_l,
            &target_volume_r,
            &gain_none_left,
        );

        // Evaluate None to right

        let gain_none_right = logistic_gain(
            &xy_l,
            &x_l,
            &y_l,
            &(xy_r + none_weight_xy),
            &(x_r + none_weight_x),
            &(y_r + none_weight_y),
            &target_volume,
            &target_volume_l,
            &target_volume_r,
            &restrictions.dataset_size,
        );
        let (is_valid_none_right, reason_right) = restrictions.is_point_valid(
            &xy_l,
            &x_l,
            &y_l,
            &(xy_r + none_weight_xy),
            &(x_r + none_weight_x),
            &(y_r + none_weight_y),
            &target_volume,
            &target_volume_l,
            &target_volume_r,
            &gain_none_right,
        );

        if !is_valid_none_left && !is_valid_none_right {
            last_invalid_reason = Some(reason_left);
            continue;
        }

        let (gain, none_to_left) = if !is_valid_none_left {
            (gain_none_right, Some(false))
        } else if !is_valid_none_right {
            (gain_none_left, Some(true))
        } else if gain_none_left > gain_none_right {
            (gain_none_left, Some(true))
        } else if gain_none_right > gain_none_left {
            (gain_none_right, Some(false))
        } else {
            (gain_none_left, None)
        };

        if gain > best_gain {
            best_gain = gain;
            best_m = Some(split_m);
            best_none_to_left = none_to_left;
        }
    }

    if best_m.is_none() || best_gain.is_infinite() || best_gain.is_nan() {
        return SplitResult::InvalidSplit(format!(
            "No valid subset split found. Last reason: {:?}",
            last_invalid_reason
        ));
    }

    // Build the subset S* = {c_(1), ..., c_(m*)}
    let best_split_m = best_m.unwrap();
    let mut subset: Vec<u32> = sorted_categories[..best_split_m].to_vec();
    subset.sort(); // Sort for deterministic output

    SplitResult::CategoricalSplit(subset, best_gain, best_none_to_left)
}

fn add_domain_to_edges_if_finite(
    unique_values: &mut Vec<f64>,
    domain: (f64, f64),
) -> &mut Vec<f64> {
    if domain.0.is_finite() {
        unique_values.insert(0, domain.0);
    }
    if domain.1.is_finite() {
        unique_values.push(domain.1);
    }
    unique_values
}
