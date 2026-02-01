use crate::conf::TARGET_PREFIX;
use itertools::Itertools;
use polars::datatypes::*;
use polars::error::ErrString;
use polars::prelude::SortOptions;
use polars::prelude::*;
use std::collections::HashMap;

// Reusable ordered weighted-count utility
fn ordered_weighted_counts<I>(iter: I, weights: Float64Chunked) -> (Vec<f64>, Vec<u32>)
where
    I: Iterator<Item = Option<u32>>,
{
    let mut pos_map: HashMap<u32, usize> = HashMap::new();
    let mut counts: Vec<f64> = Vec::new();
    let mut unique: Vec<u32> = Vec::new();

    for (opt_code, opt_weight) in iter.zip(weights.into_iter()) {
        let weight = opt_weight.unwrap_or(0.0);
        if weight == 0.0 {
            continue;
        }
        if let Some(code) = opt_code {
            if let Some(&pos) = pos_map.get(&code) {
                counts[pos] += weight;
            } else {
                let pos = counts.len();
                pos_map.insert(code, pos);
                counts.push(weight);
                unique.push(code);
            }
        }
    }
    (counts, unique)
}

// Per-width helpers
trait PTreeUIntCodesExt {
    fn weighted_counts(&self, weights: &Float64Chunked) -> (Vec<f64>, Vec<u32>);
}
impl PTreeUIntCodesExt for UInt8Chunked {
    fn weighted_counts(&self, weights: &Float64Chunked) -> (Vec<f64>, Vec<u32>) {
        ordered_weighted_counts(
            self.into_iter().map(|o| o.map(|v| v as u32)),
            weights.clone(),
        )
    }
}
impl PTreeUIntCodesExt for UInt16Chunked {
    fn weighted_counts(&self, weights: &Float64Chunked) -> (Vec<f64>, Vec<u32>) {
        ordered_weighted_counts(
            self.into_iter().map(|o| o.map(|v| v as u32)),
            weights.clone(),
        )
    }
}
impl PTreeUIntCodesExt for UInt32Chunked {
    fn weighted_counts(&self, weights: &Float64Chunked) -> (Vec<f64>, Vec<u32>) {
        ordered_weighted_counts(self.into_iter(), weights.clone())
    }
}

/// Result of sorting numeric values with None handling
#[derive(Debug, Clone, PartialEq)]
pub struct NumericSortResult<T> {
    pub sorted_values: Vec<T>,
    pub unique_values: Vec<T>,
    pub sorted_weights: Vec<f64>,
    pub none_count: usize,
    pub none_weight: f64,
}

/// Result of counting categories with None handling
#[derive(Debug, Clone)]
pub struct CategoryCountResult {
    pub counts: Vec<f64>,
    pub unique_categories: Vec<u32>,
    pub none_count: usize,
    pub none_weight: f64,
}

/// Extension methods for Series
pub trait PTreeSeriesExt {
    /// For numeric series: given indices, return sorted values with None count
    /// Optionally return the unique values (in sorted order)
    fn sorted_numeric_values<T: PolarsNumericType>(
        &self,
        idx: &UInt32Chunked,
        return_unique: bool,
        sample_weights: &Float64Chunked,
    ) -> PolarsResult<NumericSortResult<T::Native>>;

    fn count_categories(
        &self,
        idx: &UInt32Chunked,
        sample_weights: &Float64Chunked,
    ) -> PolarsResult<CategoryCountResult>;
}

impl PTreeSeriesExt for Series {
    fn sorted_numeric_values<T: PolarsNumericType>(
        &self,
        idx: &UInt32Chunked,
        return_unique: bool,
        sample_weights: &Float64Chunked,
    ) -> PolarsResult<NumericSortResult<T::Native>> {
        let is_numeric = matches!(
            self.dtype(),
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
        );
        if !is_numeric {
            panic!("sorted_numeric_values can only be called on numeric series");
        }

        // Collect Option<f64> for requested indices
        let values = self.take(idx)?;

        let sorted_idx = values.arg_sort(
            SortOptions::default()
                .with_order_descending(false)
                .with_nulls_last(false)
                .with_multithreaded(false),
        );

        // Select the weights for the selected indices
        let sorted_weights = sample_weights.take(idx)?.take(&sorted_idx)?;

        // Split at None boundary
        let none_count = values.null_count() as usize;

        let sorted_values_with_nulls = values.take(&sorted_idx)?;
        let null_mask = sorted_values_with_nulls.is_null();
        let none_weight = null_mask.into_iter().zip(sorted_weights.into_iter()).fold(
            0.0,
            |acc, (is_null, weight)| {
                if matches!(is_null, Some(true)) {
                    acc + weight.unwrap_or(0.0)
                } else {
                    acc
                }
            },
        );

        // Single pass: build sorted_values and (optionally) unique_values
        let taken = sorted_values_with_nulls.drop_nulls();
        let sorted_values: Vec<T::Native> = taken
            .unpack::<T>()
            .expect("unpack failed")
            .into_no_null_iter()
            .collect();

        let unique_values = sorted_values.iter().copied().dedup().collect();

        let sorted_weights: Vec<f64> = sorted_weights.into_no_null_iter().collect();

        if !return_unique {
            return PolarsResult::Ok(NumericSortResult {
                sorted_values,
                unique_values,
                sorted_weights,
                none_count,
                none_weight,
            });
        } else {
            PolarsResult::Ok(NumericSortResult {
                sorted_values,
                unique_values,
                sorted_weights,
                none_count,
                none_weight,
            })
        }
    }

    fn count_categories(
        &self,
        idx: &UInt32Chunked,
        sample_weights: &Float64Chunked,
    ) -> PolarsResult<CategoryCountResult> {
        // Only Enum/Categorical supported
        if !matches!(
            self.dtype(),
            DataType::Enum(_, _) | DataType::Categorical(_, _)
        ) {
            return Err(PolarsError::AssertionError(ErrString::from(
                "count_categories can only be called on Enum (categorical) series",
            )));
        }

        // Select rows first so null_count matches the slice we count over
        let taken = self.take(idx)?;
        let weights = sample_weights.take(idx)?;
        let none_count = taken.null_count() as usize;
        let null_mask = taken.is_null();
        let none_weight = null_mask.into_iter().zip(weights.clone().into_iter()).fold(
            0.0,
            |acc, (is_null, weight)| {
                if matches!(is_null, Some(true)) {
                    acc + weight.unwrap_or(0.0)
                } else {
                    acc
                }
            },
        );

        // Work on physical codes (u8/u16/u32), preserving first-seen order
        let phys = taken.to_physical_repr();
        let (counts, unique_categories) = match phys.dtype() {
            DataType::UInt8 => phys
                .u8()
                .expect("categorical phys u8")
                .weighted_counts(&weights),
            DataType::UInt16 => phys
                .u16()
                .expect("categorical phys u16")
                .weighted_counts(&weights),
            DataType::UInt32 => phys
                .u32()
                .expect("categorical phys u32")
                .weighted_counts(&weights),
            other => {
                return Err(PolarsError::ComputeError(ErrString::from(format!(
                    "unexpected categorical physical dtype: {other:?}"
                ))));
            }
        };

        Ok(CategoryCountResult {
            counts,
            unique_categories,
            none_count,
            none_weight,
        })
    }
}

/// Extension methods for DataFrame
pub trait PTreeDataFrameExt {
    /// All column names as owned Strings
    fn column_names_vec(&self) -> Vec<String>;

    /// All target column names (by prefix)
    fn target_column_names(&self) -> Vec<String>;

    /// For numeric column by name: given indices, return sorted values with None count
    fn sorted_numeric_values<T: PolarsNumericType>(
        &self,
        name: &str,
        idx: &UInt32Chunked,
        return_unique: bool,
        sample_weights: &Float64Chunked,
    ) -> PolarsResult<NumericSortResult<T::Native>>;

    /// For string-like column by name: given indices, return counts per category with None count
    fn count_categories(
        &self,
        name: &str,
        idx: &UInt32Chunked,
        sample_weights: &Float64Chunked,
    ) -> PolarsResult<CategoryCountResult>;

    /// Get a specific row by index, returning a HashMap of column name -> stringified value
    fn get_row_as_strings(&self, row_idx: usize) -> Option<HashMap<String, String>>;
}

impl PTreeDataFrameExt for DataFrame {
    fn column_names_vec(&self) -> Vec<String> {
        self.get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn target_column_names(&self) -> Vec<String> {
        self.get_column_names()
            .iter()
            .filter(|name| name.starts_with(TARGET_PREFIX))
            .map(|s| s.to_string())
            .collect()
    }

    fn sorted_numeric_values<T: PolarsNumericType>(
        &self,
        name: &str,
        idx: &UInt32Chunked,
        return_unique: bool,
        sample_weights: &Float64Chunked,
    ) -> PolarsResult<NumericSortResult<T::Native>> {
        let s = self
            .column(name)
            .unwrap_or_else(|_| panic!("Column '{}' not found", name));
        let series = s.as_series().unwrap();

        PTreeSeriesExt::sorted_numeric_values::<T>(series, idx, return_unique, sample_weights)
    }

    fn count_categories(
        &self,
        name: &str,
        idx: &UInt32Chunked,
        sample_weights: &Float64Chunked,
    ) -> PolarsResult<CategoryCountResult> {
        let series = self.column(name)?.as_series().unwrap();
        series.count_categories(idx, sample_weights)
    }

    fn get_row_as_strings(&self, row_idx: usize) -> Option<HashMap<String, String>> {
        if row_idx >= self.height() {
            return None;
        }
        let mut row = HashMap::new();
        for s in self.get_columns() {
            let name = s.name().to_string();
            let v = s.get(row_idx).unwrap_or(AnyValue::Null);
            let val = match v {
                AnyValue::Null => "None".to_string(),
                _ => v.to_string(),
            };
            row.insert(name, val);
        }
        Some(row)
    }
}
