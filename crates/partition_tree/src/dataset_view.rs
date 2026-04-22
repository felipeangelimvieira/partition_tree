//! Dataset abstraction for the v2 partition tree.
//!
//! The [`DatasetView`] and [`ColumnView`] traits decouple the tree builder
//! from any specific dataframe library. A Polars-backed implementation is
//! provided via [`PolarsDatasetView`] and [`PolarsColumnView`].
//!
//! ## Responsibilities
//!
//! A `DatasetView` provides:
//!
//! - Column access by name, with classification as feature or target.
//! - Global presorted indices for every column (ascending, nulls last).
//! - Per-sample weight vectors for the three measures ($\mu_{XY}$,
//!   $\mu_X$, $\mu_Y$).
//!
//! A `ColumnView` provides:
//!
//! - Typed random access (`get_f64`, `get_cat`) by row index.
//! - Null detection.
//! - Logical dtype (`Continuous` or `Categorical`).
//!
//! ## Implementing a custom backend
//!
//! To use a backend other than Polars, implement [`DatasetView`] and
//! [`ColumnView`] for your types. Both traits require `Send + Sync` so
//! that column-level split search can be parallelized with `rayon`.
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::conf::TARGET_PREFIX;
use crate::rule::DynValue;

// ---------------------------------------------------------------------------
// Logical dtype enum (shared across v2)
// ---------------------------------------------------------------------------

/// The logical data type of a column, as understood by the partition tree.
///
/// This enum drives dispatch to the appropriate [`DTypePlugin`](super::dtype_plugin::DTypePlugin)
/// and [`ColumnSplitSearcher`](super::column_split::ColumnSplitSearcher).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicalDTypeKind {
    Continuous,
    Categorical,
    Integer,
    QuantizedContinuous,
}

/// Configuration for quantized-continuous columns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuantizedContinuousSpec {
    resolution: f64,
}

impl QuantizedContinuousSpec {
    pub fn new(resolution: f64) -> Result<Self, String> {
        if !resolution.is_finite() || resolution <= 0.0 {
            return Err(format!(
                "quantized-continuous resolution must be finite and positive, got {resolution}"
            ));
        }

        Ok(Self { resolution })
    }

    pub fn resolution(self) -> f64 {
        self.resolution
    }

    pub fn quantize_value(self, value: f64) -> Result<i64, String> {
        if !value.is_finite() {
            return Err(format!("value {value} is not finite"));
        }

        let scaled = value / self.resolution;
        let rounded = scaled.round();
        let tolerance = scaled.abs().max(1.0) * 1e-9;

        if (scaled - rounded).abs() > tolerance {
            return Err(format!(
                "value {value} is not aligned to resolution {}",
                self.resolution
            ));
        }

        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            return Err(format!("value {value} is outside the quantized i64 range"));
        }

        Ok(rounded as i64)
    }
}

impl PartialEq for QuantizedContinuousSpec {
    fn eq(&self, other: &Self) -> bool {
        self.resolution.to_bits() == other.resolution.to_bits()
    }
}

impl Eq for QuantizedContinuousSpec {}

impl Hash for QuantizedContinuousSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.resolution.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicalDType {
    Continuous,
    Categorical,
    Integer,
    QuantizedContinuous(QuantizedContinuousSpec),
}

impl LogicalDType {
    pub fn kind(self) -> LogicalDTypeKind {
        match self {
            Self::Continuous => LogicalDTypeKind::Continuous,
            Self::Categorical => LogicalDTypeKind::Categorical,
            Self::Integer => LogicalDTypeKind::Integer,
            Self::QuantizedContinuous(_) => LogicalDTypeKind::QuantizedContinuous,
        }
    }

    pub fn quantized_continuous(resolution: f64) -> Result<Self, String> {
        Ok(Self::QuantizedContinuous(QuantizedContinuousSpec::new(
            resolution,
        )?))
    }
}

// ---------------------------------------------------------------------------
// ColumnView trait
// ---------------------------------------------------------------------------

/// Read-only, typed access to a single column of a dataset.
///
/// Implementations must be `Send + Sync` to allow parallel split search
/// across columns via `rayon`.
///
/// # Target detection
///
/// The default [`is_target`](ColumnView::is_target) implementation checks
/// whether the column name starts with the configured `TARGET_PREFIX`
/// (typically `"target__"`).
pub trait ColumnView: Send + Sync {
    /// Column name.
    fn name(&self) -> &str;

    /// Number of rows in the full dataset.
    fn len(&self) -> usize;

    /// Logical data type.
    fn logical_dtype(&self) -> LogicalDType;

    /// Get a continuous (f64) value at row `idx`. Returns `None` for nulls.
    fn get_f64(&self, idx: usize) -> Option<f64>;

    /// Get a categorical value at row `idx` as a usize code. Returns `None` for nulls.
    fn get_cat(&self, idx: usize) -> Option<usize>;

    /// Get an integer (i64) value at row `idx`. Returns `None` for nulls.
    fn get_i64(&self, idx: usize) -> Option<i64>;

    /// Whether the value at `idx` is null.
    fn is_null(&self, idx: usize) -> bool;

    /// Whether this is a target column (name starts with `target__`).
    fn is_target(&self) -> bool {
        self.name().starts_with(TARGET_PREFIX)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sorted string labels for categorical columns, mapping code → label.
    ///
    /// Returns `None` for non-categorical columns. The returned slice has
    /// length equal to the number of categories; entry `i` is the label for
    /// code `i`.
    fn cat_labels(&self) -> Option<&[String]> {
        None
    }

    /// Get the value at `idx` as a dtype-erased [`DynValue`].
    ///
    /// The default implementation dispatches on [`logical_dtype`](ColumnView::logical_dtype).
    /// Custom backends can override this for efficiency.
    fn get_dyn_value(&self, idx: usize) -> Option<DynValue> {
        match self.logical_dtype() {
            LogicalDType::Continuous | LogicalDType::QuantizedContinuous(_) => {
                self.get_f64(idx).map(DynValue::Continuous)
            }
            LogicalDType::Categorical => self.get_cat(idx).map(DynValue::Categorical),
            LogicalDType::Integer => self.get_i64(idx).map(DynValue::Integer),
        }
    }
}

// ---------------------------------------------------------------------------
// DatasetView trait
// ---------------------------------------------------------------------------

/// Read-only, tabular view of a dataset with presorted indices and weights.
///
/// This is the main entry point for the tree builder. A `DatasetView`:
///
/// 1. Provides [`ColumnView`] handles for feature and target columns.
/// 2. Stores globally presorted row indices per column (ascending, nulls last),
///    used by the root [`Node`](super::node::Node) to initialize its sorted
///    index maps.
/// 3. Stores per-sample weights for the three measures.
///
/// # Thread Safety
///
/// Must be `Send + Sync` since the tree builder shares the dataset across
/// rayon parallel iterators.
pub trait DatasetView: Send + Sync {
    /// Total number of rows.
    fn n_rows(&self) -> usize;

    /// Total number of columns (features + targets).
    fn n_columns(&self) -> usize;

    /// Access a column by name.
    fn column(&self, name: &str) -> Option<&dyn ColumnView>;

    /// All columns.
    fn columns(&self) -> Vec<&dyn ColumnView>;

    /// Feature columns only (not prefixed with `target__`).
    fn feature_columns(&self) -> Vec<&dyn ColumnView>;

    /// Target columns only (prefixed with `target__`).
    fn target_columns(&self) -> Vec<&dyn ColumnView>;

    /// All column names.
    fn column_names(&self) -> Vec<&str>;

    /// Global pre-sorted indices for a column (ascending, nulls last).
    fn sorted_indices(&self, col: &str) -> &[u32];

    /// Per-sample weights for the XY measure.
    fn weights_xy(&self) -> &[f64];

    /// Per-sample weights for the X measure.
    fn weights_x(&self) -> &[f64];

    /// Per-sample weights for the Y measure.
    fn weights_y(&self) -> &[f64];
}

// ---------------------------------------------------------------------------
// Polars ColumnView implementation
// ---------------------------------------------------------------------------

/// A [`ColumnView`] backed by a Polars `Series`.
///
/// On construction, the series is materialized into a `Vec<Option<T>>`
/// for O(1) random access. Supported Polars dtypes:
///
/// | Polars dtype         | Logical dtype       |
/// |----------------------|---------------------|
/// | `Float64`, `Float32` | `Continuous`        |
/// | `Int32`, `Int64`     | `Integer`           |
/// | `Enum`, `Categorical`| `Categorical`       |
/// | `Null`               | `Continuous`        |
#[derive(Debug)]
enum ColumnStorage {
    Continuous(Vec<Option<f64>>),
    Integer(Vec<Option<i64>>),
    QuantizedContinuous {
        values: Vec<Option<f64>>,
        spec: QuantizedContinuousSpec,
    },
    Categorical {
        codes: Vec<Option<usize>>,
        labels: Arc<Vec<String>>,
    },
}

impl ColumnStorage {
    fn inferred_from_series(series: &Series) -> Self {
        match series.dtype() {
            DataType::Float64 | DataType::Float32 | DataType::Null => {
                Self::try_continuous(series).expect("continuous materialization should succeed")
            }
            DataType::Int32 | DataType::Int64 => {
                Self::try_integer(series).expect("integer materialization should succeed")
            }
            DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                Self::try_categorical(series, None)
                    .expect("categorical materialization should succeed")
            }
            dt => panic!("PolarsColumnView: unsupported dtype {dt:?}"),
        }
    }

    fn try_from_series_with_dtype(
        series: &Series,
        logical_dtype: LogicalDType,
        labels: Option<&[String]>,
    ) -> Result<Self, String> {
        match logical_dtype {
            LogicalDType::Continuous => Self::try_continuous(series),
            LogicalDType::Integer => Self::try_integer(series),
            LogicalDType::Categorical => Self::try_categorical(series, labels),
            LogicalDType::QuantizedContinuous(spec) => Self::try_quantized_continuous(series, spec),
        }
    }

    fn try_numeric_as_f64(series: &Series) -> Result<Vec<Option<f64>>, String> {
        let len = series.len();

        match series.dtype() {
            DataType::Float64 => Ok(series.f64().expect("f64 series").into_iter().collect()),
            DataType::Float32 => Ok(series
                .f32()
                .expect("f32 series")
                .into_iter()
                .map(|v| v.map(|x| x as f64))
                .collect()),
            DataType::Int32 => Ok(series
                .i32()
                .expect("i32 series")
                .into_iter()
                .map(|v| v.map(|x| x as f64))
                .collect()),
            DataType::Int64 => Ok(series
                .i64()
                .expect("i64 series")
                .into_iter()
                .map(|v| v.map(|x| x as f64))
                .collect()),
            DataType::Null => Ok(vec![None; len]),
            dt => Err(format!("unsupported physical dtype {dt:?}")),
        }
    }

    fn try_continuous(series: &Series) -> Result<Self, String> {
        Ok(Self::Continuous(Self::try_numeric_as_f64(series)?))
    }

    fn try_quantized_continuous(
        series: &Series,
        spec: QuantizedContinuousSpec,
    ) -> Result<Self, String> {
        let values = Self::try_numeric_as_f64(series)?;

        for value in values.iter().flatten() {
            spec.quantize_value(*value)?;
        }

        Ok(Self::QuantizedContinuous { values, spec })
    }

    fn try_integer(series: &Series) -> Result<Self, String> {
        let len = series.len();
        let values = match series.dtype() {
            DataType::Int32 => series
                .i32()
                .expect("i32 series")
                .into_iter()
                .map(|v| v.map(|x| x as i64))
                .collect(),
            DataType::Int64 => series.i64().expect("i64 series").into_iter().collect(),
            DataType::Float64 => series
                .f64()
                .expect("f64 series")
                .into_iter()
                .map(|v| v.map(Self::checked_f64_to_i64).transpose())
                .collect::<Result<Vec<_>, _>>()?,
            DataType::Float32 => series
                .f32()
                .expect("f32 series")
                .into_iter()
                .map(|v| v.map(|x| Self::checked_f64_to_i64(x as f64)).transpose())
                .collect::<Result<Vec<_>, _>>()?,
            DataType::Null => vec![None; len],
            dt => return Err(format!("unsupported physical dtype {dt:?}")),
        };

        Ok(Self::Integer(values))
    }

    fn try_categorical(series: &Series, labels: Option<&[String]>) -> Result<Self, String> {
        let len = series.len();

        match series.dtype() {
            DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                let str_series = series
                    .cast(&DataType::String)
                    .map_err(|e| format!("failed to cast categorical column to String: {e}"))?;
                let str_ca = str_series
                    .str()
                    .map_err(|e| format!("failed to view categorical values as String: {e}"))?;
                let str_vals: Vec<Option<String>> = str_ca
                    .into_iter()
                    .map(|v| v.map(|s| s.to_owned()))
                    .collect();

                let label_values = match labels {
                    Some(existing) => existing.to_vec(),
                    None => {
                        let mut uniques: Vec<String> = str_vals
                            .iter()
                            .filter_map(|v| v.clone())
                            .collect::<std::collections::HashSet<_>>()
                            .into_iter()
                            .collect();
                        uniques.sort();
                        uniques
                    }
                };

                let str_to_code: HashMap<String, usize> = label_values
                    .iter()
                    .enumerate()
                    .map(|(code, label)| (label.clone(), code))
                    .collect();

                let codes = str_vals
                    .into_iter()
                    .map(|v| v.and_then(|s| str_to_code.get(&s).copied()))
                    .collect();

                Ok(Self::Categorical {
                    codes,
                    labels: Arc::new(label_values),
                })
            }
            DataType::Null => Ok(Self::Categorical {
                codes: vec![None; len],
                labels: Arc::new(labels.map_or_else(Vec::new, |existing| existing.to_vec())),
            }),
            dt => Err(format!("unsupported physical dtype {dt:?}")),
        }
    }

    fn checked_f64_to_i64(value: f64) -> Result<i64, String> {
        if !value.is_finite() {
            return Err(format!("value {value} is not finite"));
        }

        let rounded = value.round();
        if (value - rounded).abs() > f64::EPSILON {
            return Err(format!("value {value} is not an exact integer"));
        }

        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            return Err(format!("value {value} is outside the i64 range"));
        }

        Ok(rounded as i64)
    }

    fn len(&self) -> usize {
        match self {
            Self::Continuous(values) => values.len(),
            Self::Integer(values) => values.len(),
            Self::QuantizedContinuous { values, .. } => values.len(),
            Self::Categorical { codes, .. } => codes.len(),
        }
    }

    fn logical_dtype(&self) -> LogicalDType {
        match self {
            Self::Continuous(_) => LogicalDType::Continuous,
            Self::Integer(_) => LogicalDType::Integer,
            Self::QuantizedContinuous { spec, .. } => LogicalDType::QuantizedContinuous(*spec),
            Self::Categorical { .. } => LogicalDType::Categorical,
        }
    }

    fn get_f64(&self, idx: usize) -> Option<f64> {
        match self {
            Self::Continuous(values) => values.get(idx).copied().flatten(),
            Self::QuantizedContinuous { values, .. } => values.get(idx).copied().flatten(),
            Self::Integer(_) | Self::Categorical { .. } => None,
        }
    }

    fn get_cat(&self, idx: usize) -> Option<usize> {
        match self {
            Self::Categorical { codes, .. } => codes.get(idx).copied().flatten(),
            Self::Continuous(_) | Self::Integer(_) | Self::QuantizedContinuous { .. } => None,
        }
    }

    fn get_i64(&self, idx: usize) -> Option<i64> {
        match self {
            Self::Integer(values) => values.get(idx).copied().flatten(),
            Self::Continuous(_) | Self::QuantizedContinuous { .. } | Self::Categorical { .. } => {
                None
            }
        }
    }

    fn is_null(&self, idx: usize) -> bool {
        match self {
            Self::Continuous(values) => values.get(idx).is_none_or(|value| value.is_none()),
            Self::Integer(values) => values.get(idx).is_none_or(|value| value.is_none()),
            Self::QuantizedContinuous { values, .. } => {
                values.get(idx).is_none_or(|value| value.is_none())
            }
            Self::Categorical { codes, .. } => codes.get(idx).is_none_or(|value| value.is_none()),
        }
    }

    fn cat_label_arc(&self) -> Option<&Arc<Vec<String>>> {
        match self {
            Self::Categorical { labels, .. } => Some(labels),
            Self::Continuous(_) | Self::Integer(_) | Self::QuantizedContinuous { .. } => None,
        }
    }
}

pub struct PolarsColumnView {
    name: String,
    storage: ColumnStorage,
}

impl PolarsColumnView {
    /// Build a `PolarsColumnView` from a Polars `Series`.
    ///
    /// # Panics
    ///
    /// Panics if the series has an unsupported dtype (e.g., `Utf8`, `Boolean`).
    pub fn from_series(series: &Series) -> Self {
        Self {
            name: series.name().to_string(),
            storage: ColumnStorage::inferred_from_series(series),
        }
    }

    /// Build a `PolarsColumnView` from a Polars `Series`, forcing the
    /// logical dtype used by the tree.
    pub fn try_from_series_with_dtype(
        series: &Series,
        logical_dtype: LogicalDType,
    ) -> Result<Self, String> {
        Self::try_from_series_with_dtype_and_labels(series, logical_dtype, None)
    }

    /// Build a `PolarsColumnView` from a Polars `Series`, forcing the
    /// logical dtype and optionally stabilizing categorical labels.
    pub fn try_from_series_with_dtype_and_labels(
        series: &Series,
        logical_dtype: LogicalDType,
        labels: Option<&[String]>,
    ) -> Result<Self, String> {
        let name = series.name().to_string();
        let make_err = |details: String| {
            format!(
                "Column '{}' cannot be materialized as {:?}: {}",
                name, logical_dtype, details
            )
        };

        let storage = ColumnStorage::try_from_series_with_dtype(series, logical_dtype, labels)
            .map_err(make_err)?;

        Ok(Self { name, storage })
    }

    /// Build a `PolarsColumnView` from a Polars `Series`, using a
    /// pre-existing label→code mapping for categorical columns.
    ///
    /// This ensures that the same string label always maps to the same
    /// integer code, regardless of which categories appear in this
    /// particular series. Strings not present in `labels` are treated
    /// as null (unknown category).
    ///
    /// For non-categorical series, this falls back to [`from_series`].
    pub fn from_series_with_labels(series: &Series, labels: &[String]) -> Self {
        match series.dtype() {
            DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                Self::try_from_series_with_dtype_and_labels(
                    series,
                    LogicalDType::Categorical,
                    Some(labels),
                )
                .expect("categorical materialization with provided labels should succeed")
            }
            _ => Self::from_series(series),
        }
    }

    /// The sorted label list for categorical columns (code → label).
    pub fn cat_label_arc(&self) -> Option<&Arc<Vec<String>>> {
        self.storage.cat_label_arc()
    }
}

impl ColumnView for PolarsColumnView {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn logical_dtype(&self) -> LogicalDType {
        self.storage.logical_dtype()
    }

    fn get_f64(&self, idx: usize) -> Option<f64> {
        self.storage.get_f64(idx)
    }

    fn get_cat(&self, idx: usize) -> Option<usize> {
        self.storage.get_cat(idx)
    }

    fn get_i64(&self, idx: usize) -> Option<i64> {
        self.storage.get_i64(idx)
    }

    fn is_null(&self, idx: usize) -> bool {
        self.storage.is_null(idx)
    }

    fn cat_labels(&self) -> Option<&[String]> {
        self.storage.cat_label_arc().map(|labels| labels.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Polars DatasetView implementation
// ---------------------------------------------------------------------------

/// A [`DatasetView`] backed by a Polars `DataFrame`.
///
/// On construction it:
///
/// 1. Materializes each column into a [`PolarsColumnView`] (no repeated Polars access).
/// 2. Presorts row indices for every column (ascending, nulls last).
/// 3. Creates uniform `1.0` weights (or user-supplied weights).
///
/// # Examples
///
/// ```rust,ignore
/// let df: DataFrame = /* … */;
/// let view = PolarsDatasetView::new(&df);
/// let tree = TreeBuilder::new(config, loss, registry).build(&view);
/// ```
pub struct PolarsDatasetView {
    columns: Vec<PolarsColumnView>,
    col_index: HashMap<String, usize>,
    sorted_indices: HashMap<String, Vec<u32>>,
    weights_xy: Vec<f64>,
    weights_x: Vec<f64>,
    weights_y: Vec<f64>,
    n_rows: usize,
}

impl PolarsDatasetView {
    /// Build from a Polars `DataFrame` with uniform weights of `1.0`.
    pub fn new(df: &DataFrame) -> Self {
        Self::with_weights(df, None, None, None)
    }

    /// Build from a Polars `DataFrame` with optional per-sample weights.
    ///
    /// If a weight vector is `None`, uniform weights of `1.0` are used.
    /// Each vector must have length equal to `df.height()`.
    pub fn with_weights(
        df: &DataFrame,
        weights_xy: Option<Vec<f64>>,
        weights_x: Option<Vec<f64>>,
        weights_y: Option<Vec<f64>>,
    ) -> Self {
        Self::build_inner(df, weights_xy, weights_x, weights_y, None)
    }

    /// Build from a Polars `DataFrame` using pre-existing category label
    /// mappings so that categorical codes are consistent with a previously
    /// fitted model.
    ///
    /// `cat_labels` maps column name → sorted label list (code → label).
    /// Columns present in the map use [`PolarsColumnView::from_series_with_labels`];
    /// all others use the default [`PolarsColumnView::from_series`].
    pub fn with_category_labels(df: &DataFrame, cat_labels: &HashMap<String, Vec<String>>) -> Self {
        Self::build_inner(df, None, None, None, Some(cat_labels))
    }

    /// Build from a Polars `DataFrame` while forcing logical dtype overrides
    /// for specific named columns.
    pub fn try_with_dtype_overrides(
        df: &DataFrame,
        dtype_overrides: &HashMap<String, LogicalDType>,
    ) -> Result<Self, String> {
        Self::try_build_inner(df, None, None, None, None, Some(dtype_overrides))
    }

    /// Build from a Polars `DataFrame` using pre-existing category labels and
    /// per-column logical dtype overrides.
    pub fn try_with_category_labels_and_dtype_overrides(
        df: &DataFrame,
        cat_labels: &HashMap<String, Vec<String>>,
        dtype_overrides: &HashMap<String, LogicalDType>,
    ) -> Result<Self, String> {
        Self::try_build_inner(
            df,
            None,
            None,
            None,
            Some(cat_labels),
            Some(dtype_overrides),
        )
    }

    /// Extract the category label mappings from all categorical columns.
    ///
    /// Returns a map of column name → sorted labels (code → label).
    /// Only categorical columns are included.
    pub fn category_labels(&self) -> HashMap<String, Vec<String>> {
        self.columns
            .iter()
            .filter_map(|c| {
                c.cat_label_arc()
                    .map(|labels| (c.name().to_string(), labels.as_ref().clone()))
            })
            .collect()
    }

    /// Shared builder used by all constructors.
    fn build_inner(
        df: &DataFrame,
        weights_xy: Option<Vec<f64>>,
        weights_x: Option<Vec<f64>>,
        weights_y: Option<Vec<f64>>,
        cat_labels: Option<&HashMap<String, Vec<String>>>,
    ) -> Self {
        Self::try_build_inner(df, weights_xy, weights_x, weights_y, cat_labels, None)
            .unwrap_or_else(|e| panic!("PolarsDatasetView: {e}"))
    }

    fn try_build_inner(
        df: &DataFrame,
        weights_xy: Option<Vec<f64>>,
        weights_x: Option<Vec<f64>>,
        weights_y: Option<Vec<f64>>,
        cat_labels: Option<&HashMap<String, Vec<String>>>,
        dtype_overrides: Option<&HashMap<String, LogicalDType>>,
    ) -> Result<Self, String> {
        if let Some(overrides) = dtype_overrides {
            let mut missing: Vec<String> = overrides
                .keys()
                .filter(|name| df.column(name.as_str()).is_err())
                .cloned()
                .collect();
            if !missing.is_empty() {
                missing.sort();
                return Err(format!(
                    "dtype overrides reference unknown columns: {}",
                    missing.join(", ")
                ));
            }
        }

        let n_rows = df.height();

        let columns: Vec<PolarsColumnView> = df
            .get_columns()
            .iter()
            .map(|c| {
                let series = match c.as_series() {
                    Some(s) => s.clone(),
                    None => Series::new_null(c.name().clone(), c.len()),
                };
                let col_name = c.name().to_string();
                let labels = cat_labels.and_then(|map| map.get(&col_name).map(Vec::as_slice));

                match dtype_overrides.and_then(|map| map.get(&col_name)).copied() {
                    Some(dtype) => PolarsColumnView::try_from_series_with_dtype_and_labels(
                        &series, dtype, labels,
                    ),
                    None => match labels {
                        Some(existing) => {
                            Ok(PolarsColumnView::from_series_with_labels(&series, existing))
                        }
                        None => Ok(PolarsColumnView::from_series(&series)),
                    },
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let col_index: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name().to_string(), i))
            .collect();

        let sorted_indices: HashMap<String, Vec<u32>> = columns
            .iter()
            .map(|col| {
                let sorted = Self::sort_column_indices(col, n_rows);
                (col.name().to_string(), sorted)
            })
            .collect();

        let w_xy = weights_xy.unwrap_or_else(|| vec![1.0; n_rows]);
        let w_x = weights_x.unwrap_or_else(|| vec![1.0; n_rows]);
        let w_y = weights_y.unwrap_or_else(|| vec![1.0; n_rows]);

        Ok(Self {
            columns,
            col_index,
            sorted_indices,
            weights_xy: w_xy,
            weights_x: w_x,
            weights_y: w_y,
            n_rows,
        })
    }

    /// Sort indices by column values (ascending, nulls last).
    fn sort_column_indices(col: &PolarsColumnView, n_rows: usize) -> Vec<u32> {
        let mut indices: Vec<u32> = (0..n_rows as u32).collect();

        match col.logical_dtype() {
            LogicalDType::Continuous | LogicalDType::QuantizedContinuous(_) => {
                indices.sort_by(|&a, &b| {
                    let va = col.get_f64(a as usize);
                    let vb = col.get_f64(b as usize);
                    match (va, vb) {
                        (Some(x), Some(y)) => {
                            x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                });
            }
            LogicalDType::Categorical => {
                indices.sort_by(|&a, &b| {
                    let va = col.get_cat(a as usize);
                    let vb = col.get_cat(b as usize);
                    match (va, vb) {
                        (Some(x), Some(y)) => x.cmp(&y),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                });
            }
            LogicalDType::Integer => {
                indices.sort_by(|&a, &b| {
                    let va = col.get_i64(a as usize);
                    let vb = col.get_i64(b as usize);
                    match (va, vb) {
                        (Some(x), Some(y)) => x.cmp(&y),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                });
            }
        }

        indices
    }
}

impl DatasetView for PolarsDatasetView {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_columns(&self) -> usize {
        self.columns.len()
    }

    fn column(&self, name: &str) -> Option<&dyn ColumnView> {
        self.col_index
            .get(name)
            .map(|&i| &self.columns[i] as &dyn ColumnView)
    }

    fn columns(&self) -> Vec<&dyn ColumnView> {
        self.columns.iter().map(|c| c as &dyn ColumnView).collect()
    }

    fn feature_columns(&self) -> Vec<&dyn ColumnView> {
        self.columns
            .iter()
            .filter(|c| !c.is_target())
            .map(|c| c as &dyn ColumnView)
            .collect()
    }

    fn target_columns(&self) -> Vec<&dyn ColumnView> {
        self.columns
            .iter()
            .filter(|c| c.is_target())
            .map(|c| c as &dyn ColumnView)
            .collect()
    }

    fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name()).collect()
    }

    fn sorted_indices(&self, col: &str) -> &[u32] {
        self.sorted_indices
            .get(col)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn weights_xy(&self) -> &[f64] {
        &self.weights_xy
    }

    fn weights_x(&self) -> &[f64] {
        &self.weights_x
    }

    fn weights_y(&self) -> &[f64] {
        &self.weights_y
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df() -> DataFrame {
        DataFrame::new(vec![
            Column::new("x1".into(), &[1.0_f64, 3.0, 2.0, 5.0, 4.0]),
            Column::new("target__y1".into(), &[10.0_f64, 30.0, 20.0, 50.0, 40.0]),
        ])
        .unwrap()
    }

    #[test]
    fn column_view_basic() {
        let df = make_df();
        let view = PolarsDatasetView::new(&df);
        let col = view.column("x1").unwrap();

        assert_eq!(col.name(), "x1");
        assert_eq!(col.len(), 5);
        assert_eq!(col.logical_dtype(), LogicalDType::Continuous);
        assert_eq!(col.get_f64(0), Some(1.0));
        assert_eq!(col.get_f64(3), Some(5.0));
        assert!(!col.is_target());
    }

    #[test]
    fn target_column_detection() {
        let df = make_df();
        let view = PolarsDatasetView::new(&df);

        assert_eq!(view.feature_columns().len(), 1);
        assert_eq!(view.target_columns().len(), 1);
        assert!(view.target_columns()[0].is_target());
    }

    #[test]
    fn sorted_indices_ascending() {
        let df = make_df();
        let view = PolarsDatasetView::new(&df);
        let sorted = view.sorted_indices("x1");

        // Values: [1.0, 3.0, 2.0, 5.0, 4.0]
        // Sorted order: idx 0(1.0), 2(2.0), 1(3.0), 4(4.0), 3(5.0)
        assert_eq!(sorted, &[0, 2, 1, 4, 3]);
    }

    #[test]
    fn uniform_weights() {
        let df = make_df();
        let view = PolarsDatasetView::new(&df);

        assert_eq!(view.weights_xy().len(), 5);
        assert!(view.weights_xy().iter().all(|&w| (w - 1.0).abs() < 1e-10));
    }

    #[test]
    fn quantized_continuous_override_materializes_numeric_column() {
        let df = DataFrame::new(vec![Column::new(
            "x1".into(),
            &[0.0_f64, 0.5, 1.0, 1.5, 2.0],
        )])
        .unwrap();

        let mut overrides = HashMap::new();
        overrides.insert(
            "x1".to_string(),
            LogicalDType::quantized_continuous(0.5).unwrap(),
        );

        let view = PolarsDatasetView::try_with_dtype_overrides(&df, &overrides)
            .expect("quantized override should succeed");
        let col = view.column("x1").unwrap();

        assert_eq!(
            col.logical_dtype(),
            LogicalDType::quantized_continuous(0.5).unwrap()
        );
        assert_eq!(col.get_f64(1), Some(0.5));
    }

    #[test]
    fn quantized_continuous_override_rejects_off_lattice_values() {
        let df = DataFrame::new(vec![Column::new("x1".into(), &[0.0_f64, 0.3, 1.0])]).unwrap();

        let mut overrides = HashMap::new();
        overrides.insert(
            "x1".to_string(),
            LogicalDType::quantized_continuous(0.5).unwrap(),
        );

        let err = match PolarsDatasetView::try_with_dtype_overrides(&df, &overrides) {
            Ok(_) => panic!("off-lattice values should be rejected"),
            Err(err) => err,
        };
        assert!(err.contains("not aligned to resolution 0.5"));
    }

    // -----------------------------------------------------------------------
    // Categorical code stability tests
    // -----------------------------------------------------------------------

    /// Helper: create a categorical (Enum) series from string slices.
    fn make_cat_series(name: &str, values: &[&str], categories: &[&str]) -> Series {
        let str_series = Series::new(name.into(), values);
        let cats = polars::prelude::FrozenCategories::new(categories.iter().copied()).unwrap();
        str_series
            .cast(&DataType::from_frozen_categories(cats))
            .expect("cast to Enum")
    }

    /// Proves that `from_series` alone assigns different codes when category
    /// sets differ, but `from_series_with_labels` stabilises them by reusing
    /// the training-time label mapping.
    ///
    /// Training: ["apple", "banana", "cherry"] → lex codes: apple=0, banana=1, cherry=2
    /// Prediction raw: ["banana", "cherry", "date"] → lex codes: banana=0, cherry=1, date=2  (WRONG)
    /// Prediction fixed (reuse training labels): banana=1, cherry=2, date=None   (CORRECT)
    #[test]
    fn categorical_codes_stable_across_different_category_sets() {
        // -- Training dataset: 3 categories --
        let train_series = make_cat_series(
            "color",
            &["apple", "banana", "cherry", "banana"],
            &["apple", "banana", "cherry"],
        );
        let train_view = PolarsColumnView::from_series(&train_series);

        let banana_code_train = train_view.get_cat(1).unwrap(); // "banana"
        let cherry_code_train = train_view.get_cat(2).unwrap(); // "cherry"

        // -- Raw `from_series` on shifted categories gives WRONG codes --
        let pred_series = make_cat_series(
            "color",
            &["banana", "cherry", "date", "cherry"],
            &["banana", "cherry", "date"],
        );
        let raw_pred = PolarsColumnView::from_series(&pred_series);
        assert_ne!(
            banana_code_train,
            raw_pred.get_cat(0).unwrap(),
            "from_series should produce different codes (proves the underlying bug)"
        );

        // -- `from_series_with_labels` reuses training labels → CORRECT codes --
        let training_labels = train_view.cat_labels().unwrap().to_vec();
        let fixed_pred = PolarsColumnView::from_series_with_labels(&pred_series, &training_labels);

        let banana_code_fixed = fixed_pred.get_cat(0).unwrap();
        let cherry_code_fixed = fixed_pred.get_cat(1).unwrap();

        assert_eq!(
            banana_code_train, banana_code_fixed,
            "with_labels: 'banana' should get the same code as training ({banana_code_train}), got {banana_code_fixed}"
        );
        assert_eq!(
            cherry_code_train, cherry_code_fixed,
            "with_labels: 'cherry' should get the same code as training ({cherry_code_train}), got {cherry_code_fixed}"
        );

        // "date" is unknown to the training mapping → treated as null
        assert!(
            fixed_pred.get_cat(2).is_none(),
            "'date' was not in training labels, so it should map to None"
        );
    }

    /// End-to-end proof: a tree trained on {apple, banana, cherry} misroutes
    /// samples when predicting on {banana, cherry, date} because codes shift.
    ///
    /// Setup:
    ///   - Feature `color` perfectly predicts target `y`:
    ///     apple → y=10, banana → y=20, cherry → y=30
    ///   - Train the tree so it learns this mapping.
    ///   - Predict on data containing ["banana", "cherry"] but with "date"
    ///     present in the category set, shifting the lex codes.
    ///   - The prediction for "banana" should be ≈20, but the code shift
    ///     makes the tree see a different category.
    #[test]
    fn categorical_prediction_wrong_with_shifted_categories() {
        use crate::estimators::PartitionTree;
        use estimators::api::Estimator;

        let n = 60; // 20 each of apple, banana, cherry

        // -- Training data --
        let labels_train: Vec<&str> = (0..n)
            .map(|i| match i % 3 {
                0 => "apple",
                1 => "banana",
                _ => "cherry",
            })
            .collect();
        let y_train: Vec<f64> = (0..n)
            .map(|i| match i % 3 {
                0 => 10.0,
                1 => 20.0,
                _ => 30.0,
            })
            .collect();

        let x_train_series =
            make_cat_series("color", &labels_train, &["apple", "banana", "cherry"]);
        let x_train = DataFrame::new(vec![x_train_series.into()]).unwrap();
        let y_train = DataFrame::new(vec![Column::new("y".into(), y_train)]).unwrap();

        // Fit tree — enough leaves to learn the 3-way split
        let mut model = PartitionTree::new(
            10,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            10,
            2.0,
            None,
            true,
            None,
            None,
            None,
            std::collections::HashMap::new(),
        );
        let fitted = model
            .fit(&x_train, &y_train, None)
            .expect("fit should succeed");

        // -- Verify training predictions are correct --
        let train_preds = fitted.predict(&x_train).expect("predict on train");
        let train_col = train_preds.column("y").unwrap().f64().unwrap();
        // Apple rows (i%3==0) should predict ≈10, banana (i%3==1) ≈20, cherry ≈30
        for i in 0..n {
            let pred = train_col.get(i).unwrap();
            let expected = match i % 3 {
                0 => 10.0,
                1 => 20.0,
                _ => 30.0,
            };
            assert!(
                (pred - expected).abs() < 5.0,
                "Training row {i}: predicted {pred}, expected ≈{expected}"
            );
        }

        // -- Prediction data: same labels but with "date" added to category set --
        // This shifts lex codes: banana=0→0, cherry=1→1, date=2 (new)
        // Wait — with {banana, cherry, date}: banana=0, cherry=1, date=2
        // Training had {apple, banana, cherry}: apple=0, banana=1, cherry=2
        //
        // So "banana" was code 1 in training, but code 0 in prediction.
        // "cherry" was code 2 in training, but code 1 in prediction.
        let x_pred_series = make_cat_series(
            "color",
            &["banana", "cherry", "banana", "cherry"],
            &["banana", "cherry", "date"],
        );
        let x_pred = DataFrame::new(vec![x_pred_series.into()]).unwrap();

        let pred_result = fitted
            .predict(&x_pred)
            .expect("predict on shifted categories");
        let pred_col = pred_result.column("y").unwrap().f64().unwrap();

        // "banana" rows should predict ≈20, "cherry" rows should predict ≈30
        let banana_pred = pred_col.get(0).unwrap();
        let cherry_pred = pred_col.get(1).unwrap();

        // This assertion should FAIL because of code mismatch:
        // "banana" gets code 0 at prediction time, but code 0 was "apple" in training
        assert!(
            (banana_pred - 20.0).abs() < 5.0,
            "banana should predict ≈20.0, but got {banana_pred} \
             (likely matched 'apple' rule due to code shift)"
        );
        assert!(
            (cherry_pred - 30.0).abs() < 5.0,
            "cherry should predict ≈30.0, but got {cherry_pred} \
             (likely matched 'banana' rule due to code shift)"
        );
    }
}
