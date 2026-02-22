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
use std::sync::Arc;

use polars::prelude::*;

use super::rule::DynValue;
use crate::conf::TARGET_PREFIX;

// ---------------------------------------------------------------------------
// Logical dtype enum (shared across v2)
// ---------------------------------------------------------------------------

/// The logical data type of a column, as understood by the partition tree.
///
/// This enum drives dispatch to the appropriate [`DTypePlugin`](super::dtype_plugin::DTypePlugin)
/// and [`ColumnSplitSearcher`](super::column_split::ColumnSplitSearcher).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalDType {
    Continuous,
    Categorical,
    Integer,
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

    /// Get the value at `idx` as a dtype-erased [`DynValue`].
    ///
    /// The default implementation dispatches on [`logical_dtype`](ColumnView::logical_dtype).
    /// Custom backends can override this for efficiency.
    fn get_dyn_value(&self, idx: usize) -> Option<DynValue> {
        match self.logical_dtype() {
            LogicalDType::Continuous => self.get_f64(idx).map(DynValue::Continuous),
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
/// | `Int32`, `Int64`     | `Continuous`        |
/// | `Enum`, `Categorical`| `Categorical`       |
pub struct PolarsColumnView {
    name: String,
    logical_dtype: LogicalDType,
    /// For continuous columns: f64 values indexed by row.
    f64_values: Option<Vec<Option<f64>>>,
    /// For categorical columns: usize codes indexed by row.
    cat_values: Option<Vec<Option<usize>>>,
    /// For integer columns: i64 values indexed by row.
    i64_values: Option<Vec<Option<i64>>>,
    len: usize,
}

impl PolarsColumnView {
    /// Build a `PolarsColumnView` from a Polars `Series`.
    ///
    /// # Panics
    ///
    /// Panics if the series has an unsupported dtype (e.g., `Utf8`, `Boolean`).
    pub fn from_series(series: &Series) -> Self {
        let name = series.name().to_string();
        let len = series.len();

        let (logical_dtype, f64_values, cat_values, i64_values) = match series.dtype() {
            DataType::Float64 => {
                let ca = series.f64().expect("f64 series");
                let vals: Vec<Option<f64>> = ca.into_iter().collect();
                (LogicalDType::Continuous, Some(vals), None, None)
            }
            DataType::Float32 => {
                let ca = series.f32().expect("f32 series");
                let vals: Vec<Option<f64>> = ca.into_iter().map(|v| v.map(|x| x as f64)).collect();
                (LogicalDType::Continuous, Some(vals), None, None)
            }
            DataType::Int32 => {
                let ca = series.i32().expect("i32 series");
                let vals: Vec<Option<i64>> = ca.into_iter().map(|v| v.map(|x| x as i64)).collect();
                (LogicalDType::Integer, None, None, Some(vals))
            }
            DataType::Int64 => {
                let ca = series.i64().expect("i64 series");
                let vals: Vec<Option<i64>> = ca.into_iter().collect();
                (LogicalDType::Integer, None, None, Some(vals))
            }
            DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                let phys = series.to_physical_repr();
                let phys_u32 = phys.cast(&DataType::UInt32).expect("cast to u32");
                let ca = phys_u32.u32().expect("u32 chunked");
                let vals: Vec<Option<usize>> =
                    ca.into_iter().map(|v| v.map(|x| x as usize)).collect();
                (LogicalDType::Categorical, None, Some(vals), None)
            }
            dt => panic!("PolarsColumnView: unsupported dtype {dt:?}"),
        };

        Self {
            name,
            logical_dtype,
            f64_values,
            cat_values,
            i64_values,
            len,
        }
    }
}

impl ColumnView for PolarsColumnView {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        self.len
    }

    fn logical_dtype(&self) -> LogicalDType {
        self.logical_dtype
    }

    fn get_f64(&self, idx: usize) -> Option<f64> {
        self.f64_values
            .as_ref()
            .and_then(|v| v.get(idx).copied().flatten())
    }

    fn get_cat(&self, idx: usize) -> Option<usize> {
        self.cat_values
            .as_ref()
            .and_then(|v| v.get(idx).copied().flatten())
    }

    fn get_i64(&self, idx: usize) -> Option<i64> {
        self.i64_values
            .as_ref()
            .and_then(|v| v.get(idx).copied().flatten())
    }

    fn is_null(&self, idx: usize) -> bool {
        match self.logical_dtype {
            LogicalDType::Continuous => self
                .f64_values
                .as_ref()
                .map_or(true, |v| v.get(idx).map_or(true, |x| x.is_none())),
            LogicalDType::Categorical => self
                .cat_values
                .as_ref()
                .map_or(true, |v| v.get(idx).map_or(true, |x| x.is_none())),
            LogicalDType::Integer => self
                .i64_values
                .as_ref()
                .map_or(true, |v| v.get(idx).map_or(true, |x| x.is_none())),
        }
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
        let n_rows = df.height();

        // Materialize columns
        let columns: Vec<PolarsColumnView> = df
            .get_columns()
            .iter()
            .map(|c| PolarsColumnView::from_series(c.as_series().unwrap()))
            .collect();

        let col_index: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name().to_string(), i))
            .collect();

        // Pre-sort indices for every column
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

        Self {
            columns,
            col_index,
            sorted_indices,
            weights_xy: w_xy,
            weights_x: w_x,
            weights_y: w_y,
            n_rows,
        }
    }

    /// Sort indices by column values (ascending, nulls last).
    fn sort_column_indices(col: &PolarsColumnView, n_rows: usize) -> Vec<u32> {
        let mut indices: Vec<u32> = (0..n_rows as u32).collect();

        match col.logical_dtype() {
            LogicalDType::Continuous => {
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
}
