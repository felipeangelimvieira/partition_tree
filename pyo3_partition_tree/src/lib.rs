use estimator_api::api::*;
use partition_tree::estimator::*;
use partition_tree::estimator_forest::PartitionForest;
use partition_tree::predict::probability::*;
use partition_tree::tree::{LeafInfo, PartitionInfo};
use polars::prelude::DataFrame as PolarsDataFrame;
use polars::prelude::*;
use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::*;
use pyo3_polars::{PolarsAllocator, PyDataFrame};

#[derive(Debug, Clone)]
pub enum PyValue {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
}
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pyclass]
pub struct PyPartitionTree {
    inner: PartitionTree,
}

#[pymethods]
impl PyPartitionTree {
    #[new]
    #[pyo3(signature = (
        max_iter,
        min_samples_split,
        min_samples_leaf_y,
        min_samples_leaf_x,
        min_samples_leaf,
        min_target_volume,
        max_depth,
        min_split_gain,
        boundaries_expansion_factor,
        min_density_value = 0.0,
        max_density_value = f64::INFINITY,
        max_measure_value = f64::INFINITY,
        exploration_split_budget = 0,
        feature_split_fraction = 0.0,
        seed = None,
    ))]
    pub fn new(
        max_iter: usize,
        min_samples_split: usize,
        min_samples_leaf_y: usize,
        min_samples_leaf_x: usize,
        min_samples_leaf: usize,
        min_target_volume: f64,
        max_depth: usize,
        min_split_gain: f64,
        boundaries_expansion_factor: f64,
        min_density_value: f64,
        max_density_value: f64,
        max_measure_value: f64,
        exploration_split_budget: usize,
        feature_split_fraction: Option<f64>,
        seed: Option<usize>,
    ) -> Self {
        Self {
            inner: PartitionTree::new(
                max_iter,
                min_samples_split,
                min_samples_leaf_y,
                min_samples_leaf_x,
                min_samples_leaf,
                min_target_volume,
                max_depth,
                min_split_gain,
                min_density_value,
                max_density_value,
                max_measure_value,
                boundaries_expansion_factor,
                exploration_split_budget,
                feature_split_fraction,
                seed,
            ),
        }
    }

    pub fn fit(
        &mut self,
        x: PyDataFrame,
        y: PyDataFrame,
        sample_weights: Option<PyDataFrame>,
    ) -> PyResult<()> {
        // Convert PyDataFrame to owned DataFrame, then pass references as required by the trait.

        let x_df: PolarsDataFrame = x.into();
        let y_df: PolarsDataFrame = y.into();
        // Convert sample_weights to Float64Chunked if present, else weights of 1
        let w: Float64Chunked = match sample_weights {
            Some(sw) => {
                let sw_df: PolarsDataFrame = sw.into();
                sw_df
                    .column("sample_weights")
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                    .f64()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                    .clone()
            }
            None => Float64Chunked::full(
                PlSmallStr::from_static("sample_weights"),
                1.0,
                x_df.height(),
            ),
        };

        // IMPORTANT: Estimator::fit returns a new PartitionTree with the trained state (tree moved).
        // We must assign it back to self.inner, otherwise self.inner.tree remains None and predict() fails.
        let fitted = self
            .inner
            .fit(&x_df, &y_df, Some(&w))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner = fitted;
        Ok(())
    }

    pub fn tree_build_status(&self) -> PyResult<String> {
        PyResult::Ok(self.inner.tree_build_status())
    }

    pub fn predict(&self, x: PyDataFrame) -> PyResult<PyDataFrame> {
        let x_df: PolarsDataFrame = x.into();
        let preds = self
            .inner
            .predict(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyDataFrame(preds))
    }

    pub fn predict_proba(&self, x: PyDataFrame) -> PyResult<Vec<PyProbabilityDistributionSingle>> {
        let x_df: PolarsDataFrame = x.into();
        let proba = self
            .inner
            .predict_proba(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .iter()
            .map(|dist| PyProbabilityDistributionSingle {
                inner: dist.clone(),
            })
            .collect();
        Ok(proba)
    }

    pub fn predict_categorical_masses(
        &self,
        x: PyDataFrame,
    ) -> PyResult<Vec<Vec<(f64, HashMap<String, Vec<String>>)>>> {
        let x_df: PolarsDataFrame = x.into();
        let categorical_masses = self
            .inner
            .predict_categorical_masses(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(categorical_masses)
    }

    pub fn tree_info(&self) -> PyResult<String> {
        PyResult::Ok(self.inner.tree_info())
    }

    /// Apply the tree to the input data, returning the leaf index for each sample.
    /// Returns a list where each element is a list of leaf indices (positions in the leaves array)
    /// that the sample belongs to. Use get_leaves_info() to get details about each leaf by position.
    pub fn apply(&self, x: PyDataFrame) -> PyResult<Vec<Vec<usize>>> {
        let x_df: PolarsDataFrame = x.into();
        Ok(self.inner.apply(&x_df))
    }

    /// Get detailed information about all leaves in the tree.
    /// Returns a list of dictionaries, one per leaf, containing:
    /// - leaf_index: index of the leaf node
    /// - depth: depth of the leaf in the tree
    /// - n_samples: number of training samples in this leaf
    /// - partitions: dict mapping column names to their partition info
    ///   - For continuous: {"type": "continuous", "low": float, "high": float, "lower_closed": bool, "upper_closed": bool}
    ///   - For categorical: {"type": "categorical", "categories": [str, ...]}
    /// - indices_xy: list of sample indices matching both X and Y constraints
    /// - indices_x: list of sample indices matching X constraints only
    /// - indices_y: list of sample indices matching Y constraints only
    pub fn get_leaves_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let leaves_info = self.inner.get_leaves_info();

        let py_leaves: Vec<PyObject> = leaves_info
            .into_iter()
            .map(|leaf| {
                let leaf_dict = PyDict::new(py);
                leaf_dict.set_item("leaf_index", leaf.leaf_index).unwrap();
                leaf_dict.set_item("depth", leaf.depth).unwrap();
                leaf_dict.set_item("n_samples", leaf.n_samples).unwrap();
                leaf_dict.set_item("indices_xy", leaf.indices_xy).unwrap();
                leaf_dict.set_item("indices_x", leaf.indices_x).unwrap();
                leaf_dict.set_item("indices_y", leaf.indices_y).unwrap();

                let partitions_dict = PyDict::new(py);
                for (name, info) in leaf.partitions {
                    let info_dict = PyDict::new(py);
                    match info {
                        PartitionInfo::Continuous {
                            low,
                            high,
                            lower_closed,
                            upper_closed,
                        } => {
                            info_dict.set_item("type", "continuous").unwrap();
                            info_dict.set_item("low", low).unwrap();
                            info_dict.set_item("high", high).unwrap();
                            info_dict.set_item("lower_closed", lower_closed).unwrap();
                            info_dict.set_item("upper_closed", upper_closed).unwrap();
                        }
                        PartitionInfo::Categorical { categories } => {
                            info_dict.set_item("type", "categorical").unwrap();
                            info_dict.set_item("categories", categories).unwrap();
                        }
                    }
                    partitions_dict.set_item(name, info_dict).unwrap();
                }
                leaf_dict.set_item("partitions", partitions_dict).unwrap();

                // Add feature_contributions
                let contributions_dict = PyDict::new(py);
                for (feature_name, gain) in leaf.feature_contributions {
                    contributions_dict.set_item(feature_name, gain).unwrap();
                }
                leaf_dict
                    .set_item("feature_contributions", contributions_dict)
                    .unwrap();

                leaf_dict.into()
            })
            .collect();

        Ok(PyList::new(py, py_leaves)?.into())
    }

    /// Compute feature importances based on cumulative gain from all splits.
    /// Each split is counted exactly once (no double-counting across leaves).
    ///
    /// Parameters
    /// ----------
    /// normalize : bool, default True
    ///     If True, normalize importances to sum to 1.0.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary mapping feature names to their importance scores.
    #[pyo3(signature = (normalize = true))]
    pub fn get_feature_importances(&self, normalize: bool) -> PyResult<HashMap<String, f64>> {
        Ok(self.inner.get_feature_importances(normalize))
    }
}

#[pyclass]
pub struct PyPartitionForest {
    inner: PartitionForest,
}

#[pymethods]
impl PyPartitionForest {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_estimators,
        max_iter,
        min_samples_split,
        min_samples_leaf_y,
        min_samples_leaf_x,
        min_samples_leaf,
        min_target_volume,
        max_depth,
        min_split_gain,
        boundaries_expansion_factor,
        max_features,
        max_samples,
        min_density_value = 0.0,
        max_density_value = f64::INFINITY,
        max_measure_value = f64::INFINITY,
        exploration_split_budget = 0,
        feature_split_fraction = 0.0,
        seed = None,
    ))]
    pub fn new(
        n_estimators: usize,
        max_iter: usize,
        min_samples_split: usize,
        min_samples_leaf_y: usize,
        min_samples_leaf_x: usize,
        min_samples_leaf: usize,
        min_target_volume: f64,
        max_depth: usize,
        min_split_gain: f64,
        boundaries_expansion_factor: f64,
        max_features: Option<f64>,
        max_samples: Option<f64>,
        min_density_value: f64,
        max_density_value: f64,
        max_measure_value: f64,
        exploration_split_budget: usize,
        feature_split_fraction: Option<f64>,
        seed: Option<usize>,
    ) -> Self {
        Self {
            inner: PartitionForest::new(
                n_estimators,
                max_iter,
                min_samples_split,
                min_samples_leaf_y,
                min_samples_leaf_x,
                min_samples_leaf,
                min_target_volume,
                max_depth,
                min_split_gain,
                min_density_value,
                max_density_value,
                max_measure_value,
                boundaries_expansion_factor,
                max_features,
                max_samples,
                exploration_split_budget,
                feature_split_fraction,
                seed,
            ),
        }
    }

    pub fn fit(
        &mut self,
        x: PyDataFrame,
        y: PyDataFrame,
        sample_weights: Option<PyDataFrame>,
    ) -> PyResult<()> {
        let x_df: PolarsDataFrame = x.into();
        let y_df: PolarsDataFrame = y.into();

        let w: Float64Chunked = match sample_weights {
            Some(sw) => {
                let sw_df: PolarsDataFrame = sw.into();
                sw_df
                    .column("sample_weights")
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                    .f64()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                    .clone()
            }
            None => Float64Chunked::full(
                PlSmallStr::from_static("sample_weights"),
                1.0,
                x_df.height(),
            ),
        };

        let fitted = self
            .inner
            .fit(&x_df, &y_df, Some(&w))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner = fitted;
        Ok(())
    }

    pub fn predict(&self, x: PyDataFrame) -> PyResult<PyDataFrame> {
        let x_df: PolarsDataFrame = x.into();
        let preds = self
            .inner
            .predict(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyDataFrame(preds))
    }

    pub fn predict_proba(&self, x: PyDataFrame) -> PyResult<Vec<PyProbabilityDistributionSingle>> {
        let x_df: PolarsDataFrame = x.into();
        let proba = self
            .inner
            .predict_proba(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_iter()
            .map(|dist| PyProbabilityDistributionSingle { inner: dist })
            .collect();
        Ok(proba)
    }

    pub fn predict_trees_proba(
        &self,
        x: PyDataFrame,
    ) -> PyResult<Vec<Vec<PyProbabilityDistributionSingle>>> {
        let x_df: PolarsDataFrame = x.into();
        let proba_per_tree = self
            .inner
            .predict_trees_proba(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_iter()
            .map(|tree_dists| {
                tree_dists
                    .into_iter()
                    .map(|dist| PyProbabilityDistributionSingle { inner: dist })
                    .collect()
            })
            .collect();
        Ok(proba_per_tree)
    }
}

#[pyclass]
pub struct PyProbabilityDistributionSingle {
    inner: PiecewiseConstantDistribution,
}

#[pymethods]
impl PyProbabilityDistributionSingle {
    pub fn pdf_single(&self, row: &Bound<'_, PyDict>) -> PyResult<f64> {
        // Convert PyDict to HashMap<String, PyValue>
        let mut map: HashMap<String, PyValue> = HashMap::new();

        for (key, value) in row.iter() {
            let key_str = key.extract::<String>()?;

            // Try to extract different types in order of preference
            let py_value = if let Ok(f) = value.extract::<f64>() {
                PyValue::Float(f)
            } else if let Ok(i) = value.extract::<i64>() {
                PyValue::Int(i)
            } else if let Ok(s) = value.extract::<String>() {
                PyValue::String(s)
            } else if let Ok(b) = value.extract::<bool>() {
                PyValue::Bool(b)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported type for key '{}'",
                    key_str
                )));
            };

            map.insert(key_str, py_value);
        }

        // You'll need to adapt this part based on what your pdf_single method expects
        // For now, converting back to the original format
        let mut any_map: HashMap<String, Box<dyn std::any::Any>> = HashMap::new();
        for (k, v) in map {
            match v {
                PyValue::Float(f) => {
                    any_map.insert(k, Box::new(f));
                }
                PyValue::Int(i) => {
                    any_map.insert(k, Box::new(i as i32));
                } // Convert to f64 if needed
                PyValue::String(s) => {
                    any_map.insert(k, Box::new(s));
                }
                PyValue::Bool(b) => {
                    any_map.insert(k, Box::new(b));
                }
            }
        }

        let result = self.inner.pdf_single(&any_map);
        Ok(result)
    }
    //pub fn predict_proba(&self, )

    pub fn mean(&self) -> PyResult<HashMap<String, Vec<f64>>> {
        let result = self.inner.mean();
        Ok(result)
    }

    pub fn masses(&self) -> Vec<f64> {
        self.inner.masses().clone()
    }

    pub fn masses_with_categories(&self) -> Vec<(f64, HashMap<String, Vec<String>>)> {
        self.inner.masses_with_categories()
    }

    pub fn intervals(&self) -> Vec<(f64, f64)> {
        self.inner.target_intervals()
    }

    pub fn pdf_with_intervals(&self) -> Vec<(f64, (f64, f64, bool, bool))> {
        self.inner.pdf_with_intervals()
    }
}

// Partition Forest

#[pymodule]
fn partition_tree_python(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPartitionTree>()?;
    m.add_class::<PyPartitionForest>()?;
    Ok(())
}
