use bincode;
use estimators::api::*;
use partition_tree::estimators::{PartitionForest, PartitionTree};
use partition_tree::loss::{
    BalancedLogLoss, ConditionalLogLoss, LossFunc, MeanIntegratedSquaredError,
};
use partition_tree::predict::PiecewiseConstantDistribution;
use partition_tree::rules::{BelongsTo, ContinuousInterval, IntegerInterval};
use polars::prelude::DataFrame as PolarsDataFrame;
use polars::prelude::*;
use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::*;
use pyo3_polars::{PolarsAllocator, PyDataFrame};

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// Convert a single `FittedNode` (by arena index) into a Python dict.
fn fitted_node_to_dict<'py>(
    py: Python<'py>,
    idx: usize,
    node: &partition_tree::FittedNode,
) -> Bound<'py, PyDict> {
    let d = PyDict::new(py);
    d.set_item("node_index", idx).unwrap();
    d.set_item("is_leaf", node.is_leaf).unwrap();
    d.set_item("depth", node.depth).unwrap();
    d.set_item("parent", node.parent).unwrap();
    d.set_item("left_child", node.left_child).unwrap();
    d.set_item("right_child", node.right_child).unwrap();
    d.set_item("w_xy", node.w_xy).unwrap();
    d.set_item("w_x", node.w_x).unwrap();
    d.set_item("w_y", node.w_y).unwrap();
    d.set_item("conditional_density", node.conditional_density())
        .unwrap();

    let partitions = PyDict::new(py);
    for (name, rule) in &node.cell.rules {
        let info = PyDict::new(py);
        if let Some(ci) = rule.as_any().downcast_ref::<ContinuousInterval>() {
            info.set_item("type", "continuous").unwrap();
            info.set_item("low", ci.low).unwrap();
            info.set_item("high", ci.high).unwrap();
            info.set_item("lower_closed", ci.lower_closed).unwrap();
            info.set_item("upper_closed", ci.upper_closed).unwrap();
        } else if let Some(ii) = rule.as_any().downcast_ref::<IntegerInterval>() {
            info.set_item("type", "integer").unwrap();
            info.set_item("low", ii.low).unwrap();
            info.set_item("high", ii.high).unwrap();
        } else if let Some(bt) = rule.as_any().downcast_ref::<BelongsTo>() {
            info.set_item("type", "categorical").unwrap();
            let cats: Vec<String> = bt
                .values
                .iter()
                .filter_map(|&i| bt.domain_names.get(i).cloned())
                .collect();
            info.set_item("categories", cats).unwrap();
        }
        partitions.set_item(name, info).unwrap();
    }
    d.set_item("partitions", partitions).unwrap();
    d
}

#[pyclass(module = "pyo3_partition_tree")]
pub struct PyPartitionTree {
    inner: PartitionTree,
}

#[pymethods]
impl PyPartitionTree {
    /// Serialize the tree state for pickling
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Deserialize the tree state for unpickling
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode::deserialize(state.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Deserialization error: {}", e))
        })?;
        Ok(())
    }

    #[new]
    #[pyo3(signature = (
        max_leaves = 101,
        boundaries_expansion_factor = 0.1,
        min_samples_xy = 1.0,
        min_samples_x = 1.0,
        min_samples_y = 1.0,
        min_gain = 0.0,
        min_volume_fraction = 0.0,
        max_depth = usize::MAX,
        min_samples_split = 2.0,
        max_samples = None,
        max_features = None,
        loss = None,
        seed = None,
    ))]
    pub fn new(
        max_leaves: usize,
        boundaries_expansion_factor: f64,
        min_samples_xy: f64,
        min_samples_x: f64,
        min_samples_y: f64,
        min_gain: f64,
        min_volume_fraction: f64,
        max_depth: usize,
        min_samples_split: f64,
        max_samples: Option<f64>,
        max_features: Option<f64>,
        loss: Option<String>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let loss_obj: Option<Box<dyn LossFunc>> = match loss {
            Some(l) => match l.as_str() {
                "conditional_log_loss" => {
                    Ok(Some(Box::new(ConditionalLogLoss) as Box<dyn LossFunc>))
                }
                "balanced_log_loss" => Ok(Some(Box::new(BalancedLogLoss) as Box<dyn LossFunc>)),
                "mise" => Ok(Some(
                    Box::new(MeanIntegratedSquaredError) as Box<dyn LossFunc>
                )),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid loss function. Supported: 'conditional_log_loss', 'balanced_log_loss', 'mise'",
                )),
            },
            None => Ok(None),
        }?;

        Ok(Self {
            inner: PartitionTree::new(
                max_leaves,
                boundaries_expansion_factor,
                min_samples_xy,
                min_samples_x,
                min_samples_y,
                min_gain,
                min_volume_fraction,
                max_depth,
                min_samples_split,
                max_samples,
                max_features,
                loss_obj,
                seed,
            ),
        })
    }

    pub fn fit(
        &mut self,
        x: PyDataFrame,
        y: PyDataFrame,
        sample_weights: Option<PyDataFrame>,
    ) -> PyResult<()> {
        let x_df: PolarsDataFrame = x.into();
        let y_df: PolarsDataFrame = y.into();

        if x_df.height() != y_df.height() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "X and y must have the same number of rows",
            ));
        }

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

    pub fn predict_proba(&self, x: PyDataFrame) -> PyResult<Vec<PyPiecewiseDistribution>> {
        let x_df: PolarsDataFrame = x.into();
        let proba = self
            .inner
            .predict_proba(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_iter()
            .map(|dist| PyPiecewiseDistribution { inner: dist })
            .collect();
        Ok(proba)
    }

    /// Apply the tree to the input data, returning the leaf node index for each sample.
    pub fn apply(&self, x: PyDataFrame) -> PyResult<Vec<usize>> {
        let x_df: PolarsDataFrame = x.into();
        self.inner
            .apply(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get detailed information about all leaves in the tree.
    /// Returns a list of dictionaries, one per leaf, containing:
    /// - leaf_index: index of the leaf node
    /// - depth: depth of the leaf in the tree
    /// - w_xy: joint weight
    /// - w_x: feature weight
    /// - w_y: target weight
    /// - conditional_density: estimated density
    /// - target_volume: target-space volume
    /// - partitions: dict mapping column names to their partition info
    pub fn get_leaves_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let leaves_info = self
            .inner
            .leaves_info()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let tree = self
            .inner
            .tree
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model is not fitted"))?;

        let py_leaves: Vec<PyObject> = leaves_info
            .into_iter()
            .map(|leaf| {
                let leaf_dict = PyDict::new(py);
                leaf_dict.set_item("leaf_index", leaf.index).unwrap();
                leaf_dict.set_item("depth", leaf.depth).unwrap();
                leaf_dict.set_item("w_xy", leaf.w_xy).unwrap();
                leaf_dict.set_item("w_x", leaf.w_x).unwrap();
                leaf_dict.set_item("w_y", leaf.w_y).unwrap();
                leaf_dict
                    .set_item("conditional_density", leaf.conditional_density)
                    .unwrap();
                leaf_dict
                    .set_item("target_volume", leaf.target_volume)
                    .unwrap();

                // Extract partition info from the node's cell rules
                let node = &tree.nodes[leaf.index];
                let partitions_dict = PyDict::new(py);
                for (name, rule) in &node.cell.rules {
                    let info_dict = PyDict::new(py);
                    if let Some(ci) = rule.as_any().downcast_ref::<ContinuousInterval>() {
                        info_dict.set_item("type", "continuous").unwrap();
                        info_dict.set_item("low", ci.low).unwrap();
                        info_dict.set_item("high", ci.high).unwrap();
                        info_dict.set_item("lower_closed", ci.lower_closed).unwrap();
                        info_dict.set_item("upper_closed", ci.upper_closed).unwrap();
                    } else if let Some(ii) = rule.as_any().downcast_ref::<IntegerInterval>() {
                        info_dict.set_item("type", "integer").unwrap();
                        info_dict.set_item("low", ii.low).unwrap();
                        info_dict.set_item("high", ii.high).unwrap();
                    } else if let Some(bt) = rule.as_any().downcast_ref::<BelongsTo>() {
                        info_dict.set_item("type", "categorical").unwrap();
                        let cats: Vec<String> = bt
                            .values
                            .iter()
                            .filter_map(|&idx| bt.domain_names.get(idx).cloned())
                            .collect();
                        info_dict.set_item("categories", cats).unwrap();
                    }
                    partitions_dict.set_item(name, info_dict).unwrap();
                }
                leaf_dict.set_item("partitions", partitions_dict).unwrap();

                leaf_dict.into()
            })
            .collect();

        Ok(PyList::new(py, py_leaves)?.into())
    }

    /// Get information about every node in the fitted tree.
    ///
    /// Returns a list of dicts (length = total nodes), one per node in
    /// arena order. Each dict contains: node_index, is_leaf, depth,
    /// parent, left_child, right_child, w_xy, w_x, w_y,
    /// conditional_density, partitions.
    pub fn get_nodes_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let tree = self
            .inner
            .tree
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model is not fitted"))?;

        let nodes: Vec<PyObject> = tree
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| fitted_node_to_dict(py, idx, node).into())
            .collect();

        Ok(PyList::new(py, nodes)?.into())
    }

    /// Get the split history of the fitted tree.
    ///
    /// Returns a list of dicts, one per split, in best-first order.
    /// Each dict contains: parent_index, col_name, split_kind, gain,
    /// left_child_index, right_child_index.
    pub fn get_split_history(&self, py: Python<'_>) -> PyResult<PyObject> {
        let tree = self
            .inner
            .tree
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model is not fitted"))?;

        let records: Vec<PyObject> = tree
            .split_history
            .iter()
            .map(|rec| {
                let d = PyDict::new(py);
                d.set_item("parent_index", rec.parent_index).unwrap();
                d.set_item("col_name", &rec.col_name).unwrap();
                d.set_item("split_kind", rec.split_kind.to_string())
                    .unwrap();
                d.set_item("gain", rec.gain).unwrap();
                d.set_item("left_child_index", rec.left_child_index)
                    .unwrap();
                d.set_item("right_child_index", rec.right_child_index)
                    .unwrap();
                d.into()
            })
            .collect();

        Ok(PyList::new(py, records)?.into())
    }

    /// Compute feature importances based on cumulative gain from all splits.
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
        self.inner
            .feature_importances(normalize)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(module = "pyo3_partition_tree")]
pub struct PyPartitionForest {
    inner: PartitionForest,
}

#[pymethods]
impl PyPartitionForest {
    /// Serialize the forest state for pickling
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Deserialize the forest state for unpickling
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode::deserialize(state.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Deserialization error: {}", e))
        })?;
        Ok(())
    }

    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_estimators = 100,
        max_leaves = 101,
        boundaries_expansion_factor = 0.1,
        min_samples_xy = 1.0,
        min_samples_x = 1.0,
        min_samples_y = 1.0,
        min_gain = 0.0,
        min_volume_fraction = 0.0,
        max_depth = usize::MAX,
        min_samples_split = 2.0,
        max_samples = None,
        max_features = None,
        loss = None,
        seed = None,
    ))]
    pub fn new(
        n_estimators: usize,
        max_leaves: usize,
        boundaries_expansion_factor: f64,
        min_samples_xy: f64,
        min_samples_x: f64,
        min_samples_y: f64,
        min_gain: f64,
        min_volume_fraction: f64,
        max_depth: usize,
        min_samples_split: f64,
        max_samples: Option<f64>,
        max_features: Option<f64>,
        loss: Option<String>,
        seed: Option<usize>,
    ) -> PyResult<Self> {
        let loss_obj: Option<Box<dyn LossFunc>> = match loss {
            Some(l) => match l.as_str() {
                "conditional_log_loss" => {
                    Ok(Some(Box::new(ConditionalLogLoss) as Box<dyn LossFunc>))
                }
                "balanced_log_loss" => Ok(Some(Box::new(BalancedLogLoss) as Box<dyn LossFunc>)),
                "mise" => Ok(Some(
                    Box::new(MeanIntegratedSquaredError) as Box<dyn LossFunc>
                )),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid loss function. Supported: 'conditional_log_loss', 'balanced_log_loss', 'mise'",
                )),
            },
            None => Ok(None),
        }?;

        Ok(Self {
            inner: PartitionForest::new(
                n_estimators,
                max_leaves,
                boundaries_expansion_factor,
                min_samples_xy,
                min_samples_x,
                min_samples_y,
                min_gain,
                min_volume_fraction,
                max_depth,
                min_samples_split,
                max_samples,
                max_features,
                loss_obj,
                seed,
            ),
        })
    }

    pub fn fit(
        &mut self,
        x: PyDataFrame,
        y: PyDataFrame,
        sample_weights: Option<PyDataFrame>,
    ) -> PyResult<()> {
        let x_df: PolarsDataFrame = x.into();
        let y_df: PolarsDataFrame = y.into();

        if x_df.height() != y_df.height() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "X and y must have the same number of rows",
            ));
        }

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

    pub fn predict_proba(&self, x: PyDataFrame) -> PyResult<Vec<PyPiecewiseDistribution>> {
        let x_df: PolarsDataFrame = x.into();
        let proba = self
            .inner
            .predict_proba(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_iter()
            .map(|dist| PyPiecewiseDistribution { inner: dist })
            .collect();
        Ok(proba)
    }

    pub fn predict_trees_proba(
        &self,
        x: PyDataFrame,
    ) -> PyResult<Vec<Vec<PyPiecewiseDistribution>>> {
        let x_df: PolarsDataFrame = x.into();
        let proba_per_tree = self
            .inner
            .predict_trees_proba(&x_df)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_iter()
            .map(|tree_dists| {
                tree_dists
                    .into_iter()
                    .map(|dist| PyPiecewiseDistribution { inner: dist })
                    .collect()
            })
            .collect();
        Ok(proba_per_tree)
    }

    /// Get information about every node for each tree in the forest.
    ///
    /// Returns a list of lists of dicts — one inner list per tree.
    /// Each dict has the same schema as `PyPartitionTree.get_nodes_info`.
    pub fn get_nodes_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let trees = self
            .inner
            .trees
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model is not fitted"))?;

        let all_trees: Vec<PyObject> = trees
            .iter()
            .map(|tree| {
                let nodes: Vec<PyObject> = tree
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(idx, node)| fitted_node_to_dict(py, idx, node).into())
                    .collect();
                PyList::new(py, nodes).unwrap().into()
            })
            .collect();

        Ok(PyList::new(py, all_trees)?.into())
    }

    /// Get the split history for each tree in the forest.
    ///
    /// Returns a list of lists of dicts — one inner list per tree.
    /// Each dict contains: parent_index, col_name, split_kind, gain,
    /// left_child_index, right_child_index.
    pub fn get_split_history(&self, py: Python<'_>) -> PyResult<PyObject> {
        let trees = self
            .inner
            .trees
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model is not fitted"))?;

        let all_trees: Vec<PyObject> = trees
            .iter()
            .map(|tree| {
                let records: Vec<PyObject> = tree
                    .split_history
                    .iter()
                    .map(|rec| {
                        let d = PyDict::new(py);
                        d.set_item("parent_index", rec.parent_index).unwrap();
                        d.set_item("col_name", &rec.col_name).unwrap();
                        d.set_item("split_kind", rec.split_kind.to_string())
                            .unwrap();
                        d.set_item("gain", rec.gain).unwrap();
                        d.set_item("left_child_index", rec.left_child_index)
                            .unwrap();
                        d.set_item("right_child_index", rec.right_child_index)
                            .unwrap();
                        d.into()
                    })
                    .collect();
                PyList::new(py, records).unwrap().into()
            })
            .collect();

        Ok(PyList::new(py, all_trees)?.into())
    }

    /// Compute feature importances aggregated across all trees.
    #[pyo3(signature = (normalize = true))]
    pub fn get_feature_importances(&self, normalize: bool) -> PyResult<HashMap<String, f64>> {
        self.inner
            .feature_importances(normalize)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
pub struct PyPiecewiseDistribution {
    inner: PiecewiseConstantDistribution,
}

#[pymethods]
impl PyPiecewiseDistribution {
    /// Weighted mean vector across all cells.
    /// Returns a dict mapping target column names to mean values.
    pub fn mean(&self) -> PyResult<HashMap<String, Vec<f64>>> {
        Ok(self.inner.mean_vector())
    }

    /// Total mass of the distribution.
    pub fn total_mass(&self) -> f64 {
        self.inner.total_mass()
    }

    /// Number of cells in this distribution.
    pub fn n_cells(&self) -> usize {
        self.inner.n_cells()
    }

    /// For continuous targets: extract (density, low, high) segments.
    pub fn pdf_segments(&self) -> Vec<(f64, f64, f64)> {
        self.inner.pdf_segments()
    }

    /// For categorical targets: per-column probability vectors.
    pub fn category_probabilities(&self) -> HashMap<String, Vec<f64>> {
        self.inner.category_probabilities()
    }

    /// Get domain names for a categorical target column.
    pub fn categorical_domain_names(&self, col_name: &str) -> Option<Vec<String>> {
        self.inner.categorical_domain_names(col_name)
    }
}

#[pymodule]
fn pyo3_partition_tree(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPartitionTree>()?;
    m.add_class::<PyPartitionForest>()?;
    m.add_class::<PyPiecewiseDistribution>()?;
    Ok(())
}
