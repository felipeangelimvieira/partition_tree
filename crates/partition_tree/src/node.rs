use core::panic;

use crate::cell::*;
use crate::conf::*;
use crate::density::*;
use crate::dtype_adapter::*;
use crate::onedimpartition::*;
use crate::rules::*;
use polars::prelude::*;
use rand::Rng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};
#[derive(Clone)]
pub struct Node {
    pub indices_xy: UInt32Chunked,
    pub indices_x: UInt32Chunked,
    pub indices_y: UInt32Chunked,
    pub cell: Cell,
    pub depth: usize,
    pub dataset_size: f64,
}

impl Node {
    pub fn new(
        indices_xy: UInt32Chunked,
        indices_x: UInt32Chunked,
        indices_y: UInt32Chunked,
        cell: Cell,
        depth: usize,
        dataset_size: f64,
    ) -> Self {
        Self {
            indices_xy,
            indices_x,
            indices_y,
            cell,
            depth,
            dataset_size,
        }
    }

    pub fn is_root(&self) -> bool {
        self.depth == 0
    }

    // pub fn match_dataframe(...)
    pub fn split(
        &self,
        df: &DataFrame,
        name: &str,
        point: &dyn std::any::Any,
        none_to_left: Option<bool>,
        left_density: Option<&dyn std::any::Any>,
        right_density: Option<&dyn std::any::Any>,
    ) -> (Self, Self) {
        let (left_cell, right_cell) =
            self.cell
                .split(name, point, none_to_left, left_density, right_density);

        let left_indices_xy = left_cell
            .match_dataframe_xy(&df, &self.indices_xy)
            .expect("match_dataframe_xy");
        let left_indices_x = left_cell
            .match_dataframe_x(&df, &self.indices_x)
            .expect("match_dataframe_x");
        let left_indices_y = left_cell
            .match_dataframe_y(&df, &self.indices_y)
            .expect("match_dataframe_y");

        let right_indices_xy = right_cell
            .match_dataframe_xy(&df, &self.indices_xy)
            .expect("match_dataframe_xy");
        let right_indices_x = right_cell
            .match_dataframe_x(&df, &self.indices_x)
            .expect("match_dataframe_x");
        let right_indices_y = right_cell
            .match_dataframe_y(&df, &self.indices_y)
            .expect("match_dataframe_y");

        let left_node = Node::new(
            left_indices_xy,
            left_indices_x,
            left_indices_y,
            left_cell,
            self.depth + 1,
            self.dataset_size,
        );
        let right_node = Node::new(
            right_indices_xy,
            right_indices_x,
            right_indices_y,
            right_cell,
            self.depth + 1,
            self.dataset_size,
        );
        (left_node, right_node)
    }

    /// Split by a subset of categorical values.
    /// Categories in the subset go to the left child, others go to the right.
    pub fn split_categorical_subset(
        &self,
        df: &DataFrame,
        name: &str,
        subset: &HashSet<u32>,
        none_to_left: Option<bool>,
    ) -> (Self, Self) {
        let (left_cell, right_cell) = self.cell.split_categorical_subset(name, subset, none_to_left);

        let left_indices_xy = left_cell
            .match_dataframe_xy(&df, &self.indices_xy)
            .expect("match_dataframe_xy");
        let left_indices_x = left_cell
            .match_dataframe_x(&df, &self.indices_x)
            .expect("match_dataframe_x");
        let left_indices_y = left_cell
            .match_dataframe_y(&df, &self.indices_y)
            .expect("match_dataframe_y");

        let right_indices_xy = right_cell
            .match_dataframe_xy(&df, &self.indices_xy)
            .expect("match_dataframe_xy");
        let right_indices_x = right_cell
            .match_dataframe_x(&df, &self.indices_x)
            .expect("match_dataframe_x");
        let right_indices_y = right_cell
            .match_dataframe_y(&df, &self.indices_y)
            .expect("match_dataframe_y");

        let left_node = Node::new(
            left_indices_xy,
            left_indices_x,
            left_indices_y,
            left_cell,
            self.depth + 1,
            self.dataset_size,
        );
        let right_node = Node::new(
            right_indices_xy,
            right_indices_x,
            right_indices_y,
            right_cell,
            self.depth + 1,
            self.dataset_size,
        );
        (left_node, right_node)
    }

    pub fn default_from_dataframe(
        df: &DataFrame,
        boundaries_expansion_factor: f64,
        max_samples: Option<f64>,
        rng: &mut StdRng,
    ) -> Self {
        let idx: Vec<u32> = max_samples
            .map(|fraction| {
                let sample_size = ((df.height() as f64) * fraction).ceil() as usize;
                (0..sample_size)
                    .map(|_| rng.random_range(0..df.height()) as u32)
                    .collect()
            })
            .unwrap_or_else(|| (0u32..(df.height() as u32)).collect());

        let partitions: HashMap<String, Box<dyn DynOneDimPartition>> = df
            .get_columns()
            .iter()
            .map(|c| {
                let s = c.as_series().unwrap();
                let name: String = s.name().to_string();

                let adapter = DtypeAdapter::new(s);
                let partition = adapter.default_partition(s, boundaries_expansion_factor);

                (name, partition)
            })
            .collect();

        let cell = Cell::from_partitions(partitions);

        Node::new(
            UInt32Chunked::from_slice("idx".into(), &idx),
            UInt32Chunked::from_slice("idx".into(), &idx),
            UInt32Chunked::from_slice("idx".into(), &idx),
            cell,
            0,
            idx.len() as f64,
        )
    }

    pub fn conditional_measure(&self) -> f64 {
        let n_xy = self.indices_xy.len() as f64;
        let n_x = self.indices_x.len() as f64;
        if n_xy == 0.0 { 0.0 } else { n_xy / n_x }
    }

    pub fn balancing_weight_measure(&self) -> f64 {
        let n_xy = self.indices_xy.len() as f64;
        let n_x = self.indices_x.len() as f64;
        let n_y = self.indices_y.len() as f64;
        if n_xy == 0.0 {
            0.0
        } else {
            n_xy * self.dataset_size / (n_x * n_y)
        }
    }
}
