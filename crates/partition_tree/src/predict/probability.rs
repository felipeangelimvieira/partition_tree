use crate::cell::*;
use crate::conf::{TARGET_PREFIX, TargetBehaviour};
use crate::onedimpartition::DynOneDimPartition;
use crate::rules::{BelongsToGeneric, ContinuousInterval, RuleValue};
use polars::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::fmt;

// Probability
pub trait ProbabilityDistribution: Send + Sync + Clone {
    fn pdf(&self, x: Vec<f64>) -> f64;
    fn mean(&self) -> HashMap<String, Vec<f64>>;
    fn mode(&self) -> HashMap<String, Vec<f64>>;
}

#[derive(Clone)]
pub struct PiecewiseConstantDistribution {
    pub cells: Vec<Cell>,
    pub mass: Vec<f64>,
}

impl PiecewiseConstantDistribution {
    fn collect_category_names<T>(rule: &BelongsToGeneric<T>) -> Vec<String>
    where
        T: RuleValue + fmt::Display,
    {
        rule.domain
            .iter()
            .enumerate()
            .filter_map(|(idx, val)| {
                if rule.values.contains(val) {
                    let name = rule
                        .domain_names
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("{}", val));
                    Some(name)
                } else {
                    None
                }
            })
            .collect()
    }

    fn categorical_names_from_partition(part: &dyn DynOneDimPartition) -> Option<Vec<String>> {
        let rule_any = part.rule_any();

        macro_rules! try_collect {
            ($ty:ty) => {
                if let Some(rule) = rule_any.downcast_ref::<BelongsToGeneric<$ty>>() {
                    return Some(Self::collect_category_names::<$ty>(rule));
                }
            };
        }

        try_collect!(usize);
        try_collect!(u8);
        try_collect!(u16);
        try_collect!(u32);
        try_collect!(u64);
        try_collect!(i8);
        try_collect!(i16);
        try_collect!(i32);
        try_collect!(i64);
        try_collect!(bool);
        try_collect!(String);

        None
    }

    pub fn new(cells: Vec<Cell>, mass: Vec<f64>) -> Self {
        assert_eq!(cells.len(), mass.len());
        Self { cells, mass }
    }

    pub fn mean(&self) -> HashMap<String, Vec<f64>> {
        // Compute the weighted mean across
        let mut overall_mean = HashMap::new();
        let total_mass: f64 = self.mass.iter().sum();
        let behave_as_uniform = total_mass == 0.0;
        if behave_as_uniform {
            //println!("Warning: total mass is zero, behaving as uniform distribution");
        }
        for (i, cell) in self.cells.iter().enumerate() {
            let weight = self.mass[i] / total_mass;

            let cell_means = cell.mean();
            for (var, mean_vec) in cell_means {
                if behave_as_uniform {
                    // Uniform weighting
                    overall_mean
                        .entry(var)
                        .and_modify(|e: &mut Vec<f64>| {
                            for (j, val) in mean_vec.iter().enumerate() {
                                e[j] += val / (self.cells.len() as f64);
                            }
                        })
                        .or_insert(mean_vec.clone());
                    continue;
                }
                let weighted_mean: Vec<f64> = mean_vec.iter().map(|m| m * weight).collect();
                overall_mean
                    .entry(var)
                    .and_modify(|e: &mut Vec<f64>| {
                        for (j, val) in weighted_mean.iter().enumerate() {
                            e[j] += val;
                        }
                    })
                    .or_insert(weighted_mean);
            }
        }
        overall_mean
    }

    fn pdf_with_volume<F>(&self, df: &DataFrame, volume_fn: F) -> Vec<f64>
    where
        F: Fn(&Cell) -> f64,
    {
        let subset = UInt32Chunked::new(PlSmallStr::from_static("idx"), 0u32..(df.height() as u32));
        let total_mass = self.mass.iter().sum::<f64>();

        let mut pdf_values = vec![0.0; df.height()];

        for (i, cell) in self.cells.iter().enumerate() {
            let matches = cell
                .match_dataframe_return_mask(df, &subset, TargetBehaviour::Only)
                .expect("match_dataframe_return_mask in pdf");
            let contrib: f64;
            if total_mass == 0.0 {
                contrib = 1.0 / (volume_fn(cell) * self.cells.len() as f64);
            } else {
                contrib = self.mass[i] / (volume_fn(cell) * total_mass);
            }

            for (row_idx, is_match) in matches.into_iter().enumerate() {
                if is_match {
                    pdf_values[row_idx] += contrib;
                }
            }
        }

        pdf_values
    }

    pub fn mass(&self, df: &DataFrame) -> Vec<f64> {
        self.pdf_with_volume(df, |_| 1.0)
    }

    pub fn pdf(&self, df: &DataFrame) -> Vec<f64> {
        self.pdf_with_volume(df, |cell: &Cell| cell.target_volume())
    }

    pub fn pdf_scaled(&self, df: &DataFrame) -> Vec<f64> {
        self.pdf_with_volume(df, |cell: &Cell| cell.target_relative_volume())
    }

    fn pdf_with_volume_single<F>(&self, row: &HashMap<String, Box<dyn Any>>, volume_fn: F) -> f64
    where
        F: Fn(&Cell) -> f64,
    {
        let total_mass = self.mass.iter().sum::<f64>();
        let mut pdf_value = 0.0;

        for (i, cell) in self.cells.iter().enumerate() {
            let matches = cell.match_hashmap(row, TargetBehaviour::Only);
            if matches {
                let contrib: f64;
                if total_mass == 0.0 {
                    contrib = 1.0 / (volume_fn(cell) * self.cells.len() as f64);
                } else {
                    contrib = self.mass[i] / (volume_fn(cell) * total_mass);
                }

                pdf_value += contrib;
            }
        }

        pdf_value
    }

    pub fn pdf_single(&self, row: &HashMap<String, Box<dyn Any>>) -> f64 {
        self.pdf_with_volume_single(row, |cell: &Cell| cell.target_volume())
    }

    pub fn mass_single(&self, row: &HashMap<String, Box<dyn Any>>) -> f64 {
        self.pdf_with_volume_single(row, |_| 1.0)
    }

    pub fn pdf_scaled_single(&self, row: &HashMap<String, Box<dyn Any>>) -> f64 {
        self.pdf_with_volume_single(row, |cell: &Cell| cell.target_relative_volume())
    }

    /// Returns, for each cell with a continuous target interval, the constant pdf value
    /// on that interval together with the interval bounds and closure flags.
    /// Returns: Vec<(pdf_val, (low, high, lower_closed, upper_closed))>
    pub fn pdf_with_intervals(&self) -> Vec<(f64, (f64, f64, bool, bool))> {
        let total_mass = self.mass.iter().sum::<f64>();
        let behave_as_uniform = total_mass == 0.0;

        let mut out = Vec::with_capacity(self.cells.len());

        for (i, cell) in self.cells.iter().enumerate() {
            let target_rule = cell
                .partitions
                .iter()
                .find(|(name, _)| name.starts_with(TARGET_PREFIX))
                .map(|(_, part)| part.rule_any());

            if let Some(rule_any) = target_rule {
                if let Some(ci) = rule_any.downcast_ref::<ContinuousInterval>() {
                    let volume = cell.target_volume();
                    if volume == 0.0 {
                        continue;
                    }

                    let pdf_val = if behave_as_uniform {
                        1.0 / (volume * self.cells.len() as f64)
                    } else {
                        self.mass[i] / (volume * total_mass)
                    };

                    out.push((pdf_val, (ci.low, ci.high, ci.lower_closed, ci.upper_closed)));
                }
            }
        }

        out
    }

    pub fn unified_from<'a, I>(distributions: I) -> Self
    where
        I: IntoIterator<Item = &'a PiecewiseConstantDistribution>,
    {
        let sources: Vec<_> = distributions.into_iter().collect();
        if sources.is_empty() {
            return Self::new(Vec::new(), Vec::new());
        }

        let mut cells = Vec::new();
        let mut mass = Vec::new();
        let weight = sources.len() as f64;

        for distribution in sources {
            cells.extend(distribution.cells.clone());
            mass.extend(distribution.mass.iter().map(|m| m / weight));
        }

        Self::new(cells, mass)
    }

    /// Export raw masses (probability weights) for each cell.
    pub fn masses(&self) -> &Vec<f64> {
        &self.mass
    }

    /// Export target intervals (low, high) for each cell's target partition.
    /// Currently supports continuous target partitions (ContinuousInterval).
    /// If a cell does not contain a continuous target partition, the interval
    /// is omitted.
    pub fn target_intervals(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(self.cells.len());
        for cell in &self.cells {
            let target_rule = cell
                .partitions
                .iter()
                .find(|(name, _)| name.starts_with(TARGET_PREFIX))
                .map(|(_, part)| part.rule_any());

            if let Some(rule_any) = target_rule {
                if let Some(ci) = rule_any.downcast_ref::<ContinuousInterval>() {
                    out.push((ci.low, ci.high));
                }
            }
        }
        out
    }

    /// Export the mass for each cell together with the categorical values (as strings)
    /// that are active in that cell. Only categorical partitions are included in the
    /// categories map for each cell.
    pub fn masses_with_categories(&self) -> Vec<(f64, HashMap<String, Vec<String>>)> {
        self.cells
            .iter()
            .zip(self.mass.iter())
            .map(|(cell, mass)| {
                let mut categories: HashMap<String, Vec<String>> = HashMap::new();

                for (name, part) in &cell.partitions {
                    if let Some(names) = Self::categorical_names_from_partition(part.as_ref()) {
                        categories.insert(name.clone(), names);
                    }
                }

                (*mass, categories)
            })
            .collect()
    }
}

pub trait PiecewiseConstantDistributionEnsemble {
    fn ensemble(&self) -> PiecewiseConstantDistribution;
}

impl PiecewiseConstantDistributionEnsemble for [PiecewiseConstantDistribution] {
    fn ensemble(&self) -> PiecewiseConstantDistribution {
        PiecewiseConstantDistribution::unified_from(self.iter())
    }
}
