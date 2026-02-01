//! This file implements "Cell" and "OneDimPartition".
//!
//! OneDimPartition represents a composition of a rule and a density function.
//!
//! A Cell is an object composed of a set of OneDimPartitions,
//! one for each feature dimension.
//!

use crate::density::*;
use crate::rules::*;
use std::any::Any;
use std::collections::HashSet;

// ------------------- OneDimPartition -------------------

#[derive(Debug, Clone)]
pub struct OneDimPartition<R, D>
where
    R: Rule<D::Input>,
    D: DensityFunction,
{
    pub rule: R,
    pub density: D,
}

impl<R, D> OneDimPartition<R, D>
where
    R: Rule<D::Input>,
    D: DensityFunction,
{
    pub fn new(rule: R, density: D) -> Self {
        Self { rule, density }
    }

    // Evaluate the rule on input data
    pub fn evaluate(&self, data: &[Option<D::Input>]) -> Vec<bool> {
        self.rule.evaluate(data)
    }

    // Compute the volume of the rule
    pub fn volume(&self) -> f64 {
        self.rule.volume()
    }

    // Compute the relative volume of the rule
    pub fn relative_volume(&self) -> f64 {
        self.rule.relative_volume()
    }

    // Compute the mean of the density function
    pub fn mean(&self) -> Vec<f64> {
        self.rule.mean()
    }

    // Integrate the density over the rule
    pub fn integrate(&self) -> f64 {
        self.density.integrate_rule(&self.rule)
    }

    pub fn density(&self, point: D::Input) -> f64 {
        self.density.eval(&point)
    }
    pub fn split(
        &self,
        point: D::Input,
        none_to_left: Option<bool>,
        left_density: Option<D>,
        right_density: Option<D>,
    ) -> (Self, Self)
    where
        D: Clone,
        D::Input: Clone,
    {
        let point = point.into();
        let (left_rule, right_rule) = self.rule.split(point, none_to_left);
        let left_density = left_density.unwrap_or_else(|| self.density.clone());
        let right_density = right_density.unwrap_or_else(|| self.density.clone());
        (
            OneDimPartition::new(left_rule, left_density),
            OneDimPartition::new(right_rule, right_density),
        )
    }
}

// ------------------- Dynamic (type-erased) API -------------------

pub trait DynOneDimPartition: Send + Sync {
    fn clone_box(&self) -> Box<dyn DynOneDimPartition>;

    fn evaluate_dyn(&self, data: &dyn Any) -> Result<Vec<bool>, String>;
    fn eval_one_dyn(&self, value: &dyn Any) -> Result<bool, String>;

    fn density_dyn(&self, point: &dyn Any) -> f64;
    fn integrate_dyn(&self) -> f64;
    fn volume_dyn(&self) -> f64;
    fn relative_volume_dyn(&self) -> f64;

    fn mean_dyn(&self) -> Vec<f64>;

    fn rule_any(&self) -> &dyn Any;
    fn density_any(&self) -> &dyn Any;
    fn as_any(&self) -> &dyn Any;

    fn split_dyn(
        &self,
        point: &dyn Any,
        none_to_left: Option<bool>,
        left_density: Option<&dyn Any>,
        right_density: Option<&dyn Any>,
    ) -> (Box<dyn DynOneDimPartition>, Box<dyn DynOneDimPartition>);

    /// Split by a subset of categorical values (for categorical rules only).
    /// Returns None if the rule is not categorical.
    fn split_subset_dyn(
        &self,
        subset: &HashSet<u32>,
        none_to_left: Option<bool>,
        left_density: Option<&dyn Any>,
        right_density: Option<&dyn Any>,
    ) -> Option<(Box<dyn DynOneDimPartition>, Box<dyn DynOneDimPartition>)>;

    /// Length/diameter metric for geometric exploration.
    /// Continuous rules should return phi-transformed length; categorical rules return volume.
    fn phi_volume(&self) -> f64;
}

impl<R, D> DynOneDimPartition for OneDimPartition<R, D>
where
    R: Rule<D::Input> + 'static + Send + Sync + Clone,
    D: DensityFunction + 'static + Send + Sync + Clone,
    D::Input: 'static + Clone,
{
    fn clone_box(&self) -> Box<dyn DynOneDimPartition> {
        Box::new(self.clone())
    }

    fn density_dyn(&self, point: &dyn Any) -> f64 {
        if let Some(v) = point.downcast_ref::<D::Input>() {
            self.density.eval(v)
        } else {
            panic!("Type mismatch: expected Input type for density evaluation");
        }
    }
    fn evaluate_dyn(&self, data: &dyn Any) -> Result<Vec<bool>, String> {
        // Use owned types with `Any`, then view them as slices.
        if let Some(v) = data.downcast_ref::<Vec<Option<D::Input>>>() {
            Ok(self.evaluate(v))
        } else if let Some(b) = data.downcast_ref::<Box<[Option<D::Input>]>>() {
            Ok(self.evaluate(&b[..]))
        } else if let Some(v) = data.downcast_ref::<Vec<D::Input>>() {
            // Map Vec<D::Input> to Vec<Option<D::Input>> by wrapping each element in Some
            let wrapped: Vec<Option<D::Input>> = v.iter().cloned().map(Some).collect();
            Ok(self.evaluate(&wrapped))
        } else if let Some(b) = data.downcast_ref::<Box<[D::Input]>>() {
            // Map Box<[D::Input]> to Vec<Option<D::Input>> by wrapping each element in Some
            let wrapped: Vec<Option<D::Input>> = b.iter().cloned().map(Some).collect();
            Ok(self.evaluate(&wrapped))
        } else {
            Err(format!(
                "Type mismatch: expected Vec<Option<Input>>, Box<[Option<Input>]>, Vec<Input>, or Box<[Input]>, but found type: {}",
                std::any::type_name_of_val(data)
            ))
        }
    }

    fn eval_one_dyn(&self, value: &dyn Any) -> Result<bool, String> {
        if let Some(v) = value.downcast_ref::<Option<D::Input>>() {
            Ok(self.rule.evaluate(std::slice::from_ref(v))[0])
        } else if let Some(v) = value.downcast_ref::<D::Input>() {
            // Map D::Input to Option<D::Input> by wrapping in Some
            let wrapped = Some(v.clone());
            Ok(self.rule.evaluate(std::slice::from_ref(&wrapped))[0])
        } else if let Some(v) = value.downcast_ref::<&Option<D::Input>>() {
            // Handle reference to Option<D::Input>
            Ok(self.rule.evaluate(std::slice::from_ref(v))[0])
        } else if let Some(v) = value.downcast_ref::<&D::Input>() {
            // Handle reference to D::Input
            let wrapped = Some((*v).clone());
            Ok(self.rule.evaluate(std::slice::from_ref(&wrapped))[0])
        } else if let Some(boxed) = value.downcast_ref::<Box<dyn Any>>() {
            // Handle boxed dyn Any - recursively try to extract the inner value
            return self.eval_one_dyn(boxed.as_ref());
        } else {
            // Create a more detailed error message with debugging information
            Err(format!(
                "Type mismatch: expected Option<{}>, {}, &Option<{}>, &{}, or Box<dyn Any>, but found type: {} (type_id: {:?}). \
                This often happens when a Box<dyn Any> contains a different type than expected. \
                Expected Input type: {}",
                std::any::type_name::<D::Input>(),
                std::any::type_name::<D::Input>(),
                std::any::type_name::<D::Input>(),
                std::any::type_name::<D::Input>(),
                std::any::type_name_of_val(value),
                value.type_id(),
                std::any::type_name::<D::Input>()
            ))
        }
    }

    fn integrate_dyn(&self) -> f64 {
        self.integrate()
    }
    fn volume_dyn(&self) -> f64 {
        self.volume()
    }
    fn relative_volume_dyn(&self) -> f64 {
        self.relative_volume()
    }
    fn mean_dyn(&self) -> Vec<f64> {
        self.mean()
    }

    fn rule_any(&self) -> &dyn Any {
        &self.rule
    }
    fn density_any(&self) -> &dyn Any {
        &self.density
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn split_dyn(
        &self,
        point: &dyn Any,
        none_to_left: Option<bool>,
        left_density: Option<&dyn Any>,
        right_density: Option<&dyn Any>,
    ) -> (Box<dyn DynOneDimPartition>, Box<dyn DynOneDimPartition>) {
        // Downcast point
        let point = if let Some(v) = point.downcast_ref::<D::Input>() {
            v.clone()
        } else {
            panic!("Type mismatch: expected Input type for split point");
        };

        // Downcast left density if provided
        let left_density = if let Some(ld) = left_density {
            if let Some(ld) = ld.downcast_ref::<D>() {
                Some(ld.clone())
            } else {
                panic!("Type mismatch: expected Density type for left density");
            }
        } else {
            None
        };

        // Downcast right density if provided
        let right_density = if let Some(rd) = right_density {
            if let Some(rd) = rd.downcast_ref::<D>() {
                Some(rd.clone())
            } else {
                panic!("Type mismatch: expected Density type for right density");
            }
        } else {
            None
        };

        let (left, right) = self.split(point, none_to_left, left_density, right_density);
        (Box::new(left), Box::new(right))
    }

    fn split_subset_dyn(
        &self,
        subset: &HashSet<u32>,
        none_to_left: Option<bool>,
        _left_density: Option<&dyn Any>,
        _right_density: Option<&dyn Any>,
    ) -> Option<(Box<dyn DynOneDimPartition>, Box<dyn DynOneDimPartition>)> {
        // Check if the rule is a BelongsToU32 (categorical with u32 codes)
        let rule_any = self.rule_any();
        if let Some(belongs_to) = rule_any.downcast_ref::<BelongsToU32>() {
            // Create the split using split_subset
            let (left_rule, right_rule) = belongs_to.split_subset(subset.clone(), none_to_left);

            // Compute new density values based on volumes
            let left_vol = left_rule.volume().max(1.0);
            let right_vol = right_rule.volume().max(1.0);

            let left_part = OneDimPartition::new(left_rule, ConstantU32::new(1.0 / left_vol));
            let right_part = OneDimPartition::new(right_rule, ConstantU32::new(1.0 / right_vol));

            return Some((Box::new(left_part), Box::new(right_part)));
        }

        // Not a categorical rule, return None
        None
    }

    fn phi_volume(&self) -> f64 {
        // ContinuousInterval uses phi-based length; otherwise use rule volume (categorical size).
        self.rule.phi_volume()
    }
}

impl Clone for Box<dyn DynOneDimPartition> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
