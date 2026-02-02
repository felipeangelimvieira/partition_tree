//! Density functions and related traits
//! Density functions define the probability density function (pdf), that is unware of the
//! support over which it is defined.
//!
//! They implement the following:
//!
//! - `pdf(x)`: Probability density function at point x
//! - `mean()`: Mean of the distribution, if defined.
//! - `integrate_rule(rule)`: Integrate the density over a given rule (interval or categorical set).

use crate::rules::*;
use serde::{Deserialize, Serialize};
use statrs::function::erf::erf;
use std::f64::consts::PI;
use std::marker::PhantomData;

// Generic density trait with associated types for input and mean.
// Object safety is intentionally dropped in favor of full generic usage.
pub trait DensityFunction: Send + Sync {
    type Input;
    fn eval(&self, x: &Self::Input) -> f64;
    fn mean(&self) -> Option<Vec<Self::Input>>; // None when undefined
    fn integrate_wo_constant<R: Rule<Self::Input>>(&self, rule: &R) -> f64 {
        rule.volume()
    }
    fn integrate_rule<R: Rule<Self::Input>>(&self, rule: &R) -> f64;
}

// ------------------- ConstantDensity (generic) -------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantDensity<T> {
    pub c: f64,              // constant value
    _marker: PhantomData<T>, // tracks the input type
}

impl<T> ConstantDensity<T> {
    pub fn new(c: f64) -> Self {
        Self {
            c,
            _marker: PhantomData,
        }
    }
}

impl<T> DensityFunction for ConstantDensity<T>
where
    T: Send + Sync, // to satisfy DensityFunction supertraits
{
    type Input = T;

    fn eval(&self, _x: &Self::Input) -> f64 {
        self.c
    }

    fn mean(&self) -> Option<Vec<Self::Input>> {
        None // undefined without bounded support
    }

    fn integrate_wo_constant<R: Rule<Self::Input>>(&self, rule: &R) -> f64 {
        rule.volume()
    }

    fn integrate_rule<R: Rule<Self::Input>>(&self, rule: &R) -> f64 {
        self.c * self.integrate_wo_constant(rule)
    }
}

// Convenient aliases
pub type ConstantF64 = ConstantDensity<f64>;
pub type ConstantI32 = ConstantDensity<i32>;
pub type ConstantU32 = ConstantDensity<u32>;

// ------------------- HalfNormalDensity (continuous) -------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalfNormalDensity {
    pub scale: f64,   // sigma
    pub offset: f64,  // mu (location)
    pub side: f64,    // +1 right, -1 left
    pub measure: f64, // external scaling factor
}

impl HalfNormalDensity {
    fn raw_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let sigma = self.scale;
        let norm = (2.0_f64 / PI).sqrt() / sigma; // sqrt(2/pi)/sigma
        norm * (-x * x / (2.0 * sigma * sigma)).exp() * self.measure
    }
    fn raw_cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x.is_infinite() {
            return 1.0;
        }
        let sigma = self.scale;
        // Standard half-normal CDF: erf(x / (sqrt(2)*sigma))
        erf(x / (sigma * 2.0_f64.sqrt()))
    }
}

impl DensityFunction for HalfNormalDensity {
    type Input = f64;

    fn eval(&self, x: &Self::Input) -> f64 {
        self.raw_pdf(self.side * (x - self.offset))
    }

    fn mean(&self) -> Option<Vec<Self::Input>> {
        // Mean of half-normal = offset + side * sigma * sqrt(2/pi), scaled by measure
        let base = self.offset + self.side * self.scale * (2.0 / PI).sqrt();
        Some(vec![base])
    }

    fn integrate_wo_constant<R: Rule<Self::Input>>(&self, rule: &R) -> f64 {
        if let Some(iv) = rule.as_any().downcast_ref::<ContinuousInterval>() {
            let a = self.side * (iv.low - self.offset);
            let b = self.side * (iv.high - self.offset);
            // For side > 0 (increasing transform), integral = CDF(b) - CDF(a)
            // For side < 0 (decreasing transform), integral = CDF(a) - CDF(b)
            if self.side >= 0.0 {
                self.raw_cdf(b) - self.raw_cdf(a)
            } else {
                self.raw_cdf(a) - self.raw_cdf(b)
            }
        } else {
            panic!("HalfNormalDensity only supports ContinuousInterval rules")
        }
    }

    fn integrate_rule<R: Rule<Self::Input>>(&self, rule: &R) -> f64 {
        self.measure * self.integrate_wo_constant(rule)
    }
}
