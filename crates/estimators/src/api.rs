use polars::prelude::*;
use std::any::Any;
use std::collections::HashMap;

#[derive(Debug)]
pub enum FitError {
    InvalidInput(String),
    Numerical(String),
    // add more as needed
}

#[derive(Debug)]
pub enum PredictError {
    NotFitted,
    InvalidInput(String),
    Numerical(String),
    Unknown(String),
}

impl PredictError {
    pub fn to_string(&self) -> String {
        match self {
            PredictError::NotFitted => "Model is not fitted".to_string(),
            PredictError::InvalidInput(msg) => format!("Invalid input: {}", msg),
            PredictError::Numerical(msg) => format!("Numerical error: {}", msg),
            PredictError::Unknown(msg) => format!("Unknown error: {}", msg),
        }
    }
}

impl FitError {
    pub fn to_string(&self) -> String {
        match self {
            FitError::InvalidInput(msg) => format!("Invalid input: {}", msg),
            FitError::Numerical(msg) => format!("Numerical error: {}", msg),
        }
    }
}

pub trait Estimator: Sized {
    fn fit(
        &mut self,
        X: &DataFrame,
        y: &DataFrame,
        sample_weights: Option<&Float64Chunked>,
    ) -> Result<Self, FitError> {
        self._fit_impl(&X, &y, sample_weights)
    }

    fn _fit_impl(
        &mut self,
        X: &DataFrame,
        y: &DataFrame,
        sample_weights: Option<&Float64Chunked>,
    ) -> Result<Self, FitError> {
        // Default implementation can be overridden by specific estimators
        Err(FitError::InvalidInput(
            "Default fit implementation not provided".to_string(),
        ))
    }

    fn predict(&self, X: &DataFrame) -> Result<DataFrame, PredictError> {
        self._predict_impl(&X)
    }

    fn _predict_impl(&self, X: &DataFrame) -> Result<DataFrame, PredictError> {
        // Default implementation can be overridden by specific estimators
        Err(PredictError::NotFitted)
    }
}
