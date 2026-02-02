//! Custom serde for Option<DataFrame> - we only need to preserve the schema (column names and dtypes)
//! We serialize as Option<Vec<(String, String)>> to be bincode-compatible

use polars::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<S>(df: &Option<DataFrame>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match df {
        None => serializer.serialize_none(),
        Some(df) => {
            let schema = df.schema();
            let schema_vec: Vec<(String, String)> = schema
                .iter()
                .map(|(name, dtype)| (name.as_str().to_string(), dtype.to_string()))
                .collect();
            serializer.serialize_some(&schema_vec)
        }
    }
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<DataFrame>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt: Option<Vec<(String, String)>> = Option::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(schema_vec) if schema_vec.is_empty() => Ok(None),
        Some(schema_vec) => {
            let columns: Vec<Column> = schema_vec
                .into_iter()
                .map(|(name, dtype_str)| {
                    let dtype = parse_dtype_from_string(&dtype_str);
                    let series = Series::new_empty(PlSmallStr::from_str(&name), &dtype);
                    series.into()
                })
                .collect();
            Ok(Some(DataFrame::new(columns).expect("DataFrame::new")))
        }
    }
}

fn parse_dtype_from_string(s: &str) -> DataType {
    // Common dtype patterns from polars .to_string()
    match s {
        "f64" | "Float64" => DataType::Float64,
        "f32" | "Float32" => DataType::Float32,
        "i64" | "Int64" => DataType::Int64,
        "i32" | "Int32" => DataType::Int32,
        "i16" | "Int16" => DataType::Int16,
        "i8" | "Int8" => DataType::Int8,
        "u64" | "UInt64" => DataType::UInt64,
        "u32" | "UInt32" => DataType::UInt32,
        "u16" | "UInt16" => DataType::UInt16,
        "u8" | "UInt8" => DataType::UInt8,
        "bool" | "Boolean" => DataType::Boolean,
        "str" | "String" => DataType::String,
        // Categorical types are complex, use String as a surrogate for schema purposes
        _ if s.starts_with("cat") || s.starts_with("Categorical") || s.starts_with("Enum") => {
            DataType::String
        }
        _ => DataType::String, // fallback
    }
}
