//! Custom serde for Polars Schema - serialize as Vec<(String, String)>

use polars::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<S>(schema: &Option<Schema>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match schema {
        Some(s) => {
            let pairs: Vec<(String, String)> = s
                .iter()
                .map(|(name, dtype)| (name.to_string(), format!("{:?}", dtype)))
                .collect();
            Some(pairs).serialize(serializer)
        }
        None => None::<Vec<(String, String)>>.serialize(serializer),
    }
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Schema>, D::Error>
where
    D: Deserializer<'de>,
{
    let pairs: Option<Vec<(String, String)>> = Option::deserialize(deserializer)?;
    Ok(pairs.map(|pairs| {
        let mut schema = Schema::default();
        for (name, dtype_str) in pairs {
            // Parse dtype string back to DataType - use Float64 as fallback
            let dtype = match dtype_str.as_str() {
                "Float64" => DataType::Float64,
                "Float32" => DataType::Float32,
                "Int64" => DataType::Int64,
                "Int32" => DataType::Int32,
                "Int16" => DataType::Int16,
                "Int8" => DataType::Int8,
                "UInt64" => DataType::UInt64,
                "UInt32" => DataType::UInt32,
                "UInt16" => DataType::UInt16,
                "UInt8" => DataType::UInt8,
                "Boolean" => DataType::Boolean,
                "String" => DataType::String,
                _ => DataType::Float64, // Fallback for complex types
            };
            schema.insert(PlSmallStr::from_str(&name), dtype);
        }
        schema
    }))
}
