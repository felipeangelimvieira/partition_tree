//! Custom serde for `HashMap<String, Box<dyn DynRule>>`.
//!
//! Uses [`RuleType`](crate::rules::RuleType) as a serialization surrogate.
//! Each concrete `DynRule` implementation (`ContinuousInterval`, `BelongsTo`,
//! `IntegerInterval`) is downcast via `as_any()`, wrapped in the
//! corresponding `RuleType` variant, and serialized. Deserialization
//! reconstitutes the concrete type and boxes it.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::rule::DynRule;
use crate::rules::{BelongsTo, ContinuousInterval, IntegerInterval, RuleType};

/// Convert a `&dyn DynRule` into a serializable `RuleType`.
fn to_variant(rule: &dyn DynRule) -> RuleType {
    if let Some(ci) = rule.as_any().downcast_ref::<ContinuousInterval>() {
        return RuleType::Continuous(ci.clone());
    }
    if let Some(bt) = rule.as_any().downcast_ref::<BelongsTo>() {
        return RuleType::BelongsTo(bt.clone());
    }
    if let Some(ii) = rule.as_any().downcast_ref::<IntegerInterval>() {
        return RuleType::Integer(ii.clone());
    }
    panic!(
        "Unknown DynRule concrete type for serialization: {:?}",
        rule
    );
}

/// Convert a `RuleType` back into a `Box<dyn DynRule>`.
fn from_variant(variant: RuleType) -> Box<dyn DynRule> {
    match variant {
        RuleType::Continuous(ci) => Box::new(ci),
        RuleType::BelongsTo(bt) => Box::new(bt),
        RuleType::Integer(ii) => Box::new(ii),
    }
}

pub fn serialize<S>(
    rules: &HashMap<String, Box<dyn DynRule>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let variants: HashMap<&str, RuleType> = rules
        .iter()
        .map(|(k, v)| (k.as_str(), to_variant(v.as_ref())))
        .collect();
    variants.serialize(serializer)
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<String, Box<dyn DynRule>>, D::Error>
where
    D: Deserializer<'de>,
{
    let variants: HashMap<String, RuleType> = HashMap::deserialize(deserializer)?;
    Ok(variants
        .into_iter()
        .map(|(k, v)| (k, from_variant(v)))
        .collect())
}
