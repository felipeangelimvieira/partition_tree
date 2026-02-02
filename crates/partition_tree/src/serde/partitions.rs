//! Custom serde for HashMap<String, Box<dyn DynOneDimPartition>>

use crate::onedimpartition::{DynOneDimPartition, PartitionVariant};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

pub fn serialize<S>(
    partitions: &HashMap<String, Box<dyn DynOneDimPartition>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let variants: HashMap<String, PartitionVariant> = partitions
        .iter()
        .map(|(k, v)| (k.clone(), PartitionVariant::from_dyn(v.as_ref())))
        .collect();
    variants.serialize(serializer)
}

pub fn deserialize<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, Box<dyn DynOneDimPartition>>, D::Error>
where
    D: Deserializer<'de>,
{
    let variants: HashMap<String, PartitionVariant> = HashMap::deserialize(deserializer)?;
    Ok(variants
        .into_iter()
        .map(|(k, v)| (k, v.into_dyn()))
        .collect())
}
