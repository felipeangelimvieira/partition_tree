pub const TARGET_PREFIX: &str = "target";
pub const MAX_CANDIDATE_SPLIT_POINTS: usize = usize::MAX;
#[derive(PartialEq, Debug, Clone)]
pub enum TargetBehaviour {
    Include,
    Exclude,
    Only,
}

impl TargetBehaviour {
    pub fn includes(&self, col_name: &str) -> bool {
        match self {
            TargetBehaviour::Include => true,
            TargetBehaviour::Exclude => !col_name.starts_with(TARGET_PREFIX),
            TargetBehaviour::Only => col_name.starts_with(TARGET_PREFIX),
        }
    }
}
