//! Test to verify that training samples always have non-zero PDF.
//!
//! When predicting probability distributions for training samples,
//! each sample's Y value must fall within at least one of the returned
//! intervals. If there are gaps in the interval coverage, training
//! samples can have zero PDF which is incorrect.

use partition_tree::tree::Tree;
use polars::prelude::*;
use rand::SeedableRng;

fn build_tree(max_iter: usize, max_depth: usize, min_samples_leaf: usize, seed: usize) -> Tree {
    Tree::new(
        max_iter,
        1,                // min_samples_split
        0,                // min_samples_leaf_y
        0,                // min_samples_leaf_x
        min_samples_leaf, // min_samples_leaf
        max_depth,
        0.0,           // min_target_volume
        0.0,           // min_split_gain
        0.0,           // min_density_value
        f64::INFINITY, // max_density_value
        f64::INFINITY, // max_measure_value
        1.0,           // boundaries_expansion_factor
        None,          // max_samples
        None,          // max_features
        0,             // exploration_split_budget
        None,          // feature_split_fraction
        Some(seed),    // seed
    )
}

/// Check if a value falls within an interval considering open/closed boundaries.
fn value_in_interval(
    value: f64,
    low: f64,
    high: f64,
    lower_closed: bool,
    upper_closed: bool,
) -> bool {
    let lower_ok = if lower_closed {
        value >= low
    } else {
        value > low
    };
    let upper_ok = if upper_closed {
        value <= high
    } else {
        value < high
    };
    lower_ok && upper_ok
}

/// Test that all training samples have their Y value covered by at least one interval.
/// This test uses a simple synthetic dataset.
#[test]
fn training_samples_covered_by_intervals_simple() {
    // Create a simple dataset with clear structure
    let n_samples = 50;
    let x_values: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();
    let y_values: Vec<f64> = x_values.iter().map(|&x| x * 2.0 + 10.0).collect();

    let x = Series::new(PlSmallStr::from_static("x"), &x_values);
    let target = Series::new(PlSmallStr::from_static("target_y"), &y_values);

    let df = DataFrame::new(vec![x.into(), target.into()]).unwrap();

    let mut tree = build_tree(20, 5, 1, 42);
    tree.fit(&df, None);

    // Get probability distributions for all training samples
    let probas = tree.predict_proba(&df);

    // For each training sample, check that its Y value is covered
    for (sample_idx, dist) in probas.iter().enumerate() {
        let y_value = y_values[sample_idx];
        let intervals = dist.pdf_with_intervals();

        let mut covered = false;
        for (_pdf_val, (low, high, lower_closed, upper_closed)) in &intervals {
            if value_in_interval(y_value, *low, *high, *lower_closed, *upper_closed) {
                covered = true;
                break;
            }
        }

        assert!(
            covered,
            "Sample {} with Y={} is not covered by any interval. Intervals: {:?}",
            sample_idx, y_value, intervals
        );
    }
}

/// Test with a more complex dataset that has varied X and Y relationships.
/// This is closer to real-world scenarios where gaps might appear.
#[test]
fn training_samples_covered_by_intervals_complex() {
    // Create dataset with multiple features and varied relationships
    let n_samples = 100;

    // Use a deterministic pattern
    let x1_values: Vec<f64> = (0..n_samples).map(|i| (i % 10) as f64).collect();
    let x2_values: Vec<f64> = (0..n_samples).map(|i| (i / 10) as f64).collect();
    let y_values: Vec<f64> = (0..n_samples)
        .map(|i| {
            let x1 = (i % 10) as f64;
            let x2 = (i / 10) as f64;
            x1 * 10.0 + x2 * 5.0 + 50.0
        })
        .collect();

    let x1 = Series::new(PlSmallStr::from_static("x1"), &x1_values);
    let x2 = Series::new(PlSmallStr::from_static("x2"), &x2_values);
    let target = Series::new(PlSmallStr::from_static("target_y"), &y_values);

    let df = DataFrame::new(vec![x1.into(), x2.into(), target.into()]).unwrap();

    let mut tree = build_tree(50, 10, 1, 23);
    tree.fit(&df, None);

    let probas = tree.predict_proba(&df);

    let mut uncovered_samples = Vec::new();

    for (sample_idx, dist) in probas.iter().enumerate() {
        let y_value = y_values[sample_idx];
        let intervals = dist.pdf_with_intervals();

        let mut covered = false;
        for (_pdf_val, (low, high, lower_closed, upper_closed)) in &intervals {
            if value_in_interval(y_value, *low, *high, *lower_closed, *upper_closed) {
                covered = true;
                break;
            }
        }

        if !covered {
            uncovered_samples.push((sample_idx, y_value));
        }
    }

    assert!(
        uncovered_samples.is_empty(),
        "Found {} training samples not covered by any interval: {:?}",
        uncovered_samples.len(),
        &uncovered_samples[..std::cmp::min(10, uncovered_samples.len())]
    );
}

/// Test that verifies interval coverage is contiguous (no gaps).
/// For each sample's distribution, the intervals should cover a contiguous range.
#[test]
fn intervals_are_contiguous_no_gaps() {
    let n_samples = 50;
    let x_values: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();
    let y_values: Vec<f64> = x_values.iter().map(|&x| x * 2.0 + 10.0).collect();

    let x = Series::new(PlSmallStr::from_static("x"), &x_values);
    let target = Series::new(PlSmallStr::from_static("target_y"), &y_values);

    let df = DataFrame::new(vec![x.into(), target.into()]).unwrap();

    let mut tree = build_tree(20, 5, 1, 42);
    tree.fit(&df, None);

    let probas = tree.predict_proba(&df);

    for (sample_idx, dist) in probas.iter().enumerate() {
        let mut intervals: Vec<(f64, f64, bool, bool)> = dist
            .pdf_with_intervals()
            .iter()
            .map(|(_, bounds)| *bounds)
            .collect();

        // Sort by lower bound
        intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Check for gaps between consecutive intervals
        let mut gaps = Vec::new();
        for i in 0..intervals.len().saturating_sub(1) {
            let (_, high_i, _, upper_closed_i) = intervals[i];
            let (low_next, _, lower_closed_next, _) = intervals[i + 1];

            // There's a gap if:
            // - The intervals don't touch (high_i < low_next)
            // - OR they touch but both boundaries exclude the point
            let has_gap = if (high_i - low_next).abs() < 1e-10 {
                // They touch at the same point
                // Gap exists if both exclude the point: (..., X) and (X, ...)
                !upper_closed_i && !lower_closed_next
            } else {
                // They don't touch - gap exists if high_i < low_next
                high_i < low_next - 1e-10
            };

            if has_gap {
                gaps.push((high_i, low_next));
            }
        }

        assert!(
            gaps.is_empty(),
            "Sample {} has gaps in interval coverage: {:?}",
            sample_idx,
            gaps
        );
    }
}

/// Reproduce the specific bug: training sample Y value falls in a gap.
/// This uses parameters similar to the failing Python test.
#[test]
fn no_training_sample_in_gap() {
    // Create a dataset similar to the diabetes dataset structure
    // Multiple features, continuous target with varied range
    let n_samples = 100;

    // Create deterministic pseudo-random-like data
    let mut x1_values = Vec::with_capacity(n_samples);
    let mut x2_values = Vec::with_capacity(n_samples);
    let mut x3_values = Vec::with_capacity(n_samples);
    let mut y_values = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Simulate varied feature values
        let x1 = ((i * 7) % 100) as f64 / 10.0;
        let x2 = ((i * 13) % 100) as f64 / 10.0;
        let x3 = ((i * 17) % 100) as f64 / 10.0;

        // Target with relationship to features + variation
        let y = 50.0 + x1 * 10.0 + x2 * 5.0 + ((i * 23) % 50) as f64;

        x1_values.push(x1);
        x2_values.push(x2);
        x3_values.push(x3);
        y_values.push(y);
    }

    let x1 = Series::new(PlSmallStr::from_static("x1"), &x1_values);
    let x2 = Series::new(PlSmallStr::from_static("x2"), &x2_values);
    let x3 = Series::new(PlSmallStr::from_static("x3"), &x3_values);
    let target = Series::new(PlSmallStr::from_static("target_y"), &y_values);

    let df = DataFrame::new(vec![x1.into(), x2.into(), x3.into(), target.into()]).unwrap();

    // Use parameters similar to the failing Python case
    let mut tree = build_tree(50, 10, 1, 23);
    tree.fit(&df, None);

    // Get which leaves each sample matches (X-only matching)
    let matches_per_row = tree.match_leaves_given_x(&df);
    let probas = tree.predict_proba(&df);

    // Collect all samples that fall in gaps
    let mut samples_in_gaps: Vec<(usize, f64, Vec<(f64, f64)>, usize)> = Vec::new();

    for (sample_idx, dist) in probas.iter().enumerate() {
        let y_value = y_values[sample_idx];
        let intervals = dist.pdf_with_intervals();
        let num_matched_leaves = matches_per_row[sample_idx].len();

        // Check if y_value is covered
        let covered = intervals
            .iter()
            .any(|(_, (low, high, lc, uc))| value_in_interval(y_value, *low, *high, *lc, *uc));

        if !covered {
            // Find which gap it falls into
            let mut sorted_intervals: Vec<(f64, f64)> =
                intervals.iter().map(|(_, (l, h, _, _))| (*l, *h)).collect();
            sorted_intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Find gaps near this y_value
            let nearby_intervals: Vec<(f64, f64)> = sorted_intervals
                .iter()
                .filter(|(l, h)| *l <= y_value + 20.0 && *h >= y_value - 20.0)
                .cloned()
                .collect();

            samples_in_gaps.push((sample_idx, y_value, nearby_intervals, num_matched_leaves));
        }
    }

    if !samples_in_gaps.is_empty() {
        eprintln!("Found {} samples in gaps:", samples_in_gaps.len());
        for (idx, y, nearby, num_leaves) in samples_in_gaps.iter().take(5) {
            eprintln!(
                "  Sample {}: y={}, matched {} leaves, nearby intervals: {:?}",
                idx, y, num_leaves, nearby
            );
        }

        // For the first problematic sample, show detailed leaf info
        if let Some((idx, y, _, _)) = samples_in_gaps.first() {
            eprintln!("\nDetailed analysis for sample {}:", idx);
            let matched_leaf_positions = &matches_per_row[*idx];
            eprintln!("  Matched leaf positions: {:?}", matched_leaf_positions);
            eprintln!("  Sample Y value: {}", y);

            // Show all intervals from matched leaves
            let dist = &probas[*idx];
            let intervals = dist.pdf_with_intervals();
            let mut sorted: Vec<_> = intervals.iter().map(|(_, b)| b).collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            eprintln!("  All Y intervals from matched leaves:");
            for (low, high, lc, uc) in sorted.iter() {
                let bracket_l = if *lc { "[" } else { "(" };
                let bracket_r = if *uc { "]" } else { ")" };
                eprintln!("    {}{:.2}, {:.2}{}", bracket_l, low, high, bracket_r);
            }

            // Find which leaf SHOULD contain this sample (match using both X and Y)
            eprintln!(
                "\n  Looking for leaf that contains sample {} with Y={}:",
                idx, y
            );
            let all_leaves = tree.get_leaves();

            // First, check which leaf has sample 12 in its indices_xy
            eprintln!(
                "  Checking which leaf contains sample {} in indices_xy:",
                idx
            );
            let mut found_in_any_leaf = false;
            for (leaf_pos, &leaf_node_idx) in all_leaves.iter().enumerate() {
                let node = tree.get_node(leaf_node_idx).unwrap();
                let contains_sample = node.indices_xy.iter().any(|opt| opt == Some(*idx as u32));
                if contains_sample {
                    found_in_any_leaf = true;
                    eprintln!(
                        "    Leaf {} (node {}) has sample {} in indices_xy",
                        leaf_pos, leaf_node_idx, idx
                    );
                    let was_matched = matched_leaf_positions.contains(&leaf_pos);
                    eprintln!("    Was matched by X-only? {}", was_matched);
                    if !was_matched {
                        eprintln!(
                            "    BUG: Leaf has sample in indices_xy but X-only didn't match!"
                        );
                    }
                }
            }

            if !found_in_any_leaf {
                eprintln!("  BUG: Sample {} is not in ANY leaf's indices_xy!", idx);

                // Check if it's in the root node
                let root = tree.get_node(0).unwrap();
                let in_root = root.indices_xy.iter().any(|opt| opt == Some(*idx as u32));
                eprintln!("  Is sample {} in root's indices_xy? {}", idx, in_root);
                eprintln!("  Root indices_xy length: {}", root.indices_xy.len());

                // Check total samples across all leaves
                let total_in_leaves: usize = all_leaves
                    .iter()
                    .map(|&leaf_idx| tree.get_node(leaf_idx).unwrap().indices_xy.len())
                    .sum();
                eprintln!("  Total samples across all leaves: {}", total_in_leaves);
                eprintln!("  Number of leaves: {}", all_leaves.len());

                // Check ALL nodes to see which ones have this sample
                eprintln!("\n  Checking ALL nodes for sample {}:", idx);
                let num_nodes = tree.num_nodes();
                let mut nodes_with_sample = Vec::new();
                for node_idx in 0..num_nodes {
                    let node = tree.get_node(node_idx).unwrap();
                    let has_sample = node.indices_xy.iter().any(|opt| opt == Some(*idx as u32));
                    if has_sample {
                        let is_leaf = all_leaves.contains(&node_idx);
                        nodes_with_sample.push((node_idx, is_leaf, node.indices_xy.len()));
                    }
                }
                eprintln!("    Nodes containing sample: {:?}", nodes_with_sample);
                // Check which node has the sample but ISN'T a leaf
                // This is the "lost" node
                let lost_nodes: Vec<_> = nodes_with_sample
                    .iter()
                    .filter(|(_, is_leaf, _)| !is_leaf)
                    .collect();
                if !lost_nodes.is_empty() {
                    eprintln!(
                        "\n  CRITICAL: Sample is in non-leaf nodes: {:?}",
                        lost_nodes
                    );

                    // Find the deepest non-leaf node
                    let deepest = lost_nodes.iter().max_by_key(|(node_idx, _, _)| node_idx);
                    if let Some((node_idx, _, _)) = deepest {
                        eprintln!("  Deepest non-leaf node containing sample: {}", node_idx);

                        // Check leaf_reasons for this node
                        let leaf_reasons = tree.get_leaf_reasons();
                        if let Some(reason) = leaf_reasons.get(node_idx) {
                            eprintln!("  This node has a leaf reason: {:?}", reason);
                        } else {
                            eprintln!(
                                "  This node has NO leaf reason - it was supposed to be split but wasn't!"
                            );
                        }

                        // Check if this node was ever a parent in split_history
                        let split_history = tree.get_split_history();
                        let was_parent = split_history.iter().any(|r| r.parent_index == *node_idx);
                        eprintln!("  Was this node ever a parent in splits? {}", was_parent);

                        // If not a parent and not a leaf, this is the bug!
                        if !was_parent && !all_leaves.contains(node_idx) {
                            eprintln!(
                                "  BUG CONFIRMED: Node {} was never split AND is not a leaf!",
                                node_idx
                            );
                        }
                    }
                }
                // Find where this sample was lost - trace through split history
                eprintln!("\n  Tracing through tree to find where sample was lost...");
                let split_history = tree.get_split_history();
                eprintln!("  Split history has {} records", split_history.len());

                // Track the sample through splits
                for (i, record) in split_history.iter().enumerate() {
                    let parent_node = tree.get_node(record.parent_index).unwrap();
                    let in_parent = parent_node
                        .indices_xy
                        .iter()
                        .any(|opt| opt == Some(*idx as u32));

                    if in_parent {
                        let left_node = tree.get_node(record.left_child_index).unwrap();
                        let right_node = tree.get_node(record.right_child_index).unwrap();
                        let in_left = left_node
                            .indices_xy
                            .iter()
                            .any(|opt| opt == Some(*idx as u32));
                        let in_right = right_node
                            .indices_xy
                            .iter()
                            .any(|opt| opt == Some(*idx as u32));

                        eprintln!(
                            "  Split {}: parent={}, col={}, left={} (has={}), right={} (has={})",
                            i,
                            record.parent_index,
                            record.column_name,
                            record.left_child_index,
                            in_left,
                            record.right_child_index,
                            in_right
                        );

                        if !in_left && !in_right {
                            eprintln!("  FOUND IT! Sample {} lost at split {}:", idx, i);
                            eprintln!("    Split result: {:?}", record.split_result);

                            // Check the sample's actual value in the split column
                            let col = df.column(&record.column_name).unwrap();
                            let val = col.get(*idx).unwrap();
                            eprintln!("    Sample's value in {}: {:?}", record.column_name, val);
                            break;
                        }
                    }
                }
            }

            for (leaf_pos, &leaf_node_idx) in all_leaves.iter().enumerate() {
                // Check if this leaf matches the sample using BOTH X and Y
                let node = tree.get_node(leaf_node_idx).unwrap();
                let sample_df = df.slice(*idx as i64, 1);
                let idx_range: std::ops::Range<u32> = 0u32..1u32;
                let subset = UInt32Chunked::new(PlSmallStr::from_static("idx"), idx_range);

                // Match including target
                use partition_tree::conf::TargetBehaviour;
                let match_xy = node
                    .cell
                    .match_dataframe(&sample_df, &subset, TargetBehaviour::Include)
                    .expect("match xy");

                if match_xy.len() > 0 {
                    // This leaf matches both X and Y
                    eprintln!(
                        "    Leaf {} (node {}) matches BOTH X and Y!",
                        leaf_pos, leaf_node_idx
                    );

                    // Check if this leaf was in the X-only match list
                    let was_matched = matched_leaf_positions.contains(&leaf_pos);
                    eprintln!("    Was this leaf in X-only match list? {}", was_matched);

                    if !was_matched {
                        eprintln!("    BUG: This leaf should have been matched by X but wasn't!");

                        // Check X-only matching
                        let match_x = node
                            .cell
                            .match_dataframe(&sample_df, &subset, TargetBehaviour::Exclude)
                            .expect("match x");
                        eprintln!("    X-only match result length: {}", match_x.len());
                    }
                }
            }
        }
    }

    assert!(
        samples_in_gaps.is_empty(),
        "Found {} training samples that fall in interval gaps. \
         This indicates a bug in the tree building or leaf matching logic.",
        samples_in_gaps.len()
    );
}

/// Test that after training, no leaf has zero mass (n_xy = 0).
/// Every leaf should have at least one training sample matching both X and Y constraints.
/// If a leaf has zero mass, it means it was created incorrectly during tree building.
#[test]
fn no_leaf_has_zero_mass_after_training() {
    // Use the same configuration as the Python code that reproduces the issue
    let n_samples = 353; // Similar to diabetes train split

    // Create a dataset similar to diabetes
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    use rand::Rng;

    // 10 features like diabetes
    let x0: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x1: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x2: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x3: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x4: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x5: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x6: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x7: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x8: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x9: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(-0.1..0.1)).collect();

    // Target in range similar to diabetes (25-346)
    let target: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(25.0..350.0)).collect();

    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from_static("x0"), &x0).into(),
        Series::new(PlSmallStr::from_static("x1"), &x1).into(),
        Series::new(PlSmallStr::from_static("x2"), &x2).into(),
        Series::new(PlSmallStr::from_static("x3"), &x3).into(),
        Series::new(PlSmallStr::from_static("x4"), &x4).into(),
        Series::new(PlSmallStr::from_static("x5"), &x5).into(),
        Series::new(PlSmallStr::from_static("x6"), &x6).into(),
        Series::new(PlSmallStr::from_static("x7"), &x7).into(),
        Series::new(PlSmallStr::from_static("x8"), &x8).into(),
        Series::new(PlSmallStr::from_static("x9"), &x9).into(),
        Series::new(PlSmallStr::from_static("target_0"), &target).into(),
    ])
    .unwrap();

    // Use same hyperparameters as the Python issue
    let mut tree = Tree::new(
        100,  // max_iter
        1,    // min_samples_split
        0,    // min_samples_leaf_y
        10,   // min_samples_leaf_x
        1,    // min_samples_leaf
        21,   // max_depth
        0.05, // min_target_volume
        0.0,  // min_split_gain
        0.0,  // min_density_value
        f64::INFINITY,
        f64::INFINITY,
        1.0, // boundaries_expansion_factor
        None,
        None,
        0,
        None,
        Some(23), // seed
    );

    tree.fit(&df, None);

    // Check every leaf has non-zero mass
    let leaves = tree.get_leaves();
    let mut zero_mass_leaves = Vec::new();

    for &leaf_idx in leaves {
        let node = tree.get_node(leaf_idx).expect("leaf node should exist");
        let n_xy = node.indices_xy.len();
        let n_x = node.indices_x.len();
        let mass = node.conditional_measure();

        if n_xy == 0 {
            zero_mass_leaves.push((leaf_idx, n_xy, n_x, mass));
        }
    }

    if !zero_mass_leaves.is_empty() {
        eprintln!(
            "\n=== FOUND {} LEAVES WITH ZERO MASS ===",
            zero_mass_leaves.len()
        );
        for (leaf_idx, n_xy, n_x, mass) in &zero_mass_leaves {
            eprintln!(
                "Leaf {}: n_xy={}, n_x={}, mass={}",
                leaf_idx, n_xy, n_x, mass
            );

            // Get the leaf reason
            if let Some(reason) = tree.get_leaf_reasons().get(leaf_idx) {
                eprintln!("  Reason: {:?}", reason);
            }

            // Print the cell info
            let node = tree.get_node(*leaf_idx).unwrap();
            eprintln!("  n_y={}", node.indices_y.len());
        }
    }

    assert!(
        zero_mass_leaves.is_empty(),
        "Found {} leaves with zero mass (n_xy=0). \
         This is a bug - every leaf should have at least one training sample. \
         Leaf indices: {:?}",
        zero_mass_leaves.len(),
        zero_mass_leaves
            .iter()
            .map(|(idx, _, _, _)| idx)
            .collect::<Vec<_>>()
    );
}

/// Test with real diabetes-like data loading pattern.
/// Uses deterministic data but with similar statistical properties.
#[test]
fn no_zero_mass_leaves_diabetes_config() {
    // Create 10 features with correlations similar to diabetes
    let n = 353;
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    use rand::Rng;

    // Generate features with some correlation structure
    let base: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();

    let x0: Vec<f64> = base.iter().map(|&b| b + rng.gen_range(-0.5..0.5)).collect();
    let x1: Vec<f64> = base
        .iter()
        .map(|&b| -b + rng.gen_range(-0.5..0.5))
        .collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x3: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x4: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x5: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x6: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x7: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x8: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let x9: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();

    // Target with relationship to x0 and x1
    let target: Vec<f64> = (0..n)
        .map(|i| 150.0 + 50.0 * x0[i] + 30.0 * x1[i] + rng.gen_range(-50.0..50.0))
        .collect();

    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from_static("x0"), &x0).into(),
        Series::new(PlSmallStr::from_static("x1"), &x1).into(),
        Series::new(PlSmallStr::from_static("x2"), &x2).into(),
        Series::new(PlSmallStr::from_static("x3"), &x3).into(),
        Series::new(PlSmallStr::from_static("x4"), &x4).into(),
        Series::new(PlSmallStr::from_static("x5"), &x5).into(),
        Series::new(PlSmallStr::from_static("x6"), &x6).into(),
        Series::new(PlSmallStr::from_static("x7"), &x7).into(),
        Series::new(PlSmallStr::from_static("x8"), &x8).into(),
        Series::new(PlSmallStr::from_static("x9"), &x9).into(),
        Series::new(PlSmallStr::from_static("target_0"), &target).into(),
    ])
    .unwrap();

    let mut tree = Tree::new(
        100,
        1,
        0,
        10,
        1,
        21,
        0.05,
        0.0,
        0.0,
        f64::INFINITY,
        f64::INFINITY,
        1.0,
        None,
        None,
        0,
        None,
        Some(23),
    );

    tree.fit(&df, None);

    let leaves = tree.get_leaves();
    let mut zero_mass_count = 0;

    for &leaf_idx in leaves {
        let node = tree.get_node(leaf_idx).unwrap();
        if node.indices_xy.len() == 0 {
            zero_mass_count += 1;
            eprintln!("\n=== Zero-mass leaf {} ===", leaf_idx);
            eprintln!(
                "n_xy={}, n_x={}, n_y={}, reason={:?}",
                node.indices_xy.len(),
                node.indices_x.len(),
                node.indices_y.len(),
                tree.get_leaf_reasons().get(&leaf_idx)
            );

            // Find which split created this leaf
            for record in tree.get_split_history() {
                if record.left_child_index == leaf_idx || record.right_child_index == leaf_idx {
                    eprintln!(
                        "Created by split: parent={}, column={}, result={:?}",
                        record.parent_index, record.column_name, record.split_result
                    );

                    // Get parent info
                    if let Some(parent) = tree.get_node(record.parent_index) {
                        eprintln!(
                            "Parent: n_xy={}, n_x={}, n_y={}",
                            parent.indices_xy.len(),
                            parent.indices_x.len(),
                            parent.indices_y.len()
                        );
                    }

                    // Get sibling info
                    let sibling_idx = if record.left_child_index == leaf_idx {
                        record.right_child_index
                    } else {
                        record.left_child_index
                    };
                    if let Some(sibling) = tree.get_node(sibling_idx) {
                        eprintln!(
                            "Sibling {}: n_xy={}, n_x={}, n_y={}",
                            sibling_idx,
                            sibling.indices_xy.len(),
                            sibling.indices_x.len(),
                            sibling.indices_y.len()
                        );
                    }
                }
            }
        }
    }

    assert_eq!(
        zero_mass_count, 0,
        "Found {} leaves with zero mass (n_xy=0)",
        zero_mass_count
    );
}
