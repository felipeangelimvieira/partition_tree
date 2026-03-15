//! Cross-process determinism test.
//!
//! Spawns the `determinism_check` example binary twice in separate
//! processes and asserts that the output is byte-identical. This catches
//! HashMap-iteration-order non-determinism that only manifests across
//! process boundaries (within a single process the RandomState seed is fixed).

use std::process::Command;

#[test]
fn forest_predictions_are_deterministic_across_processes() {
    // Build the example binary first (shared compilation).
    let build = Command::new("cargo")
        .args(["build", "--example", "determinism_check", "--quiet"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("failed to build determinism_check example");
    assert!(build.success(), "cargo build for example failed");

    let run = || {
        let output = Command::new("cargo")
            .args(["run", "--example", "determinism_check", "--quiet"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("failed to run determinism_check");
        assert!(
            output.status.success(),
            "determinism_check exited with error: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("non-utf8 output")
    };

    let out1 = run();
    let out2 = run();

    assert!(!out1.is_empty(), "determinism_check produced no output");

    if out1 != out2 {
        // Find the first differing line for a useful error message
        let lines1: Vec<&str> = out1.lines().collect();
        let lines2: Vec<&str> = out2.lines().collect();
        let first_diff = lines1
            .iter()
            .zip(lines2.iter())
            .enumerate()
            .find(|(_, (a, b))| a != b);
        if let Some((i, (a, b))) = first_diff {
            panic!(
                "Non-determinism detected at line {i}:\n  run1: {a}\n  run2: {b}\n\
                 Total lines: run1={}, run2={}",
                lines1.len(),
                lines2.len()
            );
        } else {
            panic!(
                "Non-determinism: outputs differ in length (run1={}, run2={})",
                lines1.len(),
                lines2.len()
            );
        }
    }
}
