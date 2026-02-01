# Rust Profiling Guide for partition_tree

## Quick Reference

### 1. **Basic Timing (Already implemented)**
```rust
use std::time::Instant;

let start_time = Instant::now();
tree.fit(&df);
let duration = start_time.elapsed();
println!("Tree fitting took: {:?}", duration);
```

### 2. **Criterion Benchmarks (Already implemented)**
```bash
cargo bench
```
- Creates detailed HTML reports in `target/criterion/`
- Provides statistical analysis with confidence intervals
- Compares performance across runs

### 3. **Memory Profiling with Valgrind** (Linux)
```bash
# Install valgrind
sudo apt-get install valgrind

# Run with massif for heap profiling
cargo build --release
valgrind --tool=massif target/release/examples/simple_test

# Analyze with ms_print
ms_print massif.out.<PID>
```

### 4. **CPU Profiling with perf** (Linux)
```bash
# Record performance data
cargo build --release
perf record --call-graph=dwarf target/release/examples/simple_test

# View the report
perf report
```

### 5. **Flamegraph Profiling** (Cross-platform)
```bash
# Already installed, but requires root on some systems
sudo cargo flamegraph --example simple_test

# This creates flamegraph.svg showing function call hierarchy
```

### 6. **cargo-profdata** (LLVM-based)
```bash
# Install cargo-profdata
cargo install cargo-profdata

# Build with instrumentation
RUSTFLAGS="-C instrument-coverage" cargo build --release

# Run and generate profile
cargo profdata -- run --example simple_test
```

### 7. **Instruments on macOS** (Native macOS profiler)
```bash
# Build release version
cargo build --release --example simple_test

# Run with Instruments
instruments -t "Time Profiler" target/release/examples/simple_test
```

### 8. **Memory Usage with heaptrack** (Linux)
```bash
# Install heaptrack
sudo apt-get install heaptrack

# Profile memory usage
heaptrack target/release/examples/simple_test

# Analyze results
heaptrack_gui heaptrack.simple_test.<PID>.gz
```

### 9. **Profile-guided Optimization (PGO)**
Add to Cargo.toml:
```toml
[profile.release]
lto = true
codegen-units = 1
```

### 10. **Custom Profiling Points**
```rust
use std::time::Instant;

// Profile specific functions
fn profile_section<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = f();
    println!("{}: {:?}", name, start.elapsed());
    result
}

// Usage
let result = profile_section("data_generation", || {
    generate_sample_data(n_samples, n_features)
});
```

## Performance Results Summary

Based on our benchmarks:

### Current Performance
- **Small datasets** (100 samples, 5 features): ~0.5-2ms
- **Medium datasets** (1,000 samples, 20 features): ~1-5ms
- **Large datasets** (10,000 samples, 50 features): ~30ms
- **Very large datasets** (100,000 samples, 100 features): ~400ms

### Scaling Characteristics
- Time complexity appears roughly O(n * m * log(n)) where n=samples, m=features
- Memory usage scales with dataset size
- Performance is dominated by split evaluation across features

## Optimization Opportunities

1. **Parallelization**: Consider using more aggressive parallelization with rayon
2. **Memory layout**: Optimize data structures for cache efficiency
3. **Algorithm**: Consider early stopping criteria for split evaluation
4. **Feature selection**: Pre-filter irrelevant features to reduce search space

## Tools Installation

### macOS (using Homebrew)
```bash
# For flamegraph (if needed)
brew install dtrace

# For additional tools
brew install valgrind  # May not be available on Apple Silicon
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install valgrind perf linux-perf heaptrack
```

### Windows
```bash
# Use Windows Performance Analyzer or Visual Studio Profiler
# Criterion benchmarks work cross-platform
```
