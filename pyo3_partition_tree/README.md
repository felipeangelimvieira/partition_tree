# Partition Tree

A Rust library for working with rules-based data partitioning and decision trees.

## Module Overview

### 📊 `dataframe` - Data Storage and Management
**Purpose**: Provides a heterogeneous DataFrame structure for storing mixed-type data.

**Key Components**:
- `DataFrame<T>`: Main data structure supporting multiple column types
- `Column<T>`: Enum for different data types (F64, I32, Categorical)
- `Categorical<T>`: Dictionary-encoded categorical data for memory efficiency
- Helper functions for sorting numeric values and counting categories

**Use Cases**: Store and manipulate datasets with mixed numeric and categorical columns.

### 🔧 `rules` - Data Filtering Rules
**Purpose**: Defines rule traits and implementations for data filtering and partitioning.

**Key Components**:
- `Rule<T>` trait: Generic interface for all rule types
- `ContinuousInterval`: Rules for numeric ranges with configurable bounds
- `BelongsTo<T>`: Optimized categorical membership rules with O(1) lookups
- `NoneWrapper<R, T>`: Handles null/missing value logic for any rule type

**Use Cases**: Create complex filtering conditions for data partitioning and decision tree construction.

### 🌳 `partition` - Rule Composition and Evaluation
**Purpose**: Combines multiple rules and evaluates them against DataFrames.

**Key Components**:
- `Partition`: Container for multiple typed rules using type erasure
- Rule evaluation with automatic type matching and downcasting
- Support for combining rules with logical AND operations

**Use Cases**: Build complex multi-column filtering conditions and evaluate them efficiently.

### 🏗️ `node` - Tree Structure (Placeholder)
**Purpose**: Foundation for partition tree node implementation.

**Current Status**: Contains basic node structure definitions for future tree construction.

## Architecture Highlights

- **Type Safety**: Extensive use of Rust's generic system for compile-time safety
- **Memory Efficiency**: Shared domains via `Arc<HashSet<T>>` to minimize memory usage
- **Performance**: O(1) membership tests for categorical rules using HashSet
- **Flexibility**: Generic rules work with any `Eq + Hash + Clone` types
- **Error Handling**: Comprehensive error types for DataFrame operations

## Rules

Define rules that can be applied to data.

### BelongsTo Rule

The `BelongsTo<T>` rule is a generic categorical rule that checks if values belong to a specific set. It has been optimized for performance and memory efficiency:

#### Key Features

- **Generic**: Works with any type `T` that implements `Eq + Hash + Clone` (integers, strings, enums, etc.)
- **O(1) Membership Tests**: Uses `HashSet<T>` for constant-time lookups instead of linear Vec searches
- **Memory Efficient**: Domain is shared via `Arc<HashSet<T>>` - no deep copies during split/intersection operations
- **Efficient Operations**: 
  - `split(point)`: Creates left rule with singleton `{point}` and right rule with remaining values
  - `BitAnd` (intersection): Uses efficient set intersection and verifies domains share the same Arc pointer

#### Usage Example

```rust
use std::collections::HashSet;
use std::sync::Arc;
use partition_tree::{BelongsTo, Rule};

// Create a shared domain
let domain = Arc::new((1..=100).collect::<HashSet<i32>>());

// Create rules that share the domain (memory efficient)
let values1: HashSet<i32> = [1, 3, 5, 7].iter().cloned().collect();
let rule1 = BelongsTo::new(values1, Arc::clone(&domain));

let values2: HashSet<i32> = [3, 5, 9, 11].iter().cloned().collect();
let rule2 = BelongsTo::new(values2, Arc::clone(&domain));

// O(1) membership tests
let data = vec![1, 2, 3, 4, 5];
let results = rule1.evaluate(&data); // [true, false, true, false, true]

// Efficient set intersection
let intersection = rule1 & rule2; // Contains {3, 5}

// Efficient splitting (domain is shared, not copied)
let (left, right) = intersection.split(3);
assert!(Arc::ptr_eq(&left.domain, &domain)); // Same memory location
```

#### Performance Benefits

- **Membership checks**: O(1) average vs O(n) for Vec-based implementation
- **Memory usage**: Domain shared across all operations via Arc reference counting
- **Split operations**: No deep copying of domain, just Arc pointer cloning
- **Intersection operations**: Native HashSet intersection vs manual filtering


## Enhancements to do

### 🔧 Complexity Reduction Proposals

#### 1. **Eliminate Type Erasure in Partition Module**
**Problem**: `Partition` uses `HashMap<String, Box<dyn Any>>` requiring unsafe downcasting.
**Solution**: Replace with `PartitionRule<T>` enum wrapping known rule types for type safety.

#### 2. **Simplify DataFrame Column Types** 
**Problem**: `Column<T>` enum creates pattern matching complexity throughout codebase.
**Solution**: Introduce trait-based `ColumnOps` approach or separate numeric/categorical DataFrames.

#### 3. **Reduce Generic Complexity in Rules** ✅ **COMPLETED**
**Problem**: Repeated trait bounds `Eq + Hash + Clone + 'static` scattered everywhere.
**Solution**: Create single trait alias `RuleValue` and use consistently.
**Status**: Implemented - Added `RuleValue` trait alias and updated all modules to use it consistently.

#### 4. **Consolidate Rule Evaluation Logic**
**Problem**: Complex type matching in `evaluate_rule_on_column` method.
**Solution**: Move evaluation logic into rules themselves via common trait methods.

#### 5. **Simplify NoneWrapper Design**
**Problem**: `NoneWrapper<R, T>` adds complexity layer for null handling.
**Solution**: Build null handling directly into base `Rule` trait with `evaluate_optional` method.

#### 6. **Streamline Arc Usage**
**Problem**: Arc usage creates ownership complexity despite memory benefits.
**Solution**: Create helper methods abstracting Arc manipulation and clearer documentation.

#### 7. **Reduce Pattern Matching Complexity**
**Problem**: Extensive pattern matching on column/rule types throughout codebase.
**Solution**: Use visitor pattern or double dispatch, create specialized methods.

#### 8. **Simplify Public API**
**Problem**: API exposes many implementation details.
**Solution**: Create builder patterns, hide Arc management, provide higher-level combined operations.

#### 9. **Consolidate Error Types**
**Problem**: Different error types scattered across modules.
**Solution**: Unified error enum with `thiserror` crate for better error context.

#### 10. **Domain-Specific Builders**
**Problem**: Creating rules/partitions requires understanding implementation details.
**Solution**: Fluent interface builders with sensible defaults for common use cases.


