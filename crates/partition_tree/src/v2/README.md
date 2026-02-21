This file is meant to be a refactoring of Partition Tree
The goal is to make the code more modular and easier to extend,
keeping clear responsabilities for each component.

The main components are:
//
- GainObjective: the struct that represents the gain objective function
- SplitSearcher trait: the trait that represents a split search contract, 
  with different implementations for different variable types (numerical, categorical, etc.)
- Tree Builder: the struct that represents the tree building process, which will use the SplitSearcher and GainObjective to build the tree
- Rule: as/is, the contract of rules that will be used to represent the splits in the tree
- Cell: rules and their corresponding column names
Tree: the main struct that represents the Partition Tree
 