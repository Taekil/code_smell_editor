
**Code Smell Detector**
---
**Purpose of this readme.md**
- this readme.md provides the information
    - direction to running(?)
    - frameworks
    - additional tech(?)
---
**Introduction**
- Code Smell Detector design to find simple code smells,
    - function1: long parameter
    - function2: long lines
    - function3: duplicated code
---
**Library and frameworks**
- the list of frameworks are listed in cargo.toml as Dependencies
    - iced version 
    - rfd version
- optional figures should be commanted, do not forget it. 
---
**File Structure**

    project folder
    |
    |-- src
    |    |-- main.rs
    |    |-- CodeAnalyzer.rs
    |    |-- CodeFlow.rs
    |    |-- tokenizer.rs
    |    |-- abstractSyntax.rs
    |    |-- fileManager.rs 
    |
    |-- cargo.toml
    |-- readme.md

main -> CodeAnalyzer -> tokenizer: Create Token -> CodeFlow: Create CFG -> return CFG to CodeAnalyzer and do Analysis 
                                                -> abstractSyntax: Create AST -> back to Code Analyzer and do Analysis for Code Structure


Module Responsibilities & Code Smell Detection
1. main.rs (Entry Point)

Calls fileManager.rs to read the source code.
Calls tokenizer.rs to tokenize the code.
Calls abstractSyntax.rs to build the AST.
Calls CodeFlow.rs to generate the CFG.
Passes CFG & AST to CodeAnalyzer.rs for detecting code smells.
Displays results.
2. fileManager.rs (File Handling)

Reads .rs source files from a directory.
Provides an interface to retrieve code content as a string.
3. tokenizer.rs (Lexical Analysis)

Breaks the code into tokens.
Helps identify long function names.
Extracts function signatures for further analysis.
4. abstractSyntax.rs (AST Generation & Structural Analysis)

Builds the AST for code structure analysis.
Detects:
Long Function Names (checks function declaration nodes).
Long Parameter List (counts parameters in function declarations).
Duplicated Code (finds structurally similar AST subtrees).
Semantic Duplicated Code (extra credit, compares AST for logic similarity).
5. CodeFlow.rs (CFG Analysis for Flow-based Smells)

Builds the CFG for control flow analysis.
Detects dead code (e.g., unreachable functions, infinite loops).
(For Extra Credit) Can help identify semantic duplicate code by comparing execution paths.
6. CodeAnalyzer.rs (Main Code Smell Detector)

Receives AST & CFG and performs code smell analysis:
Long Function Names: Extract from AST and apply length threshold.
Long Parameter Lists: Extract parameter count from AST.
Duplicated Code: Compare AST subtrees to detect identical code blocks.
Semantic Duplicated Code (Extra Credit):
Compare AST patterns for functions with similar logic.
Use CFG paths to find logically similar but differently structured functions.

---
**Instruction**
1. install rust
    - recommended higher version 1.84.0 for iced.rs(version 0.13.1)
2. cargo build
3. cargo run
---
**Future Improvement**

---

