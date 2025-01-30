
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
    |    |-- analyzer.rs
    |
    |-- cargo.toml
    |-- readme.md

---
**Instruction**
1. install rust
    - recommended higher version 1.84.0 for iced.rs(version 0.13.1)
2. cargo build
3. cargo run
---