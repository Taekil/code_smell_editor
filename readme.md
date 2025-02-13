# **Code Smell Detector**  

## **Purpose of this README.md**  
This `README.md` provides essential information about the **Code Smell Detector** project, including:  
- Directions for running the project  
- Libraries and frameworks used  
- Additional technologies (if applicable)  

---

## **Introduction**  
The **Code Smell Detector** is a tool designed to identify common **code smells** in Rust programs. It analyzes source code for the following issues:  
- **Long Parameter Lists**: Functions with too many parameters  
- **Long Lines of Code**: Functions that exceed a reasonable length  
- **Duplicated Code**: Identifies repeated code patterns and potential redundancy  

---

## **Libraries and Frameworks**  
The dependencies used in this project are listed in **`Cargo.toml`**:  
- **iced** (Version: `0.13.1`) → Used for UI rendering  
- **rfd** → Used for file dialog operations  

> *Optional features should be commented out if not in use.*  

---

## **File Structure**  

```
project folder
|
|-- src
|    |-- main.rs
|    |-- codeAnalyzer.rs
|    |-- tokenizer.rs
|    |-- astBuilder.rs
|    |-- fileManager.rs 
|
|-- Cargo.toml
|-- README.md
```

### **Module Responsibilities & Code Smell Detection**  

#### **1. `main.rs` (Entry Point)**  
- Calls `fileManager.rs` to read source code  
- Calls `tokenizer.rs` to tokenize the code  
- Calls `abstractSyntax.rs` to build the AST  
- Passes **AST & Tokens** to `codeAnalyzer.rs` for detecting code smells  
- Displays the results  

#### **2. `fileManager.rs` (File Handling)**  
- Reads `.rs` source files from a directory  
- Provides an interface to retrieve code content as a string  

#### **3. `tokenizer.rs` (Lexical Analysis)**  
- Breaks the code into tokens  
- Helps identify long function names  
- Extracts function signatures for further analysis  

#### **4. `astBuilder.rs` (AST Generation & Structural Analysis)**  
- Builds the **Abstract Syntax Tree (AST)** for analyzing code structure  

#### **5. `codeAnalyzer.rs` (Main Code Smell Detector)**  
- Receives **AST & Tokens**
- performs **code smell analysis**:  
  - **Long Function Names**: Extract from AST and apply length threshold  
  - **Long Parameter Lists**: Extract parameter count from AST  
  - **Duplicated Code**: Compare Tokens to detect identical code blocks  
  - **Semantic Duplicated Code (Extra Feature)**:  
    - Compare AST patterns for functions with similar logic  
---

## **Instructions**  

### **1. Install Rust**  
- Recommended **Rust Version 1.84.0** *(for compatibility with `iced` v0.13.1)*  
- Install Rust using **rustup**:  
  ```sh
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
  Or update Rust:  
  ```sh
  rustup update
  ```

### **2. Build the Project**  
```sh
cargo build
```

### **3. Run the Project**  
```sh
cargo run
```

---

## **Future Improvements**  
- Improve **semantic duplication detection** using **AST comparisons**  
- Implement **Jaccard similarity metrics** for better **metrics-based duplication analysis**  
- Enhance **user interface** and result visualization with `iced`  
- Add **support for additional programming languages**  

---

## **Updates**  
- *To be updated with new features and improvements.*  

---
