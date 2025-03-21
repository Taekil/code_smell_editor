# **Code Smell Editor**  

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
- **Semantic Duplicated Code**: Identifies `while` and `for` loop. 

---

## **Libraries and Frameworks**  
The dependencies used in this project are listed in **`Cargo.toml`**:  
- **iced** (Version: `0.13.1`) → Used for UI rendering  
- **rfd** → Used for file dialog operations  
- **pyo3** → For binding Rust code with Python
- **syn** → A Rust library for parsing Rust Code as an Abstact Syntax Tree(AST)
- **quote** → Used `syn' to transform Rust AST into Valid Rust Code
- **pytorch** → A Pytorch libarary used for deep learning and tensor operations  
- **pickle** → Python module for serializing and deserializing objects
- **re** → Used for pattern matching and text processing

> *Optional features should be commented out if not in use.*  

---

## **File Structure**  

```
project folder
|
|-- dl_training_file
|    |-- dl_traiing_model.py
|    |-- rust_semantic_duplicate_data.csv
|
|-- src
|    |-- main.rs
|    |-- code_analyzer.rs
|    |-- file_manager.rs 
|    |-- semantic_detector.rs 
|
|-- Cargo.toml
|-- README.md
|-- inference.py
|-- semantic_duplicate_detector.pth
|-- token_to_index.pxl

```

### **Module Responsibilities & Code Smell Detection**  

#### **1. `main.rs` (Entry Point)**  
- Calls `fileManager.rs` to read source code  
- Calls `tokenizer.rs` to tokenize the code  
- Calls `astBuilder.rs` to build the AST  
- Passes **AST & Tokens** to `codeAnalyzer.rs` for detecting code smells 
- Passes **Code** to `semanticDetector.rs` for semantic duplicated code detection
- Displays the results  

#### **2. `fileManager.rs` (File Handling)**  
- Reads `.rs` source files from a directory  
- Provides an interface to retrieve code content as a string  

#### **5. `codeAnalyzer.rs` (Main Code Smell Detector)**  
- Receives **AST & Tokens**
- performs **code smell analysis**:  
  - **Long Function Names**: Extract from AST and apply length threshold  
  - **Long Parameter Lists**: Extract parameter count from AST  
  - **Duplicated Code**: Compare Tokens to detect identical code blocks  

#### **6. `semanticDetector.rs` (Main Code Smell Detector)**
  - **Semantic Duplicated Code (Extra Feature)**:  
    - Compare AST patterns for functions with similar logic  

#### **7. Additional Files**
  - These files to use sementic detection by Machine Learning(ML)
    - **inference.py**: allowing intergration of AI inference into Rust-based Code Smell Detector
    - **semantic_detect_training_model.py**: the script to train ML model 
    - **semantic_duplicate_detector.pth**: pre-trained weights
    - **token_to_index.pkl**: pre-trained vocabulary
    - **dl_traiing_file**: dl_training_model.py and rust_semantic_duplicate_data.csv to build the DL-NN for train and save

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

### **4. OPTIONAL**
#### **Instructions for AI Model Training, Saving, and Inference**
- Update Model with Additional Training
    - Add new training and test data to the dataset.
    - Load the existing model and continue training.
    - Run semantic_`detect_training_model.py` again.
    - Save the updated model in the project directory.
      - `semantic_duplicate_detector.pth`
      - `token_to_index.pkl`

#### **Verify Python3**
```sh
python3 --version
```

#### **Install Python3**
on mac
```sh
brew install python3
```
on window
- Download the lastest python3 from `python.org`

#### **Install NumPy**
```sh
pip3 install numpy
```

#### **Install PyTorch**
```sh
pip3 install torch
```

#### **Run the script**
```sh
python3 semantic_detect_training_model.py
```

---

## **Future Improvements**  
- Add **support for additional programming languages**  

---

## **Updates**  
- *To be updated with new features and improvements.*  

---
