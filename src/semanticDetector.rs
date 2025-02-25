use pyo3::prelude::*;
use pyo3::types::PyList;
use syn::{parse_file, Item};
use quote::quote;

pub struct SemanticDetector {
    semantic_analysis_result: String,
}

impl SemanticDetector {
    pub fn new() -> Self {
        SemanticDetector {
            semantic_analysis_result: String::from(""),
        }
    }

    pub fn get_result(&self) -> String {
        self.semantic_analysis_result.clone()
    }

    pub fn detect_duplicates(&self, code: &str, threshold: f32) -> PyResult<()> {
        println!("Starting analysis with code length: {}", code.len());
        
        // Check function extraction
        let functions = self.extract_functions(code);
        println!("Found {} functions", functions.len());
        
        if functions.len() < 2 {
            println!("Not enough functions to compare");
            return Ok(());
        }

        // Debug print the functions found
        for (i, func) in functions.iter().enumerate() {
            println!("Function {}: {}...", i, func.chars().take(50).collect::<String>());
        }

        Python::with_gil(|py| -> PyResult<()> {
            println!("Python GIL acquired");

            // Add project root to Python's sys.path
            let sys = py.import_bound("sys")?;
            let path = sys.getattr("path")?;
            let project_root = std::env::current_dir()?.to_str().unwrap().to_string();
            path.call_method1("append", (project_root,))?;

            let inference = match PyModule::import_bound(py, "inference") {
                Ok(module) => {
                    println!("Successfully imported inference module");
                    module
                },
                Err(e) => {
                    println!("Failed to import inference: {}", e);
                    return Err(e);
                }
            };

            let mut embeddings = Vec::new();
            let get_embedding = inference.getattr("get_embedding")?;
            let compute_similarity = inference.getattr("compute_similarity")?;

            // Generate embeddings
            for (i, func) in functions.iter().enumerate() {
                match get_embedding.call1((func,)) {
                    Ok(result) => {
                        let embedding: Vec<f32> = result.extract()?;
                        println!("Generated embedding {} with length {}", i, embedding.len());
                        embeddings.push(embedding);
                    },
                    Err(e) => println!("Embedding error for function {}: {}", i, e),
                }
            }

            // Compare embeddings
            println!("Starting similarity comparisons");
            for i in 0..functions.len() {
                for j in (i + 1)..functions.len() {
                    let similarity: f32 = compute_similarity
                        .call1((embeddings[i].clone(), embeddings[j].clone()))?
                        .extract()?;
                    println!("Similarity between {} and {}: {}", i, j, similarity);
                    
                    if similarity >= threshold {
                        println!("Potential duplicate detected (Similarity: {:.3}):", similarity);
                        println!("Function {}: {}...", i + 1, functions[i].chars().take(50).collect::<String>());
                        println!("Function {}: {}...", j + 1, functions[j].chars().take(50).collect::<String>());
                        println!();
                    }
                }
            }
            Ok(())
        })?;
        println!("Analysis completed");
        Ok(())
    }

    fn extract_functions(&self, code: &str) -> Vec<String> {
        match parse_file(code) {
            Ok(syntax) => {
                let mut functions = Vec::new();
                for item in syntax.items {
                    if let Item::Fn(func) = item {
                        let func_str = quote!(#func).to_string();
                        functions.push(func_str);
                    }
                }
                functions
            }
            Err(e) => {
                println!("Parse error: {}", e);
                Vec::new()
            }
        }
    }
}
