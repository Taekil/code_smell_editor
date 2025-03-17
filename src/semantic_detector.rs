// Taekil Oh
// Start Date: Jan 23rd 2025
// CSPC5260 WQ25 Version 1.0
// semantic_detector.rs

use pyo3::prelude::*;
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

    pub fn detect_duplicates(&mut self, code: &str, threshold: f32) -> PyResult<()> {
        let mut result = String::new();
        result.push_str(&format!("** Starting analysis with character count: {} **\n", code.len()));
         
        let functions = self.extract_functions(code);
        result.push_str(&format!("Found {} functions\n", functions.len()));
        
        if functions.len() < 2 {
            println!("Not enough functions to compare\n");
            self.semantic_analysis_result = result;
            return Ok(());
        }

        Python::with_gil(|py| -> PyResult<()> {
            
            // Add project root to Python's sys.path
            let sys = py.import_bound("sys")?;
            let path = sys.getattr("path")?;
            let project_root = std::env::current_dir()?.to_str().unwrap().to_string();
            path.call_method1("append", (project_root,))?;

            let inference = match PyModule::import_bound(py, "inference") {
                Ok(module) => {
                    result.push_str("Successfully imported inference module\n");
                    module
                },
                Err(e) => {
                    result.push_str(&format!("Failed to import inference: {}\n", e));
                    return Err(e);
                }
            };

            let mut embeddings = Vec::new();
            let get_embedding = inference.getattr("get_embedding")?;
            let compute_similarity = inference.getattr("compute_similarity")?;

            for (i, func) in functions.iter().enumerate() {
                match get_embedding.call1((func,)) {
                    Ok(result_embedding) => {
                        let embedding: Vec<f32> = result_embedding.extract()?;
                        result.push_str(&format!(
                            "Generated embedding {} with length {}\n",
                            i,
                            embedding.len()
                        ));

                        embeddings.push(embedding);
                    },
                    Err(e) => result.push_str(&format!("Embedding error for function {}: {}\n", i, e)),
                }
            }

            result.push_str("\n** Similarity Comparisons **\n\n");
            for i in 0..functions.len() {
                for j in (i + 1)..functions.len() {
                    let similarity: f32 = compute_similarity
                        .call1((embeddings[i].clone(), embeddings[j].clone()))?
                        .extract()?;
                    result.push_str(&format!("Similarity between {} and {}: {}\n", i, j, similarity));
                    
                    if similarity >= threshold {
                        result.push_str(&format!(
                            "\nPotential duplicate detected (Similarity: {:.3}):\n", 
                            similarity
                        ));
                        result.push_str(&format!(
                            "Function {}: {}...\n", 
                            i, 
                            functions[i].chars().take(50).collect::<String>()
                        ));
                        result.push_str(&format!(
                            "Function {}: {}...\n", 
                            j, 
                            functions[j].chars().take(50).collect::<String>()
                        ));
                        result.push_str("\n");
                    }
                }
            }
            Ok(())
        })?;
        
        result.push_str("- End -\n");
        self.semantic_analysis_result = result;
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
