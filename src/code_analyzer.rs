// Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// 1st due Date:
// final due date:  
// CSPC5260 WQ25 Version 0.0
// analyzer.rs
// purpose

use syn::{parse_file, Item, ItemFn};
use syn::spanned::Spanned;
use std::collections::HashSet;
use quote::ToTokens;

#[derive(Clone)] 
pub struct FunctionInfo {
    pub name: String,
    #[cfg_attr(feature = "debug", derive(Debug))] 
    pub ast: ItemFn,
}

pub struct CodeAnalyzer {
    ast_content: Option<syn::File>,
    analysis_result: String,
}

impl CodeAnalyzer {
    pub fn new() -> Self {
        CodeAnalyzer {
            ast_content: None,
            analysis_result: String::from(""),
        }
    }

    pub fn set_ast_content(&mut self, source: String) -> Result<(), syn::Error>{
        self.ast_content = Some(parse_file(&source)?);
        Ok(())
    }

    pub fn get_analysis_result(&mut self) -> String {
        self.analysis_result.clear();
        self.analysis_result.push_str("** LOC **\n");
        self.get_line_of_code();
        self.analysis_result.push_str("** Long Method Name **\n");
        self.find_long_method_name();
        self.analysis_result.push_str("** Long Parameter List **\n");
        self.find_long_parameter_list();
        self.analysis_result.push_str("** Jaccard Metrics **\n");
        self.find_duplicated_by_jaccard();

        self.analysis_result.clone()
    }

    pub fn refactored_by_jaccard_result(&mut self) -> String {
        if let Some(ast) = &self.ast_content {
            let functions = self.collect_functions(ast);
            let mut unique_functions: Vec<ItemFn> = Vec::new();
            let mut seen: HashSet<String> = HashSet::new();
            let threshold = 0.9;

            for i in 0..functions.len() {
                let mut is_duplicate = false;
                for j in 0..i {
                    let similarity = self.jaccard_similarity(&functions[i].ast, &functions[j].ast);
                    if similarity >= threshold {
                        is_duplicate = true;
                        break;
                    }
                }
                if !is_duplicate {
                    let name = functions[i].name.clone();
                    unique_functions.push(functions[i].ast.clone());
                    seen.insert(name);
                }
            }

            let mut new_ast = ast.clone();
            new_ast.items = unique_functions.into_iter().map(Item::Fn).collect();

            self.format_ast_to_string(&new_ast)
        } else {
            "// No AST content available for refactoring".to_string()
        }
    }

    fn get_line_of_code(&mut self) {
        if let Some(ast) = &self.ast_content {
            if ast.items.is_empty() {
                self.analysis_result.push_str(" - LOC of updated Code is 0\n");
                return;
            }

            let (mut min_line, mut max_line) = (usize::MAX, 0);

            for item in &ast.items {
                let span = match item {
                    Item::Fn(func) => {
                        // Use the block's span
                        func.block.span()
                    }
                    _ => {
                        // Or the entire item's span if not a function
                        item.span()
                    }
                };

                let start_line = span.start().line;
                let end_line   = span.end().line;

                if start_line < min_line {
                    min_line = start_line;
                }
                if end_line > max_line {
                    max_line = end_line;
                }
            }

            let loc = if min_line == usize::MAX {
                0
            } else {
                max_line.saturating_sub(min_line) + 1
            };

            self.analysis_result
                .push_str(&format!(" - LOC of updated Code is {}\n", loc));
        } else {
            self.analysis_result.push_str(" - No AST content available\n");
        }
    }

    fn find_long_method_name(&mut self) {
        const THRESHOLD: usize = 20;
        if let Some(ast) = &self.ast_content {
            for item in &ast.items {
                if let Item::Fn(func) = item {
                    let name = func.sig.ident.to_string();
                    if name.len() > THRESHOLD {
                        self.analysis_result.push_str(&format!(
                            " - Function '{}' has a long name ({} characters)\n",
                            name, name.len()
                        ));
                    }
                }
            }
        }
    }

    fn find_long_parameter_list(&mut self) {
        const THRESHOLD: usize = 3;
        if let Some(ast) = &self.ast_content {
            for item in &ast.items {
                if let Item::Fn(func) = item {
                    let name = func.sig.ident.to_string();
                    let param_count = func.sig.inputs.len();
                    if param_count > THRESHOLD {
                        self.analysis_result.push_str(&format!(
                            " - Function '{}' has too many parameters ({} parameters)\n",
                            name, param_count
                        ));
                    }
                }
            }
        }
    }

    fn find_duplicated_by_jaccard(&mut self) {
        if let Some(ast) = &self.ast_content {
            let functions = self.collect_functions(ast);
            if functions.len() < 2 {
                self.analysis_result.push_str(" - Not enough functions to compare\n");
                return;
            }

            let threshold = 0.9;
            for i in 0..functions.len() {
                for j in i + 1..functions.len() {
                    let fn1 = &functions[i];
                    let fn2 = &functions[j];
                    let similarity = self.jaccard_similarity(&fn1.ast, &fn2.ast);
                    let percentage = similarity * 100.0;
                    if similarity >= threshold {
                        self.analysis_result.push_str(&format!(
                            " - Function '{}' and '{}' are duplicated with {:.2}% similarity\n",
                            fn1.name, fn2.name, percentage
                        ));
                    }
                }
            }
        }
    }

    fn collect_functions(&self, ast: &syn::File) -> Vec<FunctionInfo> {
        let mut functions = Vec::new();
        for item in &ast.items {
            if let Item::Fn(func) = item {
                functions.push(FunctionInfo {
                    name: func.sig.ident.to_string(),
                    ast: func.clone(),
                });
            }
        }
        functions
    }

    fn jaccard_similarity(&self, fn1: &ItemFn, fn2: &ItemFn) -> f64 {
        use quote::ToTokens;

        let tokens1: Vec<String> = fn1
            .block
            .stmts
            .iter()
            .flat_map(|stmt| {
                stmt.to_token_stream()
                    .to_string()
                    .split_whitespace()
                    .map(String::from)
                    .collect::<Vec<_>>() // Collect into Vec to own the data
            })
            .collect();
        let tokens2: Vec<String> = fn2
            .block
            .stmts
            .iter()
            .flat_map(|stmt| {
                stmt.to_token_stream()
                    .to_string()
                    .split_whitespace()
                    .map(String::from)
                    .collect::<Vec<_>>() // Collect into Vec to own the data
            })
            .collect();

        let set1: HashSet<String> = tokens1.into_iter().collect();
        let set2: HashSet<String> = tokens2.into_iter().collect();

        let intersection = set1.intersection(&set2).count() as f64;
        let union = set1.union(&set2).count() as f64;

        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    fn format_ast_to_string(&self, ast: &syn::File) -> String {
        let mut code = String::new();
        for item in &ast.items {
            let item_str = item.into_token_stream().to_string();
            code.push_str(&item_str);
            code.push_str("\n\n");
        }
        code
    }
}