// Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// 1st due Date:
// final due date:  
// CSPC5260 WQ25 Version 0.0
// analyzer.rs
// purpose

use syn::{parse_file, File, Item, ItemFn, FnArg, PatType, Pat, Type, ReturnType,};
use crate::tokenizer::{ Token, TokenType };

#[derive(Debug, Clone)]
pub struct Splittedfunction {
    pub name: String,
    pub tokens: Vec<Token>,
}

pub struct CodeAnalyzer {
    // text container
    ast_content: Option<syn::File>, // should be changed with new astBuilder
    tokenized_content: Vec<Token>,
    splitted_functions: Vec<Splittedfunction>,
    analysis_result: String,
}

impl CodeAnalyzer {

    pub fn new() -> Self {

        CodeAnalyzer {
            ast_content: None,
            tokenized_content: Vec::new(),
            splitted_functions: Vec::new(),
            analysis_result: String::from(""),
        }
    }

    pub fn set_ast_content(&mut self, content: syn::File) {
        self.ast_content = Some(content);
    }

    pub fn set_tokenized_content(&mut self, content: Vec<Token>) {
        self.tokenized_content = content;
    }

    pub fn get_analysis_result(&mut self) -> String {

        self.analysis_result.clear();
        
        self.get_line_of_code();
        self.find_long_method_name();
        self.find_long_parameter_list();
        self.find_duplicated_by_jaccard();
        
        self.analysis_result.clone()
    }

    fn get_line_of_code(&mut self) {
        if self.tokenized_content.is_empty() {
            return;
        }

        let loc = self.tokenized_content
        .last()
        .map(|last_token| last_token.line.to_string()) 
        .unwrap_or_else(|| "Unknown".to_string());

        self.analysis_result.push_str(&format!(" - LOC of updated Code is {}\n\n", loc));
    }

    fn find_long_method_name(&mut self) {
        // using ast

    }

    fn find_long_parameter_list(&mut self) {
        // using ast
        
    }

    fn find_semantic_duplicated(&mut self) {
        // compare ast for semantic
        // compare the structures among functions
        // then return the % 
    }

    fn find_duplicated_by_jaccard (&mut self) {

        self.separate_functions();

        if self.splitted_functions.len() < 2 {
            self.analysis_result.push_str("Not Enough Functions to Compare")
        }

        let mut result_string = String::new();
        let threshold = 0.9;

        for i in 0..self.splitted_functions.len() {
            for j in i + 1..self.splitted_functions.len() {
                let fn1 = &self.splitted_functions[i];
                let fn2 = &self.splitted_functions[j];
                let similarity: f64 = self.jaccard_similarity(fn1, fn2);
                let percentage = similarity * 100.0;

                if similarity >= threshold {
                    result_string.push_str(&format!(
                        "Function {} and {} are duplicated with {:.2}% similarity\n", 
                        i+1, j+1, percentage
                    ));
                } else {
                    result_string.push_str(&format!(
                        "Function {} and Function {} are not duplicated, similarity: {:.2}%\n", 
                        i+1, j+1, percentage
                    ));
                }
            }
        }

        self.analysis_result.push_str(&result_string);

    }
    
    fn separate_functions(&mut self) {
        self.splitted_functions.clear();

        let mut current_function_tokens: Vec<Token> = Vec::new();
        let mut current_function_name = String::from("unknown");
        let mut in_function = false;

        for token in &self.tokenized_content {
            // When we see a "fn" keyword, we treat it as the start of a new function.
            if let TokenType::Keyword(ref kw) = token.token_type {
                if kw == "fn" {
                    // If we were already collecting tokens for a function,
                    // push the current function into splitted_functions.
                    if in_function && !current_function_tokens.is_empty() {
                        self.splitted_functions.push(Splittedfunction {
                            name: current_function_name.clone(),
                            tokens: current_function_tokens.iter().map(|t| Token {
                                token_type: t.token_type.clone(),
                                line: 0,
                                column: 0,
                            }).collect(),
                        });
                        // Reset for the next function.
                        current_function_tokens.clear();
                        current_function_name = String::from("unknown");
                    }
                    in_function = true;
                }
            }

            // If we are in a function, accumulate the token.
            if in_function {
                current_function_tokens.push(token.clone());

                // Heuristically set the function name:
                // The first non-whitespace identifier following the "fn" keyword
                // is assumed to be the function name.
                if current_function_name == "unknown" {
                    if let TokenType::Identifier(ref id) = token.token_type {
                        // A quick check: ensure the previous token was "fn"
                        // (i.e. the identifier comes immediately after "fn" or after some whitespace)
                        if current_function_tokens.len() >= 2 {
                            let prev_token = &current_function_tokens[current_function_tokens.len() - 2];
                            if let TokenType::Keyword(ref prev_kw) = prev_token.token_type {
                                if prev_kw == "fn" {
                                    current_function_name = id.clone();
                                }
                            }
                        }
                    }
                }
            }
        }

        // After looping through all tokens, if we were in a function, save the last one.
        if in_function && !current_function_tokens.is_empty() {
            self.splitted_functions.push(Splittedfunction {
                name: current_function_name,
                tokens: current_function_tokens.iter().map(|t| Token {
                    token_type: t.token_type.clone(),
                    line: 0,
                    column: 0,
                }).collect(),
            });
        }
    }
    
    fn jaccard_similarity (&self, fn1: &Splittedfunction, fn2: &Splittedfunction) -> f64 {
        use std::collections::HashSet;

        let set1: HashSet<String> = fn1
        .tokens
        .iter()
        .filter_map(|t| self.token_as_string(t))
        .collect();

        let set2: HashSet<String> = fn2
        .tokens
        .iter()
        .filter_map(|t| self.token_as_string(t))
        .collect();

        let intersection = set1.intersection(&set2).count() as f64;
        let union = set1.union(&set2).count() as f64;

        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    fn token_as_string(&self, token: &Token) -> Option<String> {
        match &token.token_type {
            TokenType::Keyword(kw) => Some(kw.clone()),
            TokenType::Identifier(id) => Some(id.clone()),  // You may replace this with "<ID>" if needed
            TokenType::Number(num) => Some(num.clone()),
            TokenType::Symbol(s) => Some(s.clone()),
            TokenType::StringLitral(s) => Some(s.clone()),
            TokenType::Whitespace => None,    // Ignore whitespace
            TokenType::Comment(_) => None,    // Ignore comments
            TokenType::Unknown(_) => None,    // Ignore unknown chars
        }
    }
}
