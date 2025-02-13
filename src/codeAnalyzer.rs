// Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// 1st due Date:
// final due date:  
// CSPC5260 WQ25 Version 0.0
// analyzer.rs
// purpose

use crate::astBuilder::{ FileAst };
use crate::tokenizer::{ Token, TokenType };
use regex::Regex;

pub struct CodeAnalyzer {
    // text container
    ast_content: Option<FileAst>,
    tokenized_content: Vec<Token>,
    analysis_result: String,
}

impl CodeAnalyzer {

    pub fn new() -> Self {

        CodeAnalyzer {
            ast_content: None,
            tokenized_content: Vec::new(),
            analysis_result: String::from(""),
        }
    }

    pub fn set_ast_content(&mut self, content: FileAst) {
        self.ast_content = Some(content);
    }

    pub fn set_tokenized_content(&mut self, content: Vec<Token>) {
        self.tokenized_content = content;
    }

    pub fn get_analysis_result(&mut self) -> String {
        self.get_line_of_code();
        self.find_long_method_name();
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

        self.analysis_result.push_str(&format!("LOC of updated Code is {}\n", loc));
    }

    // get AST -> Analyzing
    fn find_long_method_name(&mut self) {

        let length_threshold = 30;

        let long_names: Vec<String> = <Option<FileAst> as Clone>::clone(&self.ast_content).unwrap().functions
            .iter()
            .filter(|f| f.name.len() > length_threshold)
            .map(|f| f.name.clone())
            .collect();

        // if statement when there is long method name?
        
        self.analysis_result.push_str(&format!("the long method names are {:?}\n", long_names));
    }

    fn find_long_parameter_llist(&mut self) {
        // ast is better
        // find parameter list
        // then return function name when over 3ea parameters

    }

    fn find_semantic_duplicated (&mut self) {
        // compare ast for semantic
        // compare the structures among functions
        // then return the % 
    }

    fn find_jaccard_duplicated (&mut self) {
        //based on tokenized content -> compare directly
        // save 1 -> fn
        // save 2 -> fn
        // save 3 -> fn
        //......... until there is no more fn
        // the comparing between 1 and 2
        // 2 and 3
        // continue until the end. 
        // detected -> return fn names duplicated (multiple dup possible) over 90%
    }

}
