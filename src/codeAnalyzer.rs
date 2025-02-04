// Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// 1st due Date:
// final due date:  
// CSPC5260 WQ25 Version 0.0
// analyzer.rs
// purpose

pub struct CodeAnalyzer {
    // text container
    content: String,
}

impl CodeAnalyzer {

    pub fn new() -> Self {
        CodeAnalyzer {
            content: String::new(),
        }
    }

    pub fn set_tokened_content(&mut self, content: String) {
        self.content = content;
    }

    pub fn get_content(&self) -> &str {
        &self.content
    }

    pub fn get_line_of_code() {

    }

    // get AST -> Analyzing
    
}

// helper functions 

// check basic things like LOC(SLOC), architecture...etc. (helping functions can be...)

// long parameters

// long lines 

// duplicated codes

// Semantically Equicalent Code...
