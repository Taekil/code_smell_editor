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

    pub fn set_content(&mut self, text: &str) {
        self.content = text.to_string();
    }

    pub fn perfrom_line_count(&self) -> String {
        let line_count = self.content.lines().count();
        format!("{}", line_count)
    }

}

// helper functions 

// check basic things like LOC(SLOC), architecture...etc. (helping functions can be...)

// long parameters

// long lines 

// duplicated codes