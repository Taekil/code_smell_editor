// AST
// BCS types:
// Sequence: a list of instructions with no other BCSs involved.
// Selection: if...else...
// Iteration: for() while()
// parse - syntatic analysis - building ast then these result passing to analyzer(or normalizer)
// and make report -> thing about the type to pass. 

use syn::{parse_file,};

pub struct AST_Builder {

}

impl AST_Builder {
    pub fn new() -> Self {
        AST_Builder {
        }
    }

    pub fn parse_code(&mut self, source: String) -> Result<syn::File, syn::Error> {
        let file_ast = parse_file(&source)?;
        Ok(file_ast)
    }
    
}
