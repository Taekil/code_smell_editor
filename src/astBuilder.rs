// AST
// BCS types:
// Sequence: a list of instructions with no other BCSs involved.
// Selection: if...else...
// Iteration: for() while()
// parse - syntatic analysis - building ast then these result passing to analyzer(or normalizer)
// and make report -> thing about the type to pass. 

use crate::tokenizer::{Token, TokenType};

#[derive(Debug)]
pub struct FileAst {
    pub functions: Vec<FunctionDef>,
}

#[derive(Debug)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<Parameter>,
    pub body: Block,
}

#[derive(Debug)]
pub struct Parameter {
    pub name: String,
    pub ty: String,
}

#[derive(Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

#[derive(Debug)]
pub enum Stmt {
    Let { name: String, expr: Expr },
    Expr(Expr),
    // You can add variants for control flow if you want them at the statement level.
    Selection {
        condition: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    Iteration {
        condition: Expr,
        body: Box<Stmt>,
    },
}

#[derive(Debug)]
pub enum Expr {
    Ident(String),
    StringLit(String),
    BinaryOp {
        op: String,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    MacroCall {
        macro_name: String,
        args: Vec<Expr>,
    },
}

pub struct AST_Builder {
    tokens: Vec<Token>,
    current_index: usize,
}

impl AST_Builder {

    pub fn new() -> Self {
        AST_Builder {
            tokens: Vec::new(),
            current_index: 0,
        }
    }

    pub fn set_tokens(&mut self, tokens: Vec<Token>) {
        self.tokens = tokens;
    }

    fn next_significant_token(&mut self) -> Option<Token> {
        while self.current_index < self.tokens.len() {
            let tok = &self.tokens[self.current_index];
            self.current_index += 1;

            match tok.token_type {
                TokenType::Whitespace | TokenType::Comment(_) => {
                    continue;
                },
                _=> {
                    return Some(tok.clone());
                },
            }
        }
        None
    }

    fn parse_statement(&mut self) -> Result<Stmt, String> {
        let token = self.next_significant_token().ok_or("Unexpected Coming")?;
        match token.token_type {
            _=> {
                let expr = self.parse_expression()?;
                self.expect_symbol(";")?;
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_expression(&mut self) {
        
    }

    fn parse_block() {

    }

    fn parse_function_definition() {

    }

    fn parse_selection() {

    }

    fn parse_iteration() {

    }

}