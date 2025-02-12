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
    Number(String),
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
        self.current_index = 0;
    }

    pub fn parse_tokens(&mut self) -> Result<FileAst, String> {
        let mut functions = Vec::new();
        // Continue parsing while there are tokens left.
        while let Some(token) = self.peek_significant() {
            // Here, we assume that top-level function definitions always start with "fn"
            if let TokenType::Keyword(ref s) = token.token_type {
                if s == "fn" {
                    let func = self.parse_function_definition()?;
                    functions.push(func);
                } else {
                    return Err(format!("Unexpected token at top-level: {:?}", token));
                }
            } else {
                return Err(format!("Unexpected token at top-level: {:?}", token));
            }
        }
        Ok(FileAst { functions })
    }
    
    // Returns a reference to the token at the current index (even if it's whitespace).
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current_index)
    }

    // Returns a reference to the next token that is not whitespace or a comment.
    fn peek_significant(&self) -> Option<&Token> {
        let mut idx = self.current_index;
        while idx < self.tokens.len() {
            let tok = &self.tokens[idx];
            match tok.token_type {
                TokenType::Whitespace | TokenType::Comment(_) => idx += 1,
                _ => return Some(tok),
            }
        }
        None
    }

    // Consume and return the next significant token.
    fn next_significant_token(&mut self) -> Option<Token> {
        while self.current_index < self.tokens.len() {
            let tok = &self.tokens[self.current_index];
            self.current_index += 1;
            match tok.token_type {
                TokenType::Whitespace | TokenType::Comment(_) => continue,
                _ => return Some(tok.clone()),
            }
        }
        None
    }

    fn expect_symbol(&mut self, symbol: &str) -> Result<Token, String> {
        let token = self.next_significant_token().ok_or_else(|| {
            format!("Expected symbol '{}', but reached end of input", symbol)
        })?;
        match token.token_type {
            TokenType::Symbol(ref s) if s == symbol => Ok(token),
            _ => Err(format!("Expected symbol '{}', but found a different token", symbol)),
        }
    }

    fn expect_keyword(&mut self, keyword: &str) -> Result<Token, String> {
        let token = self.next_significant_token().ok_or_else(|| {
            format!("Expected keyword '{}', but reached end of input", keyword)
        })?;
        match token.token_type {
            TokenType::Keyword(ref s) if s == keyword => Ok(token),
            _ => Err(format!("Expected keyword '{}', but found a different token", keyword)),
        }
    }

    fn parse_function_definition(&mut self) -> Result<FunctionDef, String> {
        // Expect the "fn" keyword.
        self.expect_keyword("fn")?;
    
        // Expect a function name (identifier).
        let name_token = self.next_significant_token().ok_or("Expected function name after 'fn'")?;
        let name = match name_token.token_type {
            TokenType::Identifier(ref s) => s.clone(),
            _ => return Err("Expected identifier for function name".to_string()),
        };
    
        // Expect the opening parenthesis for the parameter list.
        self.expect_symbol("(")?;
    
        let mut params = Vec::new();
    
        // Check if the next significant token indicates an empty parameter list.
        if let Some(token) = self.peek_significant() {
            match token.token_type {
                TokenType::Symbol(ref s) if s == ")" => {
                    // Empty parameter list: consume the ")".
                    self.next_significant_token();
                },
                _ => {
                    // Parameter list is not empty; parse parameters.
                    loop {
                        // Parse parameter name.
                        let param_name_token = self.next_significant_token().ok_or("Expected parameter name")?;
                        let param_name = match param_name_token.token_type {
                            TokenType::Identifier(ref s) => s.clone(),
                            _ => return Err("Expected identifier for parameter name".to_string()),
                        };
    
                        // Expect a colon.
                        self.expect_symbol(":")?;
    
                        // Parse parameter type.
                        let param_type_token = self.next_significant_token().ok_or("Expected parameter type")?;
                        let param_type = match param_type_token.token_type {
                            TokenType::Identifier(ref s) | TokenType::Keyword(ref s) => s.clone(),
                            _ => return Err("Expected identifier for parameter type".to_string()),
                        };
    
                        params.push(Parameter {
                            name: param_name,
                            ty: param_type,
                        });
    
                        // Look ahead using peek_significant():
                        // If a comma is found, consume it and continue;
                        // if a closing parenthesis is found, consume it and break.
                        if let Some(token) = self.peek_significant() {
                            if let TokenType::Symbol(ref s) = token.token_type {
                                if s == "," {
                                    self.next_significant_token(); // Consume the comma.
                                    continue;
                                } else if s == ")" {
                                    self.next_significant_token(); // Consume the ")".
                                    break;
                                } else {
                                    return Err(format!("Unexpected symbol '{}' in parameter list", s));
                                }
                            } else {
                                // If the next significant token is not a symbol, assume parameter list continues.
                                continue;
                            }
                        } else {
                            return Err("Unexpected end of tokens in parameter list".to_string());
                        }
                    }
                }
            }
        } else {
            return Err("Unexpected end of input when expecting parameter list".to_string());
        }
    
        // Parse the function body as a block.
        let body = self.parse_block()?;
    
        Ok(FunctionDef { name, params, body })
    }
    
    /// Parse a block: `{ stmt1 stmt2 ... }`
    fn parse_block(&mut self) -> Result<Block, String> {
        self.expect_symbol("{")?;
        let mut stmts = Vec::new();
        while let Some(token) = self.peek_significant() {
            // If we see a closing brace, the block is finished.
            if let TokenType::Symbol(ref s) = token.token_type {
                if s == "}" {
                    break;
                }
            }
            let stmt = self.parse_statement()?;
            stmts.push(stmt);
        }
        self.expect_symbol("}")?;
        Ok(Block { stmts })
    }

    fn parse_statement(&mut self) -> Result<Stmt, String> {
        // Use peek_significant() to decide which statement to parse.
        let token = self.peek_significant().ok_or("Unexpected end of input while parsing statement")?;
        match token.token_type {
            TokenType::Keyword(ref s) if s == "if" => self.parse_selection(),
            TokenType::Keyword(ref s) if s == "while" => self.parse_iteration(),
            TokenType::Keyword(ref s) if s == "let" => {
                // Handle let statements.
                self.next_significant_token(); // consume "let"
                let name_token = self.next_significant_token().ok_or("Expected variable name")?;
                let name = match name_token.token_type {
                    TokenType::Identifier(ref s) => s.clone(),
                    _ => return Err("Expected identifier for variable name".to_string()),
                };
                self.expect_symbol("=")?;
                let expr = self.parse_expression()?;
                self.expect_symbol(";")?;
                Ok(Stmt::Let { name, expr })
            }
            _ => {
                // Assume it's an expression statement.
                let expr = self.parse_expression()?;
                self.expect_symbol(";")?;
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_expression(&mut self) -> Result<Expr, String> {
        // Parse a primary expression.
        let mut left = self.parse_primary()?;

        // Loop while the next significant token is a binary operator.
        while let Some(token) = self.peek_significant() {
            if let TokenType::Symbol(ref s) = token.token_type {
                if s == "+" || s == "-" || s == "*" || s == "/" {
                    let op_token = self.next_significant_token().unwrap();
                    let op = if let TokenType::Symbol(ref op_str) = op_token.token_type {
                        op_str.clone()
                    } else {
                        return Err("Expected operator".to_string());
                    };

                    let right = self.parse_primary()?;
                    left = Expr::BinaryOp {
                        op,
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    };
                    continue;
                }
            }
            break;
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        let token = self.next_significant_token()
            .ok_or("Expected expression, found end of input")?;
        match token.token_type {
            TokenType::Identifier(ref s) => {
                // Check if the next significant token is "!" for a macro call.
                let is_macro_call = {
                    if let Some(next_token) = self.peek_significant() {
                        if let TokenType::Symbol(ref sym) = next_token.token_type {
                            sym == "!"
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                };
                if is_macro_call {
                    self.next_significant_token(); // consume "!" token
                    self.expect_symbol("(")?;
                    let mut args = Vec::new();
                    // Check for empty macro arguments.
                    if let Some(token) = self.peek_significant() {
                        if let TokenType::Symbol(ref sym) = token.token_type {
                            if sym == ")" {
                                self.next_significant_token(); // consume ")"
                                return Ok(Expr::MacroCall { macro_name: s.clone(), args });
                            }
                        }
                    }
                    // Parse macro arguments separated by commas.
                    loop {
                        let arg = self.parse_expression()?;
                        args.push(arg);
                        if let Some(token) = self.peek_significant() {
                            if let TokenType::Symbol(ref sym) = token.token_type {
                                if sym == "," {
                                    self.next_significant_token(); // consume comma
                                    continue;
                                } else if sym == ")" {
                                    self.next_significant_token(); // consume ")"
                                    break;
                                } else {
                                    return Err(format!("Unexpected symbol '{}' in macro arguments", sym));
                                }
                            } else {
                                return Err("Unexpected token in macro arguments".into());
                            }
                        } else {
                            return Err("Unexpected end of input in macro arguments".into());
                        }
                    }
                    return Ok(Expr::MacroCall { macro_name: s.clone(), args });
                }
                Ok(Expr::Ident(s.clone()))
            },
            TokenType::StringLitral(ref s) => Ok(Expr::StringLit(s.clone())),
            TokenType::Number(ref s) => Ok(Expr::Number(s.clone())),
            // Handle parenthesized expressions.
            TokenType::Symbol(ref s) if s == "(" => {
                let expr = self.parse_expression()?;
                self.expect_symbol(")")?;
                Ok(expr)
            },
            _ => Err("Unexpected token in expression".into()),
        }
    }

    /// Parse a selection statement (if-else).
    /// Grammar: `if (condition) statement [else statement]`
    fn parse_selection(&mut self) -> Result<Stmt, String> {
        self.expect_keyword("if")?;
        self.expect_symbol("(")?;
        let condition = self.parse_expression()?;
        self.expect_symbol(")")?;
        let then_branch = Box::new(self.parse_statement()?);
        let else_branch = if let Some(token) = self.peek_significant() {
            if let TokenType::Keyword(ref s) = token.token_type {
                if s == "else" {
                    self.next_significant_token(); // consume "else"
                    Some(Box::new(self.parse_statement()?))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        Ok(Stmt::Selection { condition, then_branch, else_branch })
    }

    /// Parse an iteration statement (while loop).
    /// Grammar: `while (condition) statement`
    fn parse_iteration(&mut self) -> Result<Stmt, String> {
        self.expect_keyword("while")?;
        self.expect_symbol("(")?;
        let condition = self.parse_expression()?;
        self.expect_symbol(")")?;
        let body = Box::new(self.parse_statement()?);
        Ok(Stmt::Iteration { condition, body })
    }
}
