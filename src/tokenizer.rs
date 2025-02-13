// Purpose only consider the rust code 

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TokenType {
    Keyword(String),
    Identifier(String),
    Number(String),
    Symbol(String),
    StringLitral(String),
    Whitespace,
    Comment(String),
    Unknown(char),
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub line: usize,
    pub column: usize,
}

pub struct Tokenizer {
    input: String,
    current_pos: usize,
    current_line: usize,
    current_col: usize,
    tokens: Vec<Token>,
}

impl Tokenizer {

    pub fn new() -> Self {
        Tokenizer {
            input: String::new(), 
            current_pos: 0,
            current_line: 1,
            current_col: 1,
            tokens: Vec::new(),
        }
    }

    pub fn set_input(&mut self, input: String) {
        self.input = input;
        self.current_pos = 0;    // Reset the reading position
        self.current_line = 1;   // Reset line counter
        self.current_col = 1;    // Reset column counter
        self.tokens.clear();
    }

    pub fn tokenize(&mut self) -> Vec<Token> {

        //println!("Tokenizing input: {}", self.input);

        if self.input.trim().is_empty() {
            println!("Tokenizer received empty input!");
            return vec![]
        }

        while let Some(ch) = self.peek_char() {
            let token = if ch.is_whitespace() {
                self.consume_whitespace()
            } else if ch.is_alphabetic() || ch == '_' {
                self.consume_identifier_or_keyword()
            } else if ch.is_ascii_digit() {
                self.consume_number()
            } else if ch== '"'{
                self.consume_string_literal()
            } else if ch == '/' {
                if let Some(next_ch) = self.peek_next_char() {
                    if next_ch == '/' {
                        self.consume_comment()
                    } else {
                        self.consume_symbol()
                    }
                } else {
                    self.consume_symbol()
                }
            } else {
                self.consume_symbol()
            };

            self.tokens.push(token);
        }

        self.tokens.clone()
    }

    pub fn print_tokens(&self) {
        for token in &self.tokens {
            println!("{:?}", token);
        }

    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.current_pos..].chars().next()
    }

    fn next_char(&mut self) -> Option<char> {
        
        let ch = match self.peek_char() {
            Some(value) => value,
            None => return None,
        };
        
        self.current_pos += ch.len_utf8();
        if ch =='\n' {
            self.current_line += 1;
            self.current_col = 1;
        }
        else {
            self.current_col += 1;
        }
        
        Some(ch)
    }

    fn peek_next_char(&self) -> Option<char> {
        let mut iter = self.input[self.current_pos..].chars();
        iter.next()?;
        iter.next()
    }

    fn consume_whitespace(&mut self) -> Token {
        let line = self.current_line;
        let col = self.current_col;

        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.next_char();
            } else {
                break;
            }
        }

        Token {
            token_type: TokenType::Whitespace,
            line, 
            column: col,
        }
    }

    fn consume_comment(&mut self) -> Token {
        let line = self.current_line;
        let col = self.current_col;

        self.next_char();
        self.next_char();

        let mut comment = String::new();

        while let Some(ch) = self.peek_char() {
            if ch == '\n' {
                break;
            }
            comment.push(ch);
            self.next_char();
        }

        Token {
            token_type: TokenType::Comment(comment),
            line,
            column: col,
        }
    }

    fn consume_identifier_or_keyword(&mut self) -> Token {

        let line = self.current_line;
        let col = self.current_col;
        let mut text = String::new();

        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || ch == '_' {
                text.push(ch);
                self.next_char();
            } else {
                break;
            }
        }

        let keywords = ["fn", "let", "mut", "if", "else", "while"];
        let token_type = if keywords.contains(&&text.as_str()) {
            TokenType::Keyword(text)
        } else {
            TokenType::Identifier(text)
        };

        Token {
            token_type,
            line,
            column: col,
        }
    }

    fn consume_number(&mut self) -> Token {
        let line = self.current_line;
        let col = self.current_col;
        let mut number_str = String::new();

        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                self.next_char();
            } else {
                break;
            }
        }

        Token {
            token_type: TokenType::Number(number_str),
            line,
            column: col,
        }
    }

    fn consume_string_literal(&mut self) -> Token {
        let line = self.current_line;
        let col = self.current_col;
        let mut literal = String::new();
        let mut escaping = false;

        self.next_char();

        while let Some(ch) = self.peek_char() {
            if escaping {
                match ch {
                    'n'=>literal.push('\n'),
                    't'=>literal.push('\t'),
                    '"'=>literal.push('\\'),
                    _=>literal.push(ch),
                }
                escaping = false;
            } else if ch == '\\' {
                escaping = true;
            } else if ch == '"' {
                self.next_char();
                break;
            } else {
                literal.push(ch);
            }
            self.next_char();  
        } 

        Token {
            token_type: TokenType::StringLitral(literal),
            line,
            column: col,
        }
    }

    fn consume_symbol(&mut self) -> Token {
        let line = self.current_line;
        let col = self.current_col;
        if let Some(ch) = self.next_char() {
            Token {
                token_type: TokenType::Symbol(ch.to_string()),
                line,
                column: col,
            }
        } else {
            //if no characters
            Token {
                token_type: TokenType::Unknown('\0'),
                line,
                column: col,
            }
        }
    }
}
