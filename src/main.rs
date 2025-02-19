// Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// 1st due Date:
// final due date:  
// CSPC5260 WQ25 Version 0.0
// main.rs
// Code Smell Detector, the purpose of main.rs 
/*
 1. main.rs (Entry Point)
Calls fileManager.rs to read the source code.
Calls tokenizer.rs to tokenize the code.
Calls astBuilder.rs to build the AST.

Displays results.

FEB 8th 2025, thinking about the result formatter to make better result organization. 
 */

use iced::widget::{button, column, container, row, scrollable, text, text_editor};
use iced::{application, Element};

mod codeAnalyzer;
mod fileManager;
mod tokenizer;
mod astBuilder;

use fileManager::FileManager;
use tokenizer::Tokenizer;
use codeAnalyzer::CodeAnalyzer;
use astBuilder::AST_Builder;

fn main() -> Result<(), iced::Error> {

    // run application
    application(CodeSmellDetector::title, CodeSmellDetector::update, CodeSmellDetector::view)
    .run_with(|| (CodeSmellDetector::new(), iced::Task::none()))
}

struct CodeSmellDetector {
    file_manager: FileManager,
    tokenizer: Tokenizer,
    codeAnalizer: CodeAnalyzer,
    astBuilder: AST_Builder,
    content: text_editor::Content,
    upload_button_label: String,
    analysis_button_label: String,
    refactor_button_label:String,
    save_button_label: String,
    clear_button_label: String,
    analysis_results: String, //Store results here
    // analyzer should be called here first?
}

#[derive(Debug, Clone)]
enum Message {
    Edit(text_editor::Action),
    UploadPressed,
    AnalysisPressed,
    RefactorPressed,
    SavePressed,
    ClearPressed,
} 

impl CodeSmellDetector {
    fn new() -> Self {
        Self{
            file_manager: FileManager::new(),
            tokenizer: Tokenizer::new(),
            codeAnalizer: CodeAnalyzer::new(),
            astBuilder: AST_Builder::new(),
            content: text_editor::Content::default(),
            upload_button_label: String::from("Upload Code"),
            analysis_button_label: String::from("Analysis"),
            refactor_button_label: String::from("Dup Refactor"),
            save_button_label: String::from("Save"),
            clear_button_label: String::from("Clear"),
            analysis_results: String::from(""), // empty Initially
        }
    }

    fn title(&self) -> String {
        String::from("Code Smell Detector")
    }

    fn update(&mut self, message: Message) {
        match message {
            Message::Edit(action) => {
                self.content.perform(action);
            }

            Message::UploadPressed => {
                match self.file_manager.upload_file() {
                        Ok(content) => {
                            self.content = text_editor::Content::with_text(&content);
                            self.upload_button_label = String::from("uploaded");

                            // self.tokenizer.set_input(content.clone()); 
                            println!("Tokenized Done in UploadPressed")
                        }
                        Err(_) => {
                            self.upload_button_label = String::from("failed");
                        }
                }
                println!("Code Uploaded");
            }

            Message::AnalysisPressed => {

                self.analysis_button_label = String::from("Started Analysis");

                let recent_code = self.content.text();
                self.tokenizer.set_input(recent_code.clone());
                
                let tokens: Vec<tokenizer::Token> = self.tokenizer.tokenize();
                self.codeAnalizer.set_tokenized_content(tokens.clone());

                let ast = self.astBuilder.parse_code(recent_code);
                match ast {
                    Ok(ast) => {
                        println!("ast built and pass to analyzer");
                        self.codeAnalizer.set_ast_content(ast);
                    } Err(err_msg) => {
                        eprintln!("Parsing Err: {}", err_msg);
                        return;
                    }
                };

                self.analysis_results = self.codeAnalizer.get_analysis_result();
                // AST-> Normalizer->Analyzer for semantic duplication analysis?
                // and how to apply the jaccard metrics? for metrics-based duplication detection?
                // but required -> limit for semantic dup? -> over 90% -> causes dup -> duplicated when refactoring. 
                // after normailzer -> Drawer to compare primary passed and updated by normalizer?
                // Normalizer can be called in Anlayzer directly, just return result at here. 
            }

            Message::RefactorPressed => {
                self.refactor_button_label = String::from("Dup Refactored");

                // adding the process refactoring
                // update the content
                // based on the analysis result
                // target -> duplicated code refactoring
            }

            Message::SavePressed => {

                let combined_content = format!(
                    "==== CODE ===\n{}\n\n==== ANALYSIS RESULTS ==== \n{}",
                    self.content.text(),
                    self.analysis_results,
                );

                let save_result = self.file_manager.save_file(&combined_content);
                self.analysis_results = save_result;
            }

            Message::ClearPressed => {
                // Reset all fields
                self.content = text_editor::Content::default();
                self.upload_button_label = String::from("Upload Code");
                self.analysis_button_label = String::from("Analysis");
                self.refactor_button_label = String::from("Dup Refactor");
                self.save_button_label = String::from("Save");
                self.analysis_results = String::from("");
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        /*
            Describes what the app looks like
            use iced::widget::text to display the message
            Wrap the text in an Element (a containder for UI Compoenents)
         */
        let input = text_editor(&self.content)
        .on_action(Message::Edit)
        .height(400.0)
        .width(1600.0);

        let upload_button = button(text(&self.upload_button_label))
        .on_press(Message::UploadPressed)
        .padding(10);

        let analysis_button = button(text(&self.analysis_button_label))
        .on_press(Message::AnalysisPressed)
        .padding(10);

        let refactor_button = button(text(&self.refactor_button_label))
        .on_press(Message::RefactorPressed)
        .padding(10);

        let save_button = button(text(&self.save_button_label))
        .on_press(Message::SavePressed)
        .padding(10);

        let clear_button = button(text(&self.clear_button_label))
        .on_press(Message::ClearPressed)
        .padding(10);

        let results_scrollable = scrollable(
            text(&self.analysis_results)
                    .size(16)
                    )
                    .height(400.0)
                    .width(1600.0);
        
        // adding the CFG space

        let button_row = row![
            upload_button,
            analysis_button,
            refactor_button,
            save_button,
            clear_button,
        ]
        .spacing(10);

        // add result row

        let layout = container(
            column![
                input,
                button_row,
                results_scrollable, // replace with result row
            ]
            .spacing(10),
        )
        .padding(10);

        layout.into()
    }
}

