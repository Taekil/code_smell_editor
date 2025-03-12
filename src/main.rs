// Taekil Oh
// Start Date: Jan 23rd 2025
// CSPC5260 WQ25 Version 1.0
// main.rs

use iced::widget::{button, column, container, row, scrollable, text, text_editor};
use iced::{application, Element};

mod code_analyzer;
mod file_manager;
mod semantic_detector;

use file_manager::FileManager;
use code_analyzer::CodeAnalyzer;
use semantic_detector::SemanticDetector;

fn main() -> Result<(), iced::Error> {

    // run application
    application(CodeSmellDetector::title, CodeSmellDetector::update, CodeSmellDetector::view)
    .run_with(|| (CodeSmellDetector::new(), iced::Task::none()))
}

struct CodeSmellDetector {
    file_manager: FileManager,
    code_analizer: CodeAnalyzer,
    semantic_detector: SemanticDetector,
    content: text_editor::Content,
    upload_button_label: String,
    analysis_button_label: String,
    semantic_button_label: String, 
    refactor_button_label:String,
    save_button_label: String,
    clear_button_label: String,
    analysis_results: String,
}

#[derive(Debug, Clone)]
enum Message {
    Edit(text_editor::Action),
    UploadPressed,
    AnalysisPressed,
    SemanticPressed,
    RefactorPressed,
    SavePressed,
    ClearPressed,
} 

impl CodeSmellDetector {
    fn new() -> Self {
        Self{
            file_manager: FileManager::new(),
            code_analizer: CodeAnalyzer::new(),
            semantic_detector: SemanticDetector::new(),
            content: text_editor::Content::default(),
            upload_button_label: String::from("Upload Code"),
            analysis_button_label: String::from("Analysis with Jaccard Matrics"),
            semantic_button_label: String::from("Semantic Similarity"),
            refactor_button_label: String::from("Refactoring"),
            save_button_label: String::from("Save"),
            clear_button_label: String::from("Clear"),
            
            analysis_results: String::from(""),
        }
    }

    fn title(&self) -> String {
        String::from("CODE SMELL DETECTOR")
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
                            // self.upload_button_label = String::from("uploaded");

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

                // self.analysis_button_label = String::from("Started Analysis");
                let recent_code = self.content.text();
                let _= self.code_analizer.set_ast_content(recent_code);
                self.analysis_results = self.code_analizer.get_analysis_result();
            }

            Message::SemanticPressed => {

                self.analysis_results.clear();
                let threshold = 0.7;
                let recent_code = self.content.text();
                let _ = self.semantic_detector.detect_duplicates(&recent_code, threshold);
                self.analysis_results = self.semantic_detector.get_result();
            }

            Message::RefactorPressed => {

                // self.refactor_button_label = String::from("Dup Refactored");
                let recent_code = self.code_analizer.refactored_by_jaccard_result();
                self.content = text_editor::Content::with_text(&recent_code);
                println!("Refactoring: Delete Duplicate function(s)")
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

                self.content = text_editor::Content::default();
                // self.upload_button_label = String::from("Upload Code");
                // self.analysis_button_label = String::from("Analysis");
                // self.refactor_button_label = String::from("Dup Refactor");
                // self.save_button_label = String::from("Save");
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

        let semantic_button = button(text(&self.semantic_button_label))
        .on_press(Message::SemanticPressed)
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
        
        let button_row = row![
            upload_button,
            analysis_button,
            semantic_button,
            refactor_button,
            save_button,
            clear_button,
        ]
        .spacing(10);

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

