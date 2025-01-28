// Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// 1st due Date:
// final due date:  
// CSPC5260 WQ25 Version 0.0
// main.rs
// Code Smell Detector, the purpose of main.rs 

use iced::widget::{button, column, container, row, scrollable, text, text_editor};
use iced::{application, Element};
use rfd::FileDialog;

mod analyzer;

fn main() -> Result<(), iced::Error> {

    // call analyzer
    analyzer::test_analyzer();

    // run application
    application(CodeSmellDetector::title, CodeSmellDetector::update, CodeSmellDetector::view)
    .run_with(|| (CodeSmellDetector::new(), iced::Task::none()))
}

struct CodeSmellDetector {
    content: text_editor::Content,
    upload_button_label: String,
    analysis_button_label: String,
    save_button_label: String,
    clear_button_label: String,
    analysis_results: String, //Store results here
}

#[derive(Debug, Clone)]
enum Message {
    Edit(text_editor::Action),
    UploadPressed,
    AnalysisPressed,
    SavePressed,
    ClearPressed,
} 

impl CodeSmellDetector {
    fn new() -> Self {
        Self{
            content: text_editor::Content::default(),
            upload_button_label: String::from("Upload Code"),
            analysis_button_label: String::from("Analysis"),
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
                if let Some(file) = FileDialog::new().pick_file() {
                    let file_path = file.display().to_string();

                    match std::fs::read_to_string(&file_path) {
                        Ok(content) => {
                            self.content = text_editor::Content::with_text(&content);
                            self.upload_button_label = String::from("uploaded");
                        }
                        Err(_) => {
                            self.upload_button_label = String::from("failed")
                        }
                    }
                }
                else {
                    self.upload_button_label = String::from("No File")
                }
            }

            Message::AnalysisPressed => {
                self.analysis_button_label = String::from("Started Analysis");

                // call analyzer
                let code_to_analyze = self.content.text();
                // make call analayzer(required to build the anlayzing algorithms...)
                let results = code_to_analyze;
                self.analysis_results = results; // result already string(test case)
            }

            Message::SavePressed => {

                let combined_content = format!(
                    "==== CODE ===\n{}\n\n==== ANALYSIS RESULTS ==== \n{}",
                    self.content.text(),
                    self.analysis_results,
                );

                if let Some(file) = FileDialog::new().save_file() {
                    let file_path = file.display().to_string();
                    match std::fs::write(&file_path, &combined_content) {
                        Ok(_) => {
                            self.analysis_results = String::from("File Saved Successfully!");
                        }
                        Err(err) => {
                            self.analysis_results = format!("Save Failed: {}", err);
                        }
                    }
                }
                else {
                    self.analysis_results = String::from("No file Selected for Saving");
                }
            }

            Message::ClearPressed => {
                // Reset all fields
                self.content = text_editor::Content::default();
                self.upload_button_label = String::from("Upload Code");
                self.analysis_button_label = String::from("Analysis");
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
            save_button,
            clear_button,
        ]
        .spacing(10);
        
        let layout = container(
            column![
                input,
                button_row,
                results_scrollable,
            ]
            .spacing(10),
        )
        .padding(10);

        layout.into()
    }
}