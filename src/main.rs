
// https://youtu.be/gcBJ7cPSALo?si=GcoX1i3tHK_UxPvK&t=1122
// https://docs.rs/iced/latest/iced/index.html
// https://redandgreen.co.uk/iced-rs-example-snippets-version-0-13/rust-programming/
// Starting from Editor, then building Code Smell Detector
//Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// Due Date:
// CSPC5260 WQ25 Version 0.0
// main.rs
// purpose is building Code Smell Detector to do....

use iced::widget::{button, column, row, container, text_editor, text};
use iced::{application, Element};

mod analyzer;

fn main() -> Result<(), iced::Error> {

    // call analyzer
    analyzer::test_analyzer();

    // run application
    application(Editor::title, Editor::update, Editor::view)
    .run_with(|| (Editor::new(), iced::Task::none()))
}

struct Editor{
    content: text_editor::Content,
    upload_button_label: String,
    analysis_button_label: String,
    save_button_label: String,
}

#[derive(Debug, Clone)]
enum Message {
    Edit(text_editor::Action),
    UploadPressed,
    AnalysisPressed,
    SavePressed,
} 

impl Editor {
    fn new() -> Self {
        Self{
            content: text_editor::Content::with_text(include_str!("main.rs")),
            upload_button_label: String::from("Upload Code"),
            analysis_button_label: String::from("Analysis"),
            save_button_label: String::from("Save")
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
                self.upload_button_label = String::from("uploaded")
            }

            Message::AnalysisPressed => {
                self.analysis_button_label = String::from("Started Analysis")
            }
            Message::SavePressed => {
                self.save_button_label = String::from("Save Done")
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

        let button_row = row![
            upload_button,
            analysis_button,
            save_button,
        ]
        .spacing(10);

        let layout = container(
            column![
                input,
                button_row,
            ]
            .spacing(10),
        )
        .padding(10);

        layout.into()
    }
}