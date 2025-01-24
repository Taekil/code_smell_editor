
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

use iced::widget::{button, column, container, text_editor, text};
use iced::{application, Element};

mod analyzer;

fn main() -> Result<(), iced::Error>{

    // call analyzer
    analyzer::test_analyzer();

    // run application
    application(Editor::title, Editor::update, Editor::view)
    .run_with(|| (Editor::new(), iced::Task::none()))
}

struct Editor{
    content: text_editor::Content,
    button_label: String,
}

#[derive(Debug, Clone)]
enum Message {
    Edit(text_editor::Action),
    ButtonPressed,
} 

impl Editor {
    fn new() -> Self {
        Self{
            content: text_editor::Content::with_text(include_str!("main.rs")),
            button_label: String::from("Analyze Code")
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
            Message::ButtonPressed => {
                self.button_label = String::from("Analysis Started") 
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
        .on_action(Message::Edit);

        let analyze_button = button(text(&self.button_label))
        .on_press(Message::ButtonPressed)
        .padding(10);

        let layout = container(
            column![
                input,
                analyze_button
            ]
            .spacing(10),
        )
        .padding(10);

        layout.into()
    }
}