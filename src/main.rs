
// https://youtu.be/gcBJ7cPSALo?si=GcoX1i3tHK_UxPvK&t=1122
// https://docs.rs/iced/latest/iced/index.html
// Starting from Editor, then building Code Smell Detector
//Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// Due Date:
// CSPC5260 WQ25 Version 0.0
// main.rs
// purpose is building Code Smell Detector to do....

use iced::widget::{container, text_editor};
use iced::{Theme, Element, Sandbox, Settings};

mod analyzer;

fn main() -> iced::Result{

    // call analyzer
    analyzer::test_analyzer();
    // Editor::run method comes from the Sandbox trait. 
    Editor::run(Settings::default())
}

struct Editor{
    content: text_editor::Content,
}

#[derive(Debug, Clone)]
enum Message {
    Edit(text_editor::Action),
} 

impl Sandbox for Editor {
    /*
        Sandbox trait
     */
    type Message = Message;

    fn new() -> Self {
        Self{
            content: text_editor::Content::with_text(include_str!("main.rs")),
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
        }
    }

    fn view(&self) -> Element<'_, Message> {
        /*
            Describes what the app looks like
            use iced::widget::text to display the message
            Wrap the text in an Element (a containder for UI Compoenents)
         */
        let input = text_editor(&self.content).on_action(Message::Edit);

        container(input).padding(10).into()
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }

}