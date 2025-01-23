// https://youtu.be/gcBJ7cPSALo?si=R3Wej4oPVgy4pY_U&t=657
// Starting from Editor, then building Code Smell Detector
//Taekil Oh
// Start Date: Jan 23rd 2025
// Update Date:
// Due Date:
// CSPC5260 WQ25 Version 0.0
// main.rs
// purpose is building Code Smell Detector to do....

use iced::widget::text;
use iced::{Element, Sandbox, Settings};

mod analyzer;

fn main() -> iced::Result{

    // call analyzer
    analyzer::test_analyzer();
    // Editor::run method comes from the Sandbox trait. 
    Editor::run(Settings::default())
}

struct Editor;

#[derive(Debug)]
enum Message {}

impl Sandbox for Editor {
    /*
        Sandbox trait
     */
    type Message = Message;

    fn new() -> Self {
        Self
    }

    fn title(&self) -> String {
        String::from("Code Smell Detector")
    }

    fn update(&mut self, message: Message) {
        match message {
            
        }
    }

    fn view(&self) -> Element<'_, Message> {
        /*
            Describes what the app looks like
            use iced::widget::text to display the message
            Wrap the text in an Element (a containder for UI Compoenents)
         */
        text("This is Code Smell Detector for CSPC5260").into()
    }
}