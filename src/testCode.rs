fn main () {
    let sentence = "Hello World!";
    println!("{}", sentence);
}

fn long_method_over_15_lines(code: &str) -> bool {
    let line_count = code.lines().count();
    let long_function_threshold = 50;

    if line_count > long_function_threshold {
        println!("Code smell: Function is too long with {} lines", line_count);
        return true;
    }

    let mut line_frequency = std::collections::HashMap::new();
    for line in code.lines() {
        let trimmed_line = line.trim();
        if trimmed_line.is_empty() {
            continue;
        }
        *line_frequency.entry(trimmed_line).or_insert(0) += 1;
    }

    let duplicate_threshold = 3;
    let mut smell_found = false;
    for (line, count) in line_frequency.iter() {
        if *count > duplicate_threshold {
            println!("Code smell: Line '{}' is repeated {} times", line, count);
            smell_found = true;
        }
    }

    if smell_found {
        println!("Code smell detected due to repetition of code lines.");
    } else {
        println!("No significant code smell detected.");
    }

    smell_found
}

fn dummy_test_long_method() {
    println!("Test Dummby for Long Method with Blanks")
















}

fn this_is_really_long_function_name_test_dummy (a: u32, b: u32) {
    let sum = a + b;
    println!("{}", sum);
}

fn long_parameter_list_dummy (a: u32, b: u32, c:u32, d: u32, e: u32) {
    let sum = a + b;
    println!("{}", sum);
}

fn long_parameter_list_dummy_dup (a: u32, b: u32, c:u32, d: u32, e: u32) {
    let sum = a + b;
    println!("{}", sum);
}

fn long_parameter_list_dummy_dup_semantic (a: u32, b: u32, c:u32, d: u32, e: u32) {
    let sum = a + c;
    println!("{}", sum);
}