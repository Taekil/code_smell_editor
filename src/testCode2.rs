// Semantic Duplicate Example 1: Different Algorithms, Same Result
// Function 1: Using a loop
fn sum_of_numbers_loop(n: u32) -> u32 {
    let mut sum = 0;
    for i in 1..=n {
        sum += i;
    }
    sum
}

// Function 2: Using a mathematical formula (more efficient)
fn sum_of_numbers_formula(n: u32) -> u32 {
    (n * (n + 1)) / 2
}

// Semantic Duplicate Example 2: Different Data Structures, Same Operation
// Function 1: Operating on a Vec
fn find_element_vec(data: &Vec<i32>, target: i32) -> bool {
    for &item in data {
        if item == target {
            return true;
        }
    }
    false
}

// Function 2: Operating on a HashSet
fn find_element_hashset(data: &std::collections::HashSet<i32>, target: i32) -> bool {
    data.contains(&target)
}