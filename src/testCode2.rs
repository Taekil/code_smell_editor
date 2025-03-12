// testCode2.rs
// SUM (Semantic Duplicates)
// Function 0: Loop-based summation
fn sum_of_numbers_loop(n: u32) -> u32 {
    let mut accum = 0;
    for counter in 1..=n {
        accum += counter;
    }
    accum
}

// Function 1: Mathematical formula
fn sum_of_numbers_formula(n: u32) -> u32 {
    // Using a closed-form expression
    (n * (n + 1)) / 2
}

// Function 2: Using fold for accumulation
fn compute_sum_fold(n: u32) -> u32 {
    (1..=n).fold(0, |acc, x| acc + x)
}

// FIND (Semantic Duplicates)

// Function 3: Operating on a Vec
fn find_element_vec(collection: &Vec<i32>, goal: i32) -> bool {
    for &elem in collection {
        if elem == goal {
            return true;
        }
    }
    false
}

// Function 4: Using a HashSet
fn find_element_hashset(collection: &std::collections::HashSet<i32>, goal: i32) -> bool {
    collection.contains(&goal)
}

// Function 5: Search in a slice
fn search_in_slice(slice: &[i32], search_value: i32) -> bool {
    slice.iter().any(|&x| x == search_value)
}
