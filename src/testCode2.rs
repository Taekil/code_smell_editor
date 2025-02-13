fn main () {
    let sentence = "Hello World!";
    println!("{}", sentence);
}

fn this_is_really_long_function_name_test_dummy (a: u32, b: u32) {
    let sum = a + b;
    println!("{}", sum);
}

fn this_is_really_long_function_name_test_dummy (a: u32, b: u32) {
    let sum = a + b;
    println!("{}", sum);
}

fn long_parameter_list_dummy (a: u32, b: u32, c:u32, d: u32, e: u32) {
    let sum = a + b + c + d + e;
    println!("{}", sum);
}

fn long_parameter_list_dummy_dup_semantic (a: u32, b: u32) {
    println!("{}", add(a, b));
}

fn semantic_while_case () {
    let a = [1, 2, 3, 4];
    let mut i = 0;
    while i < a.len() {
        println!("{}", a[i]);
        i += 1;
    }
}

fn semantic_for_case () {
    let a = [1, 2, 3, 4];
    // Iterate over each element in the array
    for element in a.iter() {
        println!("{}", element);
    }
}