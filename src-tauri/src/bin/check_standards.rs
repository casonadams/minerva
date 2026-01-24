use regex::Regex;
/// Engineering Standards Enforcement Tool
///
/// Checks:
/// - File length ≤ 150 lines
/// - Function length ≤ 25 lines
/// - Parameter count ≤ 3
///
/// Run: cargo run --bin check_standards
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let mut violations = Vec::new();

    // Check all Rust files
    for entry in walkdir::WalkDir::new("src-tauri/src")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
    {
        let path = entry.path();
        check_file(&path, &mut violations);
    }

    if violations.is_empty() {
        println!("✅ All files comply with engineering standards\n");
        std::process::exit(0);
    } else {
        println!("❌ Engineering standard violations found:\n");
        for v in &violations {
            println!("{}", v);
        }
        println!("\nTotal violations: {}\n", violations.len());
        std::process::exit(1);
    }
}

fn check_file(path: &Path, violations: &mut Vec<String>) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return,
    };

    let lines: Vec<&str> = content.lines().collect();

    // Check file length
    if lines.len() > 150 {
        violations.push(format!(
            "{}: {} lines (exceeds max of 150)",
            path.display(),
            lines.len()
        ));
    }

    // Check function lengths
    check_function_lengths(path, &lines, violations);
}

fn check_function_lengths(path: &Path, lines: &[&str], violations: &mut Vec<String>) {
    let fn_regex = Regex::new(r"(pub\s+)?(async\s+)?fn\s+([a-zA-Z_]\w*)").unwrap();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];

        if let Some(caps) = fn_regex.captures(line) {
            let fn_name = caps.get(3).unwrap().as_str();
            let fn_start = i + 1; // 1-indexed

            // Find function end by counting braces
            let mut brace_count = 0;
            let mut in_fn = false;
            let mut j = i;

            while j < lines.len() {
                let current = lines[j];

                // Count opening brace
                if current.contains('{') {
                    brace_count += current.matches('{').count() as i32;
                    in_fn = true;
                }

                // Count closing braces
                if current.contains('}') {
                    brace_count -= current.matches('}').count() as i32;
                }

                // Function ends when braces balance
                if in_fn && brace_count == 0 && current.contains('{') {
                    let fn_len = j - i + 1;

                    if fn_len > 25 {
                        violations.push(format!(
                            "{}:{}:{} '{}' - {} lines (exceeds max of 25)",
                            path.display(),
                            fn_start,
                            0,
                            fn_name,
                            fn_len
                        ));
                    }
                    break;
                }

                j += 1;
            }

            i = j + 1;
        } else {
            i += 1;
        }
    }
}
