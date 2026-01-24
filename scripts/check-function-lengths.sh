#!/bin/bash

# Engineering Standards Function Length Checker
# Enforces ≤25 lines per function in src-tauri/src/
# Fails with error if violations found

set -e

STANDARD_MAX_LINES=25
VIOLATIONS=0
ERROR_OUTPUT=""

echo "Checking function lengths (max: $STANDARD_MAX_LINES lines)..."
echo ""

# Check all Rust files in src-tauri/src
while IFS= read -r file; do
    # Extract functions and their line counts
    # This is a simple parser that looks for "fn " and counts lines until next "fn " or "}"
    
    current_fn=""
    fn_start_line=0
    fn_line_count=0
    in_fn=false
    brace_count=0
    
    while IFS= read -r line; do
        line_num=$((line_num + 1))
        
        # Check if line contains function definition
        if [[ $line =~ ^[[:space:]]*(pub[[:space:]]*)?(async[[:space:]]*)?(fn[[:space:]]+[a-zA-Z_][a-zA-Z0-9_]*) ]]; then
            # If we were in a function, check its length
            if [ "$in_fn" = true ] && [ "$fn_line_count" -gt "$STANDARD_MAX_LINES" ]; then
                VIOLATIONS=$((VIOLATIONS + 1))
                ERROR_OUTPUT="$ERROR_OUTPUT$file:$fn_start_line: $current_fn - $fn_line_count lines (exceeds max of $STANDARD_MAX_LINES)\n"
            fi
            
            # Start new function
            current_fn=$(echo "$line" | grep -oP 'fn[[:space:]]+\K[a-zA-Z_][a-zA-Z0-9_]*')
            fn_start_line=$line_num
            fn_line_count=0
            in_fn=true
            brace_count=0
        fi
        
        if [ "$in_fn" = true ]; then
            fn_line_count=$((fn_line_count + 1))
            
            # Count braces to determine when function ends
            open_braces=$(echo "$line" | grep -o '{' | wc -l)
            close_braces=$(echo "$line" | grep -o '}' | wc -l)
            brace_count=$((brace_count + open_braces - close_braces))
            
            # Function ends when brace_count returns to 0
            if [ "$brace_count" -le 0 ] && [ "$open_braces" -gt 0 ]; then
                if [ "$fn_line_count" -gt "$STANDARD_MAX_LINES" ]; then
                    VIOLATIONS=$((VIOLATIONS + 1))
                    ERROR_OUTPUT="$ERROR_OUTPUT$file:$fn_start_line: $current_fn - $fn_line_count lines (exceeds max of $STANDARD_MAX_LINES)\n"
                fi
                in_fn=false
            fi
        fi
    done < "$file"
done < <(find src-tauri/src -name "*.rs" -type f ! -path "*/tests/*")

if [ "$VIOLATIONS" -gt 0 ]; then
    echo "❌ ENGINEERING STANDARD VIOLATION: Function length limit exceeded"
    echo ""
    echo -e "$ERROR_OUTPUT"
    echo "Total violations: $VIOLATIONS"
    echo ""
    echo "Fix: Refactor large functions using extraction and composition"
    exit 1
else
    echo "✅ All functions comply with 25-line limit"
    echo ""
    exit 0
fi
