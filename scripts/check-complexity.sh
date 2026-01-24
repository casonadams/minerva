#!/bin/bash

# Cyclomatic Complexity Checker
# Estimates cyclomatic complexity (M) for each function
# Cyclomatic complexity = 1 + (number of decision points)
# Decision points: if, else, match arms, loops, && (in conditions), || (in conditions)
#
# Target: M ≤ 3 (simple functions with ≤ 2 branches)
# Yellow: M = 4-5 (moderate, consider refactoring)
# Red: M ≥ 6 (complex, should refactor)

echo "═══════════════════════════════════════════════════════════════"
echo "Cyclomatic Complexity Analysis"
echo "Target: M ≤ 3 (Yellow ≥ 4, Red ≥ 6)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

VIOLATIONS=0

while IFS= read -r file; do
    # Extract functions and analyze complexity
    awk -v file="$file" '
    /^[[:space:]]*(pub[[:space:]]+)?(async[[:space:]]+)?(unsafe[[:space:]]+)?(fn|impl)/ {
        # Start of function/impl
        in_block = 1
        fn_name = $NF
        gsub(/\(.*/, "", fn_name)
        fn_line = NR
        complexity = 1  # Base complexity
        next
    }
    
    in_block {
        # Count decision points
        # if statements
        if ($0 ~ /if[[:space:]]+/) {
            complexity += gsub(/if[[:space:]]+/, "", $0)
        }
        # else if (already counted above)
        
        # match statements - each arm adds 1
        if ($0 ~ /match[[:space:]]+/) {
            in_match = 1
        }
        if (in_match && $0 ~ /=>/) {
            complexity++
        }
        
        # Loops (for, while, loop)
        complexity += gsub(/\b(for|while|loop)[[:space:]]+/, "", $0)
        
        # Logical operators in conditions (rough estimate)
        complexity += gsub(/&&/, "", $0)
        complexity += gsub(/\|\|/, "", $0)
        
        # Check for end of function (closing brace at start of line)
        if ($0 ~ /^}/) {
            if (fn_name != "" && fn_name != "impl") {
                status = "✅"
                if (complexity >= 4 && complexity <= 5) status = "🟡"
                if (complexity >= 6) status = "🔴"
                printf "%s %s::%s (line %d) - M=%d\n", status, file, fn_name, fn_line, complexity
                
                if (complexity > 5) violations++
            }
            in_block = 0
            fn_name = ""
            complexity = 1
            in_match = 0
        }
    }
    
    END {
        if (violations > 0) exit 1
    }
    ' "$file"
done < <(find src-tauri/src -name "*.rs" -type f)

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Severity levels:"
echo "  ✅ Green (M ≤ 3): Simple, easy to understand and test"
echo "  🟡 Yellow (M 4-5): Moderate, consider refactoring"
echo "  🔴 Red (M ≥ 6): Complex, should refactor with extraction"
echo ""
echo "To refactor high-complexity functions:"
echo "  1. Extract nested conditions into helper functions"
echo "  2. Use early returns to flatten nesting"
echo "  3. Replace match statements with trait objects/enums"
echo "  4. Use filters/maps instead of nested loops"
echo ""
