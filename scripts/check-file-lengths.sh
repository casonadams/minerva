#!/bin/bash

# Engineering Standards File Length Checker
# Enforces ≤100 lines per file in src-tauri/src/
# Fails with error if violations found

set -e

STANDARD_MAX_LINES=150
VIOLATIONS=0
ERROR_OUTPUT=""

echo "Checking file lengths (max: $STANDARD_MAX_LINES lines)..."
echo ""

# Check all Rust files in src-tauri/src
while IFS= read -r file; do
    lines=$(wc -l < "$file")
    
    if [ "$lines" -gt "$STANDARD_MAX_LINES" ]; then
        VIOLATIONS=$((VIOLATIONS + 1))
        ERROR_OUTPUT="$ERROR_OUTPUT$file: $lines lines (exceeds max of $STANDARD_MAX_LINES)\n"
    fi
done < <(find src-tauri/src -name "*.rs" -type f)

if [ "$VIOLATIONS" -gt 0 ]; then
    echo "❌ ENGINEERING STANDARD VIOLATION: File size limit exceeded"
    echo ""
    echo -e "$ERROR_OUTPUT"
    echo "Total violations: $VIOLATIONS"
    echo ""
    echo "Fix: Split large files into focused modules (see Phase 10 refactoring)"
    exit 1
else
    echo "✅ All files comply with 100-line limit"
    echo ""
    exit 0
fi
