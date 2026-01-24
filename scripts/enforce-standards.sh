#!/bin/bash

# Engineering Standards Enforcement Script
# Checks:
# - File length ≤ 150 lines
# - (Visual inspection needed for 25-line function limit - use reviewers)
#
# Usage: ./scripts/enforce-standards.sh

set -e

VIOLATIONS=0
MAX_FILE_LINES=150

echo "═══════════════════════════════════════════════════════════════"
echo "Engineering Standards Enforcement Check"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# =========================================================================
# CHECK 1: File Length Limit (≤150 lines)
# =========================================================================
echo "1️⃣  Checking file length limit (≤ $MAX_FILE_LINES lines)..."
echo ""

FILE_VIOLATIONS=0
while IFS= read -r file; do
    lines=$(wc -l < "$file")
    if [ "$lines" -gt "$MAX_FILE_LINES" ]; then
        FILE_VIOLATIONS=$((FILE_VIOLATIONS + 1))
        echo "  ❌ $file: $lines lines (exceeds $MAX_FILE_LINES)"
    fi
done < <(find src-tauri/src -name "*.rs" -type f)

if [ "$FILE_VIOLATIONS" -eq 0 ]; then
    echo "  ✅ All files within 150-line limit"
else
    echo ""
    echo "  ⚠️  $FILE_VIOLATIONS files exceed 150-line limit"
    VIOLATIONS=$((VIOLATIONS + FILE_VIOLATIONS))
fi

echo ""

# =========================================================================
# CHECK 2: Function Length Limit (≤25 lines) - Visual inspection guide
# =========================================================================
echo "2️⃣  Function length limit (≤ 25 lines) - Manual review needed"
echo ""
echo "  To find long functions, use:"
echo "  $ grep -n '^[[:space:]]*\(pub\|async\|fn\)' src-tauri/src/**/*.rs"
echo ""
echo "  Note: Automated detection is complex due to syntax variations."
echo "  Reviewers should check during code review."
echo ""

# =========================================================================
# CHECK 3: Summary
# =========================================================================
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ "$VIOLATIONS" -eq 0 ]; then
    echo "✅ PASS: All automated checks passed!"
    echo ""
    echo "Remaining manual checks:"
    echo "  - Function length (≤25 lines) - review during code review"
    echo "  - Cyclomatic complexity (≤3) - clippy warns on high complexity"
    echo "  - Parameter count (≤3) - clippy enforces (too-many-arguments)"
    echo ""
    exit 0
else
    echo "❌ FAIL: $VIOLATIONS violations found"
    echo ""
    echo "Next steps:"
    echo "  1. Review files listed above"
    echo "  2. Split large files into focused modules"
    echo "  3. Follow Phase 10 refactoring pattern"
    echo ""
    exit 1
fi
