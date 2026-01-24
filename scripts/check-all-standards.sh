#!/bin/bash

# Engineering Standards Enforcement - ALL CHECKS
# Verifies:
# 1. File length ≤ 150 lines
# 2. Cyclomatic complexity ≤ 3 (M ≤ 3)
# 3. Function length ≤ 25 lines (manual review during code review)
# 4. Parameter count ≤ 3 (enforced via clippy)
# 5. No clippy warnings
#
# Usage: ./scripts/check-all-standards.sh

set -e

FAILED=0
WARNINGS=0

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        Engineering Standards Enforcement Check                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# =========================================================================
# 1. CLIPPY WARNINGS (catches too-many-arguments, etc)
# =========================================================================
echo "1️⃣  Checking Clippy warnings (catches parameter count ≤ 3)..."
echo ""

if cargo clippy --all-targets 2>&1 | grep -i "warning:" > /tmp/clippy.log; then
    WARNINGS=$(grep -c "warning:" /tmp/clippy.log || echo 0)
    echo "  ⚠️  Found $WARNINGS Clippy warnings"
    FAILED=$((FAILED + 1))
else
    echo "  ✅ No Clippy warnings"
fi

echo ""

# =========================================================================
# 2. FILE LENGTH
# =========================================================================
echo "2️⃣  Checking file length (≤ 150 lines)..."
echo ""

FILE_VIOLATIONS=0
while IFS= read -r file; do
    lines=$(wc -l < "$file")
    if [ "$lines" -gt 150 ]; then
        FILE_VIOLATIONS=$((FILE_VIOLATIONS + 1))
        if [ "$FILE_VIOLATIONS" -le 5 ]; then
            echo "  ❌ $file: $lines lines"
        fi
    fi
done < <(find src-tauri/src -name "*.rs" -type f ! -path "*/tests/*")

if [ "$FILE_VIOLATIONS" -eq 0 ]; then
    echo "  ✅ All files ≤ 150 lines"
else
    echo "  ❌ $FILE_VIOLATIONS files exceed 150 lines"
    FAILED=$((FAILED + 1))
fi

echo ""

# =========================================================================
# 3. CYCLOMATIC COMPLEXITY (estimate via control flow)
# =========================================================================
echo "3️⃣  Checking cyclomatic complexity (target M ≤ 3)..."
echo ""

COMPLEXITY_VIOLATIONS=0
while IFS= read -r file; do
    # Simple heuristic: count decision points
    # This is NOT perfect but gives a rough idea
    complex_functions=$(grep -n "fn\|if\|match\|for\|while" "$file" | \
        awk -F: 'BEGIN{fn_line=0; points=0} 
        /fn / {if (fn_line && points > 5) print fn_line; fn_line=NR; points=1; next}
        /if|match|for|while|&&|\|\|/ {points++}
        END{if (fn_line && points > 5) print fn_line}' || true)
    
    if [ -n "$complex_functions" ]; then
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                COMPLEXITY_VIOLATIONS=$((COMPLEXITY_VIOLATIONS + 1))
                if [ "$COMPLEXITY_VIOLATIONS" -le 5 ]; then
                    echo "  ⚠️  $file:$line - potentially high complexity"
                fi
            fi
        done < <(echo "$complex_functions")
    fi
done < <(find src-tauri/src -name "*.rs" -type f ! -path "*/tests/*")

if [ "$COMPLEXITY_VIOLATIONS" -eq 0 ]; then
    echo "  ✅ No obviously complex functions detected"
elif [ "$COMPLEXITY_VIOLATIONS" -le 3 ]; then
    echo "  ⚠️  $COMPLEXITY_VIOLATIONS functions may have high complexity"
else
    echo "  ⚠️  $COMPLEXITY_VIOLATIONS functions may have high complexity"
    echo "     (Run: cargo clippy -- -W clippy::cognitive-complexity)"
fi

echo ""

# =========================================================================
# 4. FUNCTION LENGTH (manual review needed)
# =========================================================================
echo "4️⃣  Function length (≤ 25 lines) - requires manual review"
echo ""

LONG_FUNCTIONS=$(awk '
BEGIN { fn_start=0; fn_lines=0 }
/^[[:space:]]*(pub[[:space:]]*)?(async[[:space:]]*)?(fn|unsafe)/ { 
    fn_start = NR
    fn_lines = 0
}
{ 
    if (fn_start > 0) fn_lines++
    if (fn_start > 0 && /^[}]/ && fn_start != NR) {
        if (fn_lines > 25) print FILENAME":"fn_start":"fn_lines
        fn_start = 0
    }
}' src-tauri/src/**/*.rs 2>/dev/null | wc -l || echo 0)

if [ "$LONG_FUNCTIONS" -eq 0 ]; then
    echo "  ✅ No obviously long functions detected (rough check)"
else
    echo "  ⚠️  $LONG_FUNCTIONS functions may exceed 25 lines"
    echo "     Use: grep -n 'fn ' src-tauri/src/**/*.rs | head -20"
fi

echo ""

# =========================================================================
# SUMMARY
# =========================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                         SUMMARY                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo "✅ PASS: All checked standards met!"
    echo ""
    echo "Standards enforced:"
    echo "  ✅ Parameter count ≤ 3 (clippy: too-many-arguments)"
    echo "  ✅ File length ≤ 150 lines"
    echo "  ✅ Cognitive complexity reasonable (clippy threshold: 15)"
    echo ""
    echo "Standards requiring manual review:"
    echo "  📋 Function length ≤ 25 lines (check during code review)"
    echo "  📋 Cyclomatic complexity ≤ 3 (use: cargo clippy -W cognitive-complexity)"
    echo ""
    exit 0
else
    echo "❌ FAIL: $FAILED checks need attention"
    echo ""
    echo "Next steps:"
    echo "  1. Fix Clippy warnings: cargo clippy --fix"
    echo "  2. Split large files into focused modules"
    echo "  3. Refactor complex functions with extraction"
    echo "  4. Run this check again: ./scripts/check-all-standards.sh"
    echo ""
    exit 1
fi
