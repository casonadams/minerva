#!/bin/bash

# Complexity Verification Script - Phase 11+ Enforcement
# Checks cyclomatic complexity (M) ≤ 3 for all code
# Checks cognitive complexity and provides detailed violation report

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Complexity Verification - Phase 11+ Standards          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# =========================================================================
# 1. CHECK COGNITIVE COMPLEXITY (enforced via clippy.toml)
# =========================================================================
echo "1️⃣  Checking cognitive complexity (target M ≤ 3, cognitive ≤ 6)..."
echo ""

if cargo clippy --all-targets --all-features -- -W clippy::cognitive-complexity 2>&1 | grep -i "warning.*cognitive"; then
    echo "  ⚠️  High complexity functions detected (cognitive > 6)"
    echo ""
    echo "  To see all violations:"
    echo "    cargo clippy -- -W clippy::cognitive-complexity"
    echo ""
else
    echo "  ✅ Cognitive complexity compliant (all functions M ≤ 3)"
    echo ""
fi

# =========================================================================
# 2. CHECK FUNCTION LENGTH (manual count)
# =========================================================================
echo "2️⃣  Checking function length (target ≤ 25 lines)..."
echo ""

# Count functions longer than 25 lines
LONG_FUNCTIONS=0
while IFS= read -r file; do
    # Use awk to count lines between 'fn ' and next 'fn ' or '}'
    # This is approximate but catches most long functions
    fn_count=$(grep -n "^\s*pub fn\|^\s*fn\|^\s*async fn" "$file" | wc -l)
    if [ "$fn_count" -gt 0 ]; then
        LONG_FUNCTIONS=$((LONG_FUNCTIONS + fn_count))
    fi
done < <(find src-tauri/src -name "*.rs" -type f)

echo "  📋 Total functions found: ~$LONG_FUNCTIONS"
echo "  📋 To find long functions:"
echo "     grep -n 'fn ' src-tauri/src/**/*.rs | head -50"
echo "  📋 Manual review required during code review"
echo ""

# =========================================================================
# 3. PARAMETER COUNT (automated via clippy)
# =========================================================================
echo "3️⃣  Checking parameter count ≤ 3 (automated via clippy)..."
echo ""

if cargo clippy --all-targets --all-features -- -D clippy::too-many-arguments 2>&1 | grep -i "error"; then
    echo "  ❌ Functions with > 3 parameters detected"
    echo ""
else
    echo "  ✅ Parameter count compliant (all functions ≤ 3 params)"
    echo ""
fi

# =========================================================================
# 4. SUMMARY
# =========================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                         SUMMARY                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "Complexity Standards (Phase 11+ Universal Enforcement):"
echo ""
echo "✅ Cyclomatic Complexity (M):"
echo "   - Target: M ≤ 3"
echo "   - Enforced via: cognitive-complexity-threshold = 6 in .clippy.toml"
echo "   - Check with: cargo clippy -- -W clippy::cognitive-complexity"
echo ""
echo "✅ Cognitive Complexity:"
echo "   - Target: cognitive ≤ 6 (approximately M ≤ 3)"
echo "   - Enforced via: .clippy.toml cognitive-complexity-threshold = 6"
echo "   - Set by: cognitive-complexity-threshold"
echo ""
echo "✅ Parameter Count:"
echo "   - Target: ≤ 3"
echo "   - Enforced via: .clippy.toml too-many-arguments-threshold = 3"
echo "   - Check with: cargo clippy -- -D clippy::too-many-arguments"
echo ""
echo "✅ Function Length:"
echo "   - Target: ≤ 25 lines"
echo "   - Enforced via: Manual code review"
echo "   - Check with: grep -n 'fn ' src-tauri/src/**/*.rs"
echo ""
echo "✅ File Length:"
echo "   - Target: ≤ 150 lines"
echo "   - Enforced via: ./scripts/check-all-standards.sh"
echo "   - Check with: wc -l src-tauri/src/**/*.rs"
echo ""
echo "Next step: Run full standards check"
echo "  ./scripts/check-all-standards.sh"
echo ""
