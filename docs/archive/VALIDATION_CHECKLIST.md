# Pre-Commit Validation Checklist

## 🚨 REQUIRED: Run Before Every Commit

### Full Validation (Copy & Paste)
```bash
# Run this BEFORE committing
pnpm lint && pnpm test

# If either fails:
# 1. STOP - do not commit
# 2. Fix the issues
# 3. Run again until both pass
```

### Individual Checks
```bash
# Check backend linting
pnpm lint:backend

# Check backend tests
pnpm test:backend

# Check frontend linting  
pnpm lint:frontend

# Check frontend tests
pnpm test:frontend

# Check everything
pnpm check:all
```

---

## ✅ Success Criteria (Before Commit)

- [ ] `pnpm lint` returns 0 violations
- [ ] `pnpm test` shows 464+ tests passing
- [ ] No warnings in build output
- [ ] All tests completed with status: **PASSED**

---

## 🔍 Phase 7 Specific Checks

### Per-Step Validation

**Step 1: Logging & Tracing**
```bash
pnpm test:backend              # Tests must pass
pnpm lint:backend              # 0 violations
grep -r "tracing" src-tauri/   # Verify tracing crate added
```

**Step 2: Error Recovery**
```bash
pnpm test:backend              # Tests must pass
pnpm lint:backend              # 0 violations
grep -r "retry\|circuit" src-tauri/ # Verify new patterns
```

**Step 3: Metrics & Observability**
```bash
pnpm test:backend              # Tests must pass
pnpm lint:backend              # 0 violations
grep -r "prometheus\|metrics" src-tauri/ # Verify metrics added
```

**Step 4: Resource Management**
```bash
pnpm test:backend              # Tests must pass
pnpm lint:backend              # 0 violations
grep -r "cleanup\|shutdown" src-tauri/ # Verify cleanup added
```

**Step 5: Error Testing**
```bash
pnpm test:backend              # Tests must pass (464+ total)
pnpm lint:backend              # 0 violations
grep -r "chaos\|failure" tests/ # Verify chaos tests added
```

---

## 📊 Expected Test Count Progression

| Phase | Target Tests | Status |
|-------|--------------|--------|
| Start (Phase 7) | 464 | ✅ Current |
| Step 1 (Logging) | 472+ | 📈 +8 tests |
| Step 2 (Resilience) | 484+ | 📈 +12 tests |
| Step 3 (Metrics) | 494+ | 📈 +10 tests |
| Step 4 (Resources) | 504+ | 📈 +10 tests |
| Step 5 (Chaos) | 519+ | 📈 +15 tests |
| **Phase 7 Complete** | **519+** | ✅ Target |

---

## 🛑 If Validation Fails

### Lint Violations
```bash
# See what failed
pnpm lint:backend

# Fix issues (usually formatting)
pnpm fmt:backend

# Try again
pnpm lint:backend
```

### Test Failures
```bash
# See which tests failed
pnpm test:backend

# Run with backtrace
RUST_BACKTRACE=1 pnpm test:backend

# Fix the code
# Then try again
pnpm test:backend
```

### Multiple Failures
1. **Don't panic!** - This is normal during development
2. Read error messages carefully
3. Fix one issue at a time
4. Re-run validation after each fix
5. Commit only when ALL checks pass

---

## 🚀 Quick Commands Reference

```bash
# Full validation (REQUIRED before commit)
pnpm lint && pnpm test

# Just linting
pnpm lint

# Just tests
pnpm test

# Format code (fixes some lint issues)
pnpm fmt

# Full quality check
pnpm check:all

# Backend only
pnpm lint:backend && pnpm test:backend

# Frontend only
pnpm lint:frontend && pnpm test:frontend
```

---

## 📝 Commit Message Template

When committing Phase 7 work, use this format:

```
feat(phase7-step1): add structured logging infrastructure

Adds tracing-based logging system with:
- Structured log levels and formatting
- Request/span context propagation
- Async logging for performance
- Log aggregation points

Tests: 464 → 472 tests (+8)
Lint: ✅ 0 violations
```

---

## 🎯 Remember

> **"If it doesn't pass linting and tests, it doesn't ship."**

Always run validation before committing. No exceptions!
