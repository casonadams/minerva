# Minerva pnpm Scripts Guide

All development tasks can be run with simple `pnpm` commands. Scripts handle both backend (Rust) and frontend (TypeScript/Svelte) automatically.

## Quick Reference

```bash
# Testing
pnpm test               # Run all tests
pnpm test:backend:watch # Watch mode tests

# Linting
pnpm lint               # Lint everything
pnpm lint:backend:fix   # Auto-fix warnings

# Formatting
pnpm fmt                # Format all code
pnpm fmt:check          # Check without changes

# Validation (Run Before Commit)
pnpm check:all
pnpm test

# Building
pnpm tauri dev          # Development mode
pnpm tauri build --release  # Production build
```

## All Available Scripts

### Testing

| Script | Purpose | Time |
|--------|---------|------|
| `pnpm test` | Run all backend tests | ~2s |
| `pnpm test:backend` | Explicit backend tests | ~2s |
| `pnpm test:backend:watch` | Watch tests with output | ~2s |
| `pnpm test:all` | All tests (currently backend only) | ~2s |

**Example:**
```bash
pnpm test
# Output:
# running 5 tests
# test server::tests::test_model_registry_empty ... ok
# test server::tests::test_model_registry_add_and_retrieve ... ok
# test server::tests::test_model_registry_remove ... ok
# test server::tests::test_list_models_endpoint ... ok
# test server::tests::test_chat_completions_model_not_found ... ok
```

### Linting & Code Quality

| Script | Purpose | Coverage | Time |
|--------|---------|----------|------|
| `pnpm lint` | Lint everything | Backend + Frontend | ~10s |
| `pnpm lint:backend` | Rust linting only | Clippy (strict) | ~1s |
| `pnpm lint:backend:fix` | Auto-fix warnings | Clippy | ~2s |
| `pnpm lint:frontend` | TypeScript/Svelte | Type checking | ~5s |

**Strict Mode:**
All Clippy warnings are treated as errors (`-D warnings`). This ensures high code quality.

### Code Formatting

| Script | Purpose | Coverage | Time |
|--------|---------|----------|------|
| `pnpm fmt` | Format all code | Rust + Frontend | ~2s |
| `pnpm fmt:backend` | Format Rust only | rustfmt | ~1s |
| `pnpm fmt:backend:check` | Check format | Rust only | ~1s |
| `pnpm fmt:check` | Check all format | All languages | ~2s |
| `pnpm fmt:frontend` | Placeholder | Ready for prettier | instant |

**Auto-formatting:**
```bash
pnpm fmt  # Automatically fixes formatting issues
```

### Comprehensive Checks

| Script | What It Checks | Components |
|--------|---|---|
| `pnpm check` | Frontend types | TypeScript + Svelte |
| `pnpm check:watch` | Frontend (watch) | With file watching |
| `pnpm check:frontend` | Frontend explicit | svelte-check + sync |
| `pnpm check:backend` | Backend compile | Cargo check |
| `pnpm check:all` | **FULL CHECK** | Frontend + Backend lint + Format check |

**Best Before Commit:**
```bash
pnpm check:all && pnpm test
```

### Building

| Script | Output | Mode |
|--------|--------|------|
| `pnpm build:backend` | Debug binary | Unoptimized |
| `pnpm build:backend:release` | Release binary | Optimized |
| `pnpm tauri dev` | Dev app | Hot reload |
| `pnpm tauri build --release` | Packaged app | Production |

### Development

| Script | Purpose |
|--------|---------|
| `pnpm dev` | Frontend dev server |
| `pnpm build` | Frontend production build |
| `pnpm preview` | Preview production build |
| `pnpm tauri` | Tauri CLI access |

## Recommended Workflows

### Daily Development

```bash
# Terminal 1: Development server
pnpm tauri dev

# Terminal 2: Watch tests
pnpm test:backend:watch

# Terminal 3: When you want to check something
pnpm lint:backend
pnpm fmt
```

### Before Committing

```bash
# Full validation
pnpm check:all
pnpm test

# If all pass, safe to commit
git add .
git commit -m "Your message"
```

### Creating a Release

```bash
# Full check
pnpm check:all
pnpm test

# Build for production
pnpm tauri build --release

# Artifacts in src-tauri/target/release/bundle/
```

### Quick Validation (30 seconds)

```bash
# Fast checks only
pnpm lint:backend
pnpm lint:frontend
pnpm fmt:backend:check
```

## Script Details

### How Backend Scripts Work

All backend-related scripts:
1. Change to `src-tauri` directory
2. Run the Cargo command
3. Return to project root

Example:
```bash
pnpm test
# Equivalent to:
# cd src-tauri && cargo test --lib && cd ..
```

### How Frontend Scripts Work

Frontend scripts use:
- **svelte-check** - Svelte component validation
- **TypeScript** - Type checking
- **SvelteKit sync** - Prepare routes and types

Example:
```bash
pnpm check:frontend
# Equivalent to:
# svelte-kit sync && svelte-check --tsconfig ./tsconfig.json
```

## Current Status

### Tests
```
✅ 5/5 tests passing
✅ Model registry tests
✅ Endpoint tests
✅ Error handling tests
```

### Linting
```
✅ 0 Clippy warnings (strict mode)
✅ 0 TypeScript errors
✅ 0 Svelte errors
```

### Code Quality
```
✅ Properly formatted
✅ SOLID principles
✅ All functions ≤ 25 lines
✅ Cyclomatic complexity ≤ 3
```

## Troubleshooting

### Tests Fail

```bash
# Clean rebuild
cd src-tauri && cargo clean && cargo test --lib && cd ..

# With backtrace
cd src-tauri && RUST_BACKTRACE=1 cargo test --lib && cd ..
```

### Linting Issues

```bash
# Try auto-fix
pnpm lint:backend:fix

# If that doesn't work, check specific issues
pnpm lint:backend
```

### Format Issues

```bash
# Auto-format everything
pnpm fmt

# Check what would change
pnpm fmt:backend:check
```

### Port Already in Use

```bash
# Find what's using port 11434
lsof -i :11434

# Kill it
kill -9 <PID>

# Or use different port in config
```

## Performance Tips

### Faster Compilation

```bash
# Incremental compilation (default)
# Already enabled in Cargo.toml

# Check compilation time
cd src-tauri && cargo build --timings && cd ..
```

### Faster Tests

```bash
# Parallel testing (default)
pnpm test

# Single-threaded (for debugging)
cd src-tauri && cargo test --lib -- --test-threads=1 && cd ..
```

### Incremental Checks

```bash
# Just lint backend (fastest)
pnpm lint:backend

# Just check frontend
pnpm check:frontend

# Just format check
pnpm fmt:backend:check
```

## Integration with CI/CD

Use these scripts in your CI pipeline:

```yaml
# Example GitHub Actions
- name: Check Code Quality
  run: pnpm check:all

- name: Run Tests
  run: pnpm test

- name: Build
  run: pnpm tauri build --release
```

## VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "editor.formatOnSave": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "editor.formatOnSave": true
  }
}
```

Then run:
```bash
pnpm fmt
```

## Shell Aliases (Optional)

Add to `.bashrc` or `.zshrc`:

```bash
alias mm-test="pnpm test"
alias mm-lint="pnpm lint"
alias mm-fmt="pnpm fmt"
alias mm-check="pnpm check:all"
alias mm-dev="pnpm tauri dev"
```

Then use:
```bash
mm-test      # pnpm test
mm-lint      # pnpm lint
mm-check     # pnpm check:all && pnpm test
mm-dev       # pnpm tauri dev
```

## Documentation

- Full development guide: See `DEVELOPMENT.md`
- Implementation plan: See `IMPLEMENTATION_PLAN.md`
- Architecture: See `README.md`

---

**Last Updated:** January 2026  
**Project:** Minerva - Local LLM Server  
**Status:** All scripts functional and tested ✅
