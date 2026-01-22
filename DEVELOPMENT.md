# Minerva Development Guide

This guide covers development workflows, testing, linting, and contributing to Minerva.

## Quick Start

### Install Dependencies

```bash
# Frontend and Node dependencies
pnpm install

# Rust dependencies are handled automatically by Cargo
```

### Development Server

```bash
# Start development mode (frontend + backend)
pnpm tauri dev
```

This will:
1. Start the Vite dev server (frontend hot reload)
2. Compile and run the Tauri backend
3. Open the application window

## pnpm Scripts

### Frontend Development

```bash
pnpm dev              # Start Vite dev server
pnpm build            # Build frontend for production
pnpm preview          # Preview production build locally
```

### Backend Testing

```bash
pnpm test             # Run all Rust backend tests
pnpm test:watch       # Run tests with output
```

**Example Output:**
```
running 3 tests
test server::tests::test_health_check ... ok
test server::tests::test_list_models_empty ... ok
test server::tests::test_chat_completions_model_not_found ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Linting & Code Quality

```bash
pnpm lint             # Run Clippy linter (strict mode)
pnpm lint:fix         # Auto-fix Clippy warnings
pnpm fmt              # Format all Rust code
pnpm fmt:check        # Check code formatting without changes
```

### Comprehensive Checks

```bash
pnpm check            # Frontend type checking and Svelte validation
pnpm check:watch      # Frontend checks with watch mode
pnpm check:all        # Run all checks (frontend + backend lint + format)
pnpm check:backend    # Backend compilation check
```

### Building

```bash
pnpm build:backend           # Debug build
pnpm build:backend:release   # Optimized release build
```

## Workflow Examples

### Before Committing Code

Run the full validation suite:

```bash
pnpm check:all
pnpm test
```

This ensures:
- ✅ No Rust compiler errors
- ✅ No Clippy warnings
- ✅ Code is properly formatted
- ✅ All tests pass
- ✅ TypeScript is correct

### During Development

```bash
# In one terminal, start dev server
pnpm tauri dev

# In another terminal, watch tests
pnpm test:watch

# In another terminal, watch formatting
cd src-tauri && cargo fmt --all -- --check && cd ..
```

### Making a Release

```bash
# Clean build with all checks
pnpm check:all
pnpm test

# Build for production
pnpm tauri build --release
```

## Testing

### Running Tests

```bash
# Run all tests
pnpm test

# Run specific test file
cd src-tauri && cargo test --lib server -- --nocapture && cd ..

# Run with backtrace for debugging
cd src-tauri && RUST_BACKTRACE=1 cargo test --lib && cd ..
```

### Writing Tests

Tests are located in `src-tauri/src/` files at the bottom with `#[cfg(test)]` modules:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        assert_eq!(2 + 2, 4);
    }
}
```

**Test Naming Convention:**
- `test_<function>_<scenario>` - For unit tests
- `test_<feature>_<behavior>` - For integration tests

Example: `test_chat_completions_model_not_found`

### Test Coverage

Current test coverage:
- ✅ Health check endpoint
- ✅ Empty models list
- ✅ Model not found error
- ✅ Error response formatting

To add more tests, modify files in `src-tauri/src/`

## Linting & Code Quality

### Clippy (Rust Linter)

```bash
pnpm lint
```

Clippy checks for:
- Unused imports
- Inefficient code patterns
- Missing documentation
- Possible bugs
- Code style issues

### Code Formatting

```bash
# Format code
pnpm fmt

# Check without formatting
pnpm fmt:check
```

Uses `rustfmt` with project configuration in `rustfmt.toml`

### Frontend Type Checking

```bash
pnpm check
```

Checks:
- TypeScript compilation errors
- Svelte component errors
- Type mismatches

## Project Structure

```
minerva/
├── src/                          # Svelte frontend
│   ├── routes/
│   │   ├── +layout.ts
│   │   └── +page.svelte
│   ├── app.html
│   └── lib/
├── src-tauri/                    # Rust backend
│   ├── src/
│   │   ├── lib.rs               # Entry point
│   │   ├── server.rs            # HTTP server
│   │   ├── models.rs            # Data types
│   │   ├── error.rs             # Error handling
│   │   └── main.rs              # Binary entry
│   ├── Cargo.toml               # Rust deps
│   └── tauri.conf.json          # App config
├── package.json                 # pnpm scripts
├── tsconfig.json                # TypeScript config
├── vite.config.js               # Vite config
└── svelte.config.js             # Svelte config
```

## Debugging

### Backend Debugging

#### View Detailed Test Output

```bash
cd src-tauri && cargo test --lib -- --nocapture --test-threads=1 && cd ..
```

#### Enable Rust Backtrace

```bash
cd src-tauri && RUST_BACKTRACE=1 cargo test --lib && cd ..
```

#### Check Rust Version

```bash
rustc --version
cargo --version
```

### Frontend Debugging

Browser DevTools are available in dev mode:
- Open DevTools: `F12` or `Cmd+Option+I`
- View console logs
- Debug TypeScript
- Inspect Svelte components

## Git Workflow

### Before Pushing

1. Run all checks:
```bash
pnpm check:all
pnpm test
```

2. Format code:
```bash
pnpm fmt
```

3. Commit with conventional message:
```bash
git add .
git commit -m "feat: description of feature"
```

### Commit Message Format

Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting
- `test:` - Test changes
- `build:` - Build system
- `chore:` - Maintenance

Example:
```bash
git commit -m "feat(server): add support for streaming responses"
```

## Troubleshooting

### Tests Failing

```bash
# Clean and rebuild
cd src-tauri && cargo clean && cargo test --lib && cd ..

# With backtrace
cd src-tauri && RUST_BACKTRACE=full cargo test --lib && cd ..
```

### Port Already in Use

If port 11434 is in use:
```bash
# Find and kill process
lsof -i :11434
kill -9 <PID>

# Or use different port in config
```

### Compilation Issues

```bash
# Update Rust
rustup update

# Clean rebuild
cd src-tauri && cargo clean && cargo build && cd ..
```

### Frontend Type Errors

```bash
# Sync SvelteKit
pnpm svelte-kit sync

# Run checks
pnpm check
```

## Code Quality Standards

All code must meet these standards (enforced by CI):

- ✅ Compiles with zero errors
- ✅ Zero Clippy warnings
- ✅ Properly formatted (`cargo fmt`)
- ✅ All tests pass
- ✅ TypeScript strict mode
- ✅ Functions ≤ 25 lines
- ✅ Cyclomatic complexity ≤ 3
- ✅ SOLID principles followed

## Performance Tips

### Faster Compilation

```bash
# Incremental compilation (already default)
# Use SSD for development

# Check compilation time
cd src-tauri && cargo build --timings && cd ..
```

### Faster Tests

```bash
# Run tests in parallel (default)
cd src-tauri && cargo test --lib && cd ..

# Single-threaded for debugging
cd src-tauri && cargo test --lib -- --test-threads=1 && cd ..
```

## Environment Setup

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Node/pnpm
brew install node pnpm
```

### Check Setup

```bash
rustc --version      # Should be 1.70+
cargo --version      # Should be 1.70+
node --version       # Should be 18+
pnpm --version       # Should be 8+
```

## Additional Resources

- [Tauri Documentation](https://tauri.app/)
- [Cargo Documentation](https://doc.rust-lang.org/cargo/)
- [Clippy Lints](https://docs.rs/clippy/)
- [Rustfmt Configuration](https://rust-lang.github.io/rustfmt/)
- [SvelteKit Documentation](https://kit.svelte.dev/)

## Getting Help

- Check existing issues
- Review documentation
- Ask in community forums
- Create detailed bug reports

---

**Last Updated:** January 2026
**Minerva Version:** 0.1.0
