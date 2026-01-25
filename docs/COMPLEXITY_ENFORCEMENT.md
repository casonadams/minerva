# Cyclomatic Complexity Enforcement - Phase 11+

**Standard**: Cyclomatic Complexity M ≤ 3  
**Last Updated**: January 24, 2026  
**Enforcement**: Automated + Manual Review

## Overview

Cyclomatic complexity (M) measures the number of linearly independent paths through code. Lower complexity means:
- Easier to understand
- Easier to test (2^M test cases for thorough coverage)
- Fewer bugs
- Better maintainability

**Our Standard**: M ≤ 3 for all code (NO EXCEPTIONS)

## Understanding Complexity Metrics

### What is Cyclomatic Complexity?

Cyclomatic complexity counts decision paths:
- Each `if`, `match`, `loop`, `&&`, `||`, `?` operator: +1
- Nested branches multiply the paths: exponential growth

```
M = 1 (straight line):
fn simple() {
    let x = 1;
    println!("{}", x);
}

M = 2 (one branch):
fn one_branch(x: bool) {
    if x {
        println!("true");
    } else {
        println!("false");
    }
}

M = 3 (acceptable):
fn acceptable(x: i32) {
    match x {
        1 => println!("one"),
        2 => println!("two"),
        _ => println!("other"),
    }
}

M = 5 (too complex - VIOLATION):
fn too_complex(a: bool, b: bool, c: bool) {
    if a {
        if b {
            if c {
                println!("a && b && c");
            }
        }
    }
}
```

### Cognitive Complexity (Clippy)

Clippy uses "cognitive complexity" which is similar but more nuanced:

```
Formula: cognitive = base + (nesting_depth - 1)

Examples:
- Simple if: +1
- Nested if: +1 + nesting
- Match arm: +1
- Loop: +1
- Closure: +1
- Recursion: +1 + depth
```

**Our Standard**: cognitive ≤ 6 (approximately M ≤ 3)

## Enforcement Mechanisms

### 1. Clippy Configuration (.clippy.toml)

```toml
cognitive-complexity-threshold = 6  # Enforces M ≤ 3
```

This setting:
- Automatically flags complex functions during development
- Blocks CI/CD if violated
- Suggests refactoring approaches

### 2. Pre-Commit Hook

The `.git/hooks/pre-commit` script runs:
```bash
cargo clippy --all-targets -- -D warnings
```

This catches complexity violations BEFORE commits.

### 3. Automated Verification

```bash
# Check cognitive complexity
cargo clippy -- -W clippy::cognitive-complexity

# Or via npm
pnpm lint:complexity
```

### 4. Manual Code Review

During code review, check:
- Deep nesting (>2 levels)
- Multiple conditions (many `if`/`match` branches)
- Long functions with loops
- Complex state machines

## Refactoring Patterns (M ≤ 3)

### Pattern 1: Extract Validation (Reduce Nesting)

```rust
// ❌ BAD: M = 5 (deep nesting)
fn process_user(user: User) -> Result<()> {
    if user.is_active {
        if user.email_verified {
            if user.subscription_valid {
                if user.has_payment {
                    // ... complex logic
                }
            }
        }
    }
    Ok(())
}

// ✅ GOOD: M ≤ 3 (extracted validation)
fn process_user(user: User) -> Result<()> {
    validate_user(&user)?;
    process_active_user(&user)?;
    Ok(())
}

fn validate_user(user: &User) -> Result<()> {
    if !user.is_active { return Err("inactive") }
    if !user.email_verified { return Err("email") }
    if !user.subscription_valid { return Err("subscription") }
    if !user.has_payment { return Err("payment") }
    Ok(())
}

fn process_active_user(user: &User) -> Result<()> {
    // ... complex logic with M ≤ 3
    Ok(())
}
```

### Pattern 2: Extract Early Returns

```rust
// ❌ BAD: M = 5 (tangled logic)
fn handle_request(req: Request) -> Response {
    if req.is_valid() {
        if req.user.is_authenticated() {
            if req.user.has_permission() {
                if req.data.is_complete() {
                    process_request(req)
                } else {
                    error_response("incomplete")
                }
            } else {
                error_response("unauthorized")
            }
        } else {
            error_response("not authenticated")
        }
    } else {
        error_response("invalid request")
    }
}

// ✅ GOOD: M ≤ 3 (early returns)
fn handle_request(req: Request) -> Response {
    if !req.is_valid() {
        return error_response("invalid request");
    }
    if !req.user.is_authenticated() {
        return error_response("not authenticated");
    }
    if !req.user.has_permission() {
        return error_response("unauthorized");
    }
    if !req.data.is_complete() {
        return error_response("incomplete");
    }
    process_request(req)
}
```

### Pattern 3: Extract Loop Logic

```rust
// ❌ BAD: M = 6 (complex loop)
fn process_items(items: Vec<Item>) -> Result<Vec<Output>> {
    let mut results = Vec::new();
    for item in items {
        if item.is_valid {
            if item.passes_checks {
                if let Some(value) = item.extract_value() {
                    if value > 0 {
                        results.push(transform(value));
                    }
                }
            }
        }
    }
    Ok(results)
}

// ✅ GOOD: M ≤ 3 (extracted helper)
fn process_items(items: Vec<Item>) -> Result<Vec<Output>> {
    items.iter()
        .filter_map(should_process)
        .map(transform)
        .collect::<Result<_>>()
}

fn should_process(item: &Item) -> Option<i32> {
    if !item.is_valid { return None; }
    if !item.passes_checks { return None; }
    
    let value = item.extract_value()?;
    if value > 0 { Some(value) } else { None }
}
```

### Pattern 4: Extract Match Arms

```rust
// ❌ BAD: M = 7 (complex match)
fn handle_state(state: State) -> Result<()> {
    match state {
        State::Init => {
            if init_config()? {
                if load_resources()? {
                    setup_done()
                }
            }
        }
        State::Running => {
            if check_health()? {
                if !is_paused() {
                    continue_work()
                }
            }
        }
        State::Shutdown => {
            if flush_data()? {
                if cleanup()? {
                    shutdown_ok()
                }
            }
        }
    }
    Ok(())
}

// ✅ GOOD: M ≤ 3 (extracted handlers)
fn handle_state(state: State) -> Result<()> {
    match state {
        State::Init => handle_init(),
        State::Running => handle_running(),
        State::Shutdown => handle_shutdown(),
    }
}

fn handle_init() -> Result<()> {
    init_config()?;
    load_resources()?;
    setup_done();
    Ok(())
}

fn handle_running() -> Result<()> {
    check_health()?;
    if !is_paused() {
        continue_work();
    }
    Ok(())
}

fn handle_shutdown() -> Result<()> {
    flush_data()?;
    cleanup()?;
    shutdown_ok();
    Ok(())
}
```

## How to Maintain M ≤ 3

### During Code Review

1. **Count branches**: Each `if`, `match`, loop = +1
2. **Check nesting**: 2+ levels = likely too complex
3. **Suggest extraction**: Break into smaller functions
4. **Measure**: Use `cargo clippy -- -W cognitive-complexity`

### During Development

```rust
// While writing, ask yourself:
// - How many branches does this function have?
// - Are branches nested?
// - Can I extract conditions into helpers?
// - Is there a simpler algorithm?

// Example: Refactor as you write
fn check_eligibility(user: &User) -> bool {
    // ✅ GOOD: Each check in separate function
    is_active(user) &&
    is_verified(user) &&
    has_subscription(user) &&
    meets_requirements(user)
}

fn is_active(user: &User) -> bool {
    user.status == Status::Active
}

fn is_verified(user: &User) -> bool {
    user.email_verified && user.phone_verified
}

fn has_subscription(user: &User) -> bool {
    user.subscription.is_valid()
}

fn meets_requirements(user: &User) -> bool {
    user.age >= 18 && user.verified_address.is_some()
}
```

### Automated Checks

```bash
# Before commit
cargo clippy -- -W cognitive-complexity

# Run lint suite
pnpm lint:complexity

# Full verification
./scripts/verify-complexity.sh
```

## Current Status

### Enforced Thresholds

| Metric | Threshold | Tool | Enforcement |
|--------|-----------|------|-------------|
| Cyclomatic Complexity (M) | ≤ 3 | Manual + Clippy | Automated |
| Cognitive Complexity | ≤ 6 | .clippy.toml | Automated |
| Parameter Count | ≤ 3 | Clippy | Automated |
| Function Length | ≤ 25 lines | Code review | Manual |
| File Length | ≤ 150 lines | CI script | Automated |

### How to Check

```bash
# Check cognitive complexity violations
cargo clippy -- -W clippy::cognitive-complexity

# Check file for complex functions
cargo clippy --all-targets --all-features -- -D warnings

# Run full verification
./scripts/verify-complexity.sh

# Full lint suite (includes complexity)
pnpm lint
```

## Examples: Good vs Bad

### Example 1: State Machine

```rust
// ❌ BAD: M = 8 (nested conditions)
fn handle_message(msg: Message, state: &mut State) {
    if state.is_initialized {
        if msg.is_valid() {
            if !state.is_paused {
                if msg.priority > THRESHOLD {
                    if state.can_process() {
                        state.process(msg);
                    }
                }
            }
        }
    }
}

// ✅ GOOD: M ≤ 3 (guard clauses)
fn handle_message(msg: Message, state: &mut State) {
    if !state.is_initialized { return; }
    if !msg.is_valid() { return; }
    if state.is_paused { return; }
    if msg.priority <= THRESHOLD { return; }
    if !state.can_process() { return; }
    
    state.process(msg);
}
```

### Example 2: Data Validation

```rust
// ❌ BAD: M = 6 (multiple nested conditions)
fn validate_request(req: &Request) -> Result<()> {
    if let Some(user) = &req.user {
        if let Some(auth) = &user.auth {
            if auth.is_valid() {
                if req.data.len() > 0 {
                    if req.data.len() < MAX_SIZE {
                        Ok(())
                    } else {
                        Err("too large")
                    }
                } else {
                    Err("empty")
                }
            } else {
                Err("invalid auth")
            }
        } else {
            Err("no auth")
        }
    } else {
        Err("no user")
    }
}

// ✅ GOOD: M ≤ 3 (extracted validators)
fn validate_request(req: &Request) -> Result<()> {
    validate_user(req)?;
    validate_data(req)?;
    Ok(())
}

fn validate_user(req: &Request) -> Result<()> {
    let user = req.user.as_ref().ok_or("no user")?;
    let auth = user.auth.as_ref().ok_or("no auth")?;
    auth.is_valid().then_some(()).ok_or("invalid auth")
}

fn validate_data(req: &Request) -> Result<()> {
    if req.data.is_empty() {
        return Err("empty");
    }
    if req.data.len() >= MAX_SIZE {
        return Err("too large");
    }
    Ok(())
}
```

## Summary

### Rules

| Rule | Threshold | Enforcement | How to Check |
|------|-----------|-------------|--------------|
| **Cyclomatic Complexity** | M ≤ 3 | Automated + Manual | `cargo clippy -- -W cognitive-complexity` |
| **Cognitive Complexity** | ≤ 6 | Automated (.clippy.toml) | Same as above |
| **NO EXCEPTIONS** | ALL code | Universal | Applies to inference/, api/, everything |

### Commands

```bash
# Check complexity
cargo clippy -- -W cognitive-complexity

# Verify all standards
./scripts/verify-complexity.sh

# Full lint suite
pnpm lint

# Pre-commit check (runs automatically)
./.git/hooks/pre-commit
```

### Key Principle

**Complexity M ≤ 3 is a hard requirement for ALL code.**

Extract functions, use early returns, and break complex logic into focused modules. The cognitive cost of understanding code scales exponentially with complexity - keeping it low is a top priority.

---

**Enforcement is automatic via CI/CD and pre-commit hooks.**  
**No complex code will be accepted into the repository.**
