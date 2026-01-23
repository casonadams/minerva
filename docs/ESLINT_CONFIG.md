# ESLint Configuration

## Overview

ESLint configuration enforces JavaScript/TypeScript code quality standards aligned with our Rust engineering standards in `.clippy.toml`. This ensures consistency across the entire codebase (Rust backend + JavaScript/TypeScript frontend).

**Configuration File**: `.eslintrc.json`

## Engineering Standards Enforcement

### Complexity Limits (Matching Rust Standards)

| Rule | Threshold | Rationale |
|------|-----------|-----------|
| **Cyclomatic Complexity** | ≤ 10 | Keep functions simple and testable |
| **Max Parameters** | ≤ 3 | Use objects for more than 3 parameters (DI pattern) |
| **Max Nested Callbacks** | ≤ 3 | Prevent callback hell |
| **Max Depth** | ≤ 4 | Avoid deep nesting |
| **Max Lines per File** | ≤ 300 | Keep files focused (skips blanks/comments) |
| **Max Line Length** | ≤ 100 | Improve readability |

### Language Requirements

#### Variables & Declarations
- **no-var**: ✅ REQUIRED - Use `const`/`let` only
- **prefer-const**: ✅ REQUIRED - Use `const` by default
- **no-unused-vars**: ✅ REQUIRED - Enforce clean code
  - Prefix unused variables with `_` to allow intentional ignoring

#### Functions
- **prefer-arrow-callback**: ✅ REQUIRED - Use arrow functions for callbacks
- **arrow-body-style**: ✅ REQUIRED - Omit braces for simple returns
- **space-before-function-paren**: Consistent spacing around function definitions

#### Code Quality
- **no-console**: ⚠️ WARN - Allow `warn` and `error` only (log via system)
- **no-debugger**: ⚠️ WARN - Remove debug statements before commit
- **no-eval**: ✅ REQUIRED - Security: never eval code
- **no-with**: ✅ REQUIRED - Deprecated JavaScript feature
- **eqeqeq**: ✅ REQUIRED - Strict equality (===) only
- **no-implicit-coercion**: ✅ REQUIRED - Explicit type conversions

#### Styling
- **quotes**: `'single'` (unless escaping needed)
- **semi**: ✅ REQUIRED - Always use semicolons
- **indent**: 2 spaces
- **comma-dangle**: Always multiline (enables cleaner diffs)
- **curly**: Always use braces, even for single statements
- **brace-style**: 1TBS (One True Brace Style)

### Parameter Handling

Like Rust's DI pattern, JavaScript functions should use parameter objects:

**❌ BAD: Too many parameters**
```javascript
function generateText(prompt, maxTokens, temperature, topP, frequencyPenalty) {
  // 5 parameters violates max-params threshold
}
```

**✅ GOOD: Use parameter object**
```javascript
function generateText(prompt, params) {
  // params = { maxTokens, temperature, topP, frequencyPenalty }
}

// Or with destructuring:
function generateText(prompt, { maxTokens, temperature, topP, frequencyPenalty }) {
  // Clean and readable
}
```

## Svelte Integration

ESLint is configured to process `.svelte` files via `svelte3/svelte3` processor (when using eslint-plugin-svelte3).

Current configuration supports:
- ✅ Standard ESLint rules for script blocks
- ⚠️ Manual checking for template best practices
- ✅ TypeScript support in Svelte components

## Configuration Reference

### Extends
- `eslint:recommended` - ESLint's recommended rules

### Parser
- ECMAScript 2021 (ES12) features supported
- Module syntax enabled

### Environments
- `browser`: DOM APIs available
- `es2021`: Modern JavaScript features
- `node`: Node.js globals available

## Usage

### Check Code Quality
```bash
pnpm lint:frontend
```

### Auto-fix Issues
```bash
pnpm lint:frontend:fix  # (Note: requires package.json script update)
```

### Format Code
```bash
pnpm fmt:frontend
```

### Run All Checks
```bash
pnpm lint
pnpm fmt:check
pnpm check:all
```

## Integration with CI/CD

Recommended pre-commit hooks:

```bash
# Before commit
pnpm lint:frontend
pnpm fmt:frontend:check
```

Recommended pre-push:

```bash
# Before push
pnpm test:all
pnpm lint
pnpm check:all
```

## Common Rules Explained

### Complexity (max: 10)
Cyclomatic complexity > 10 suggests function is doing too much.

✅ GOOD:
```javascript
function processRequest(request) {
  if (!validate(request)) {
    return null;
  }

  if (request.type === 'sync') {
    return handleSync(request);
  }

  if (request.type === 'async') {
    return handleAsync(request);
  }

  return handleDefault(request);
}
```

❌ BAD: (complexity > 10 with many nested conditions)
```javascript
function processRequest(request) {
  if (request && request.data) {
    if (request.data.type === 'A') {
      if (validate(request)) {
        // ... many more nested conditions
      }
    }
  }
}
```

### Max Parameters (max: 3)
Exceeding 3 parameters indicates a function is doing too much.

❌ BAD:
```javascript
function fetchData(url, method, headers, timeout, retries, cache) {
  // 6 parameters!
}
```

✅ GOOD:
```javascript
function fetchData(url, options) {
  const { method = 'GET', headers = {}, timeout = 5000, retries = 3, cache = true } = options;
}
```

### Max Depth (max: 4)
Deeply nested code is hard to follow.

❌ BAD: (depth > 4)
```javascript
function process(data) {
  if (data) {
    if (Array.isArray(data)) {
      data.forEach(item => {
        if (item.valid) {
          if (item.id) {
            if (isPositive(item.id)) {
              // Finally here after 5 levels!
            }
          }
        }
      });
    }
  }
}
```

✅ GOOD: (early returns, depth = 2)
```javascript
function process(data) {
  if (!data || !Array.isArray(data)) {
    return;
  }

  data.forEach(item => {
    if (!item.valid || !item.id || !isPositive(item.id)) {
      return;
    }

    // Process item
  });
}
```

## Exception Handling

### Ignore Patterns

Prefix unused variables with underscore to ignore `no-unused-vars`:

```javascript
// _params will not trigger no-unused-vars
function process(_params) {
  return staticValue;
}

// _err will not trigger no-unused-vars
try {
  doSomething();
} catch (_err) {
  // Intentionally ignoring error
}
```

### Ignore Single Lines

```javascript
// eslint-disable-next-line no-eval
const result = eval(expression);
```

### Ignore Entire Files

```javascript
// eslint-disable
// This entire file is legacy code
```

## Migration Guide

### From Loose to Strict

If you have code that doesn't meet these standards, use this checklist:

1. **Replace `var` with `const`/`let`**
   ```bash
   # Find all var declarations
   grep -r "^\s*var " src/
   ```

2. **Add parameter objects**
   ```javascript
   // Before
   function foo(a, b, c, d) { }

   // After
   function foo(a, { b, c, d } = {}) { }
   ```

3. **Reduce cyclomatic complexity**
   - Extract conditionals to separate functions
   - Use switch statements instead of if/else chains
   - Use early returns to reduce nesting

4. **Reduce file size**
   - Split large files into focused modules
   - Group related functions together

5. **Reduce nesting**
   - Use early returns
   - Extract nested logic to helper functions

## Next Steps

1. Run ESLint on current codebase:
   ```bash
   npx eslint src/ --ext .js,.ts,.svelte 2>&1 | tee eslint-report.txt
   ```

2. Create ESLint violations report (similar to CLIPPY_VIOLATIONS.md)

3. Add ESLint scripts to package.json:
   ```json
   {
     "scripts": {
       "lint:frontend": "eslint src/ --ext .js,.ts,.svelte",
       "lint:frontend:fix": "eslint src/ --ext .js,.ts,.svelte --fix",
       "lint": "pnpm lint:backend && pnpm lint:frontend"
     }
   }
   ```

4. Integrate into CI/CD pipeline

5. Add pre-commit hooks

## References

- **ESLint**: https://eslint.org/docs/rules/
- **Complexity Analysis**: https://en.wikipedia.org/wiki/Cyclomatic_complexity
- **JavaScript Best Practices**: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide
- **Rust Engineering Standards**: See `docs/ENGINEERING_STANDARDS.md`

## Related Configuration Files

- `.eslintrc.json` - This configuration file
- `.prettierrc` - Code formatting (currently using defaults)
- `tsconfig.json` - TypeScript configuration
- `.clippy.toml` - Rust linting configuration (Rust backend equivalent)
