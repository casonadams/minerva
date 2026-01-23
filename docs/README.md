# Minerva Documentation Hub

Welcome to the Minerva documentation. This is your central hub for understanding and working with Minerva, a production-ready OpenAI-compatible local LLM server built with Tauri and Rust.

## 🚀 Quick Start

**New to Minerva?** Start here:

1. **[Development Setup](DEVELOPMENT.md)** - Get the project running locally
2. **[Project Overview](../README.md)** - Understand what Minerva does
3. **[Phase Overview](PHASES.md)** - See what was built in each phase

## 📚 Core Documentation

### Project Overview
- **[Main README](../README.md)** - Complete project overview, features, API reference
- **[PHASES.md](PHASES.md)** - Summary of all 7 completed development phases
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Original high-level architecture

### Development & Guides
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Local development setup and workflow
- **[GPU_ACCELERATION.md](GPU_ACCELERATION.md)** - GPU configuration and optimization
- **[CODE_QUALITY.md](CODE_QUALITY.md)** - Code standards and quality metrics
- **[SCRIPTS.md](SCRIPTS.md)** - Available npm/pnpm scripts
- **[TEST_STRUCTURE.md](TEST_STRUCTURE.md)** - Testing patterns and organization

### Technical Details
- **[PHASE_7_PLAN.md](PHASE_7_PLAN.md)** - Latest phase (Production hardening) details
- **[ESLINT_CONFIG.md](ESLINT_CONFIG.md)** - Frontend linting configuration
- **[CLIPPY_VIOLATIONS.md](CLIPPY_VIOLATIONS.md)** - Backend code quality rules

## 🎯 For Different Audiences

### For End Users
- Start with: [../README.md](../README.md) - Quick Start section
- Then read: [DEVELOPMENT.md](DEVELOPMENT.md) - Setup instructions
- Reference: [GPU_ACCELERATION.md](GPU_ACCELERATION.md) - If using GPU

### For Contributors
1. [DEVELOPMENT.md](DEVELOPMENT.md) - Get environment set up
2. [PHASES.md](PHASES.md) - Understand what's been built
3. [CODE_QUALITY.md](CODE_QUALITY.md) - Learn our standards
4. [TEST_STRUCTURE.md](TEST_STRUCTURE.md) - Understand testing approach
5. Start coding! Look at similar code for patterns.

### For Code Reviewers
- [CODE_QUALITY.md](CODE_QUALITY.md) - Standards to check against
- [PHASES.md](PHASES.md) - Architecture overview
- [CLIPPY_VIOLATIONS.md](CLIPPY_VIOLATIONS.md) - Backend rules
- [ESLINT_CONFIG.md](ESLINT_CONFIG.md) - Frontend rules

### For DevOps/Operations
- [../README.md](../README.md) - HTTP endpoints section
- [PHASE_7_PLAN.md](PHASE_7_PLAN.md) - Observability and health checks
- [SCRIPTS.md](SCRIPTS.md) - Deployment scripts (if available)

## 🔍 Quick Navigation by Topic

### Understanding Architecture
1. **[PHASES.md](PHASES.md)** - High-level overview of all phases
2. **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Original architecture design
3. **[PHASE_7_PLAN.md](PHASE_7_PLAN.md)** - Latest production features

### Setting Up & Running
1. **[DEVELOPMENT.md](DEVELOPMENT.md)** - Complete setup guide
2. **[SCRIPTS.md](SCRIPTS.md)** - Available commands
3. **[GPU_ACCELERATION.md](GPU_ACCELERATION.md)** - GPU setup (optional)

### Writing Code
1. **[CODE_QUALITY.md](CODE_QUALITY.md)** - Standards (READ THIS FIRST!)
2. **[TEST_STRUCTURE.md](TEST_STRUCTURE.md)** - How to write tests
3. **[PHASES.md](PHASES.md)** - See patterns in existing code

### Monitoring & Operations
1. **[PHASE_7_PLAN.md](PHASE_7_PLAN.md)** - Health checks and metrics
2. **[../README.md](../README.md)** - HTTP endpoints reference
3. **[GPU_ACCELERATION.md](GPU_ACCELERATION.md)** - Performance tuning

## 📊 Project Status

| Metric | Status |
|--------|--------|
| **Phases Complete** | 7/7 ✅ |
| **Tests Passing** | 827/827 ✅ |
| **Lint Violations** | 0 ✅ |
| **Warnings** | 0 ✅ |
| **Production Ready** | YES ✅ |

### Latest Updates
- **Phase 7 Complete:** Production hardening with logging, resilience, and observability
- **827 Tests:** Full test coverage with meaningful assertions
- **Zero Violations:** Clean build, no warnings, no linting issues

## 🛠️ Common Tasks

### Running the Application
```bash
pnpm install          # Install dependencies
pnpm tauri dev        # Start development server
pnpm tauri build      # Build production app
```

### Development
```bash
pnpm fmt              # Format code
pnpm lint             # Check code quality
pnpm test             # Run all tests
pnpm test:backend:watch  # Watch mode for development
```

### Getting Help
See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Environment setup
- Troubleshooting
- Common development tasks
- IDE configuration

## 📖 Complete Phase Documentation

For detailed implementation specifics (if needed), phase-specific docs are available:

**Phase 1:** [PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md)  
**Phase 2:** [PHASE_2_PLAN.md](PHASE_2_PLAN.md)  
**Phase 3:** [PHASE_3_PLAN.md](PHASE_3_PLAN.md) | [PHASE_3_IMPLEMENTATION.md](PHASE_3_IMPLEMENTATION.md)  
**Phase 3.5:** [PHASE_3_5_IMPLEMENTATION.md](PHASE_3_5_IMPLEMENTATION.md)  
**Phase 3.5a:** [PHASE_3_5A_COMPLETION.md](PHASE_3_5A_COMPLETION.md)  
**Phase 3.5b:** [PHASE_3_5B_PLAN.md](PHASE_3_5B_PLAN.md) | [PHASE_3_5B_SESSION_SUMMARY.md](PHASE_3_5B_SESSION_SUMMARY.md)  
**Phase 4:** [PHASE_4_PROGRESS.md](PHASE_4_PROGRESS.md)  
**Phase 5:** [PHASE_5_PLAN.md](PHASE_5_PLAN.md)  
**Phase 6:** [PHASE_6_PLAN.md](PHASE_6_PLAN.md)  
**Phase 7:** [PHASE_7_PLAN.md](PHASE_7_PLAN.md)  

*(Read [PHASES.md](PHASES.md) for summaries instead - much quicker!)*

## 🔗 Key Links

- **Source Code:** [../src-tauri/src/](../src-tauri/src/)
- **Tests:** [../tests/](../tests/)
- **Frontend:** [../src/](../src/)
- **Config:** [../src-tauri/Cargo.toml](../src-tauri/Cargo.toml)
- **Engineering Standards:** [../AGENTS.md](../AGENTS.md)

## 💡 Pro Tips

1. **Just getting started?** Read [PHASES.md](PHASES.md) first - it's much shorter than individual phase docs
2. **Want to contribute?** Follow [CODE_QUALITY.md](CODE_QUALITY.md) strictly - zero tolerance for violations
3. **Debugging test failures?** Look at similar tests in codebase first, then check [TEST_STRUCTURE.md](TEST_STRUCTURE.md)
4. **Performance issues?** Check [GPU_ACCELERATION.md](GPU_ACCELERATION.md) and [PHASE_7_PLAN.md](PHASE_7_PLAN.md) for observability

## 📞 Support

- **Setup Issues:** See [DEVELOPMENT.md](DEVELOPMENT.md) - Troubleshooting section
- **Code Questions:** Check [CODE_QUALITY.md](CODE_QUALITY.md) for standards
- **Architecture Questions:** See [PHASES.md](PHASES.md) or [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- **Performance Questions:** See [GPU_ACCELERATION.md](GPU_ACCELERATION.md)

## 📋 Documentation Standards

All documentation follows:
- Clear, concise language
- Practical examples
- Links to related docs
- Updated status sections

*Something unclear? File an issue or improve the docs!*

---

**Last Updated:** Phase 7 Complete (January 2025)  
**Documentation Version:** 2.0 (Consolidated)  
**All 7 Phases Complete** ✅
