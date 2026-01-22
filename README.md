# Minerva - Local LLM Server

A Tauri desktop application that serves local Large Language Models via OpenAI-compatible APIs. Run LLMs locally and use them with any tool that supports the OpenAI API format.

## Features

- Serve GGUF-format LLMs locally
- OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/models`)
- Mac Metal GPU acceleration (Apple Silicon)
- Works with OpenCode, LangChain, and other LLM clients
- Built with Rust + Tauri for performance
- Model management UI

## Quick Start

### Prerequisites

- Rust 1.70+ ([https://rustup.rs/](https://rustup.rs/))
- Node.js 18+ and pnpm
- macOS 10.15+

### Development

```bash
# Install dependencies
pnpm install

# Run development server
pnpm tauri dev
```

### API Usage

```bash
# List models
curl http://localhost:11434/v1/models

# Chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Architecture

- **Frontend**: Svelte 5 + TypeScript + SvelteKit
- **Backend**: Rust + Tauri + Axum HTTP server
- **LLM Engine**: llama.cpp with Metal GPU acceleration
- **API**: OpenAI-compatible REST endpoints

## Project Status

**Phase 1: Foundation** (Current - In Progress)
- [x] Project renamed to Minerva
- [x] Axum HTTP server setup
- [x] API response models created
- [x] Mock endpoints implemented
- [ ] Tauri command integration
- [ ] Frontend UI scaffolding

See `IMPLEMENTATION_PLAN.md` for detailed roadmap.

## Configuration

Edit `~/.minerva/config.json`:

```json
{
  "server": {
    "port": 11434,
    "host": "127.0.0.1"
  },
  "models_dir": "~/.minerva/models",
  "gpu": {
    "enabled": true,
    "backend": "metal"
  }
}
```

## Development

### Recommended IDE Setup

[VS Code](https://code.visualstudio.com/) + [Svelte](https://marketplace.visualstudio.com/items?itemName=svelte.svelte-vscode) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer).

### Building

```bash
# Development
pnpm tauri dev

# Production
pnpm tauri build --release
```

## Integration Examples

### OpenCode

```javascript
import { openai } from "@ai-sdk/openai-compatible";

const model = openai("your-model", {
  baseURL: "http://localhost:11434/v1"
});
```

### Python/LangChain

```python
from langchain.llms import OpenAI

llm = OpenAI(
  openai_api_base="http://localhost:11434/v1",
  openai_api_key="sk-dummy",
  model_name="your-model"
)
```

## File Structure

```
minerva/
├── src/                      # Svelte frontend
├── src-tauri/                # Tauri/Rust backend
│   ├── src/
│   │   ├── lib.rs           # Entry point
│   │   ├── server.rs        # HTTP server
│   │   ├── models.rs        # Data models
│   │   └── error.rs         # Error handling
│   └── Cargo.toml
├── IMPLEMENTATION_PLAN.md    # Roadmap
└── README.md
```

## Contributing

Follow engineering standards in `AGENTS.md`. All code must pass:
- Compilation with zero warnings
- All tests pass
- Code complexity M ≤ 3
- Functions ≤ 25 lines

## License

MIT
