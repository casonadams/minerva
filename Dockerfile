# Multi-stage build for Minerva
FROM rust:1.92.0 as builder

WORKDIR /app

# Copy manifest files
COPY src-tauri/Cargo.toml src-tauri/Cargo.lock ./
COPY src-tauri/src ./src

# Build the Rust binary
RUN cargo build --release --bin minerva

# Runtime stage
FROM debian:bookworm-slim

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/minerva /usr/local/bin/minerva

# Create non-root user
RUN groupadd -r minerva && useradd -r -g minerva minerva
USER minerva

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/minerva health || exit 1

# Expose API port
EXPOSE 3000

# Default command
CMD ["minerva", "serve", "--host", "0.0.0.0", "--port", "3000"]
