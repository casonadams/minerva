# Minerva Deployment Guide

## Overview

This guide covers deploying Minerva in production environments. Minerva is designed for both GUI-based and headless (API-only) deployments.

## System Requirements

### Minimum
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB (for models)
- Network: Stable internet connection

### Recommended for Production
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+ (for multiple models)
- GPU: NVIDIA GPU with CUDA support (optional, for acceleration)
- Network: 1Gbps+ connection

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/casonadams/playground.git
cd playground

# Build backend
pnpm build:backend

# Build frontend (GUI)
pnpm build:frontend

# Or run in development
pnpm dev
```

### Pre-built Binaries

Download from releases page or build from source.

## Configuration

### Environment Variables

Create `.env` file in the project root:

```env
# Server Configuration
MINERVA_HOST=0.0.0.0
MINERVA_PORT=3000
MINERVA_WORKERS=4

# Models Directory
MINERVA_MODELS_DIR=/var/lib/minerva/models

# Rate Limiting
MINERVA_RATE_LIMIT_BURST=100
MINERVA_RATE_LIMIT_RPS=10

# Logging
MINERVA_LOG_LEVEL=info
MINERVA_LOG_FILE=/var/log/minerva/server.log

# GPU Configuration (optional)
MINERVA_ENABLE_GPU=true
MINERVA_DEVICE=cuda
MINERVA_MAX_GPU_MEMORY=16GB
```

### Configuration File

Create `minerva.toml`:

```toml
[server]
host = "0.0.0.0"
port = 3000
workers = 4

[models]
directory = "/var/lib/minerva/models"
max_loaded = 3
auto_unload_idle_minutes = 300

[rate_limiting]
burst_capacity = 100
refill_rate = 10
cleanup_interval_seconds = 300

[logging]
level = "info"
file = "/var/log/minerva/server.log"
max_file_size_mb = 100
max_backups = 5

[gpu]
enabled = true
device = "cuda"
max_memory_gb = 16
enable_tensor_cores = true

[inference]
default_temperature = 0.7
default_top_p = 1.0
max_tokens = 2048
context_window = 4096
```

## Deployment Scenarios

### 1. GUI Deployment (Default)

Run with graphical interface:

```bash
cargo tauri dev          # Development
cargo tauri build        # Production build
./src-tauri/target/release/minerva  # Run binary
```

Access GUI at: `http://localhost:5173`

### 2. Headless API Server

Run API without GUI (Phase 11):

```bash
# When Phase 11 CLI is ready:
minerva serve --host 0.0.0.0 --port 3000

# Or with configuration file:
minerva serve --config /etc/minerva/config.toml
```

**Docker Example:**

```dockerfile
FROM rust:latest as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates
COPY --from=builder /app/target/release/minerva /usr/local/bin/
EXPOSE 3000
CMD ["minerva", "serve", "--host", "0.0.0.0", "--port", "3000"]
```

### 3. Kubernetes Deployment

Example `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minerva
spec:
  replicas: 3
  selector:
    matchLabels:
      app: minerva
  template:
    metadata:
      labels:
        app: minerva
    spec:
      containers:
      - name: minerva
        image: minerva:latest
        ports:
        - containerPort: 3000
        env:
        - name: MINERVA_HOST
          value: "0.0.0.0"
        - name: MINERVA_PORT
          value: "3000"
        - name: MINERVA_MODELS_DIR
          value: "/models"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: minerva-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: minerva-service
spec:
  selector:
    app: minerva
  ports:
  - protocol: TCP
    port: 3000
    targetPort: 3000
  type: LoadBalancer
```

### 4. Systemd Service

Create `/etc/systemd/system/minerva.service`:

```ini
[Unit]
Description=Minerva AI Server
After=network.target

[Service]
Type=simple
User=minerva
WorkingDirectory=/var/lib/minerva
ExecStart=/usr/local/bin/minerva serve --host 0.0.0.0 --port 3000
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="MINERVA_MODELS_DIR=/var/lib/minerva/models"
Environment="MINERVA_LOG_LEVEL=info"

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable minerva
sudo systemctl start minerva
sudo journalctl -u minerva -f  # View logs
```

## Security Considerations

### 1. Network Security

```nginx
# Nginx reverse proxy configuration
server {
    listen 443 ssl http2;
    server_name api.minerva.example.com;

    ssl_certificate /etc/letsencrypt/live/api.minerva.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.minerva.example.com/privkey.pem;

    # Rate limiting at nginx level
    limit_req_zone $binary_remote_addr zone=minerva:10m rate=10r/s;
    limit_req zone=minerva burst=100 nodelay;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. API Keys (Future Phase)

When authentication is added:

```bash
# Generate API key
minerva auth generate-key

# Use in requests
curl -H "Authorization: Bearer sk-xxxx..." \
     http://api.minerva.example.com/v1/chat/completions
```

### 3. Model Isolation

Models should be stored in restricted directories:

```bash
sudo mkdir -p /var/lib/minerva/models
sudo chown minerva:minerva /var/lib/minerva/models
sudo chmod 700 /var/lib/minerva/models
```

### 4. Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 3000/tcp         # API
sudo ufw allow 443/tcp           # HTTPS (nginx)
sudo ufw allow 22/tcp            # SSH (admin only)
sudo ufw enable
```

## Monitoring

### Health Checks

```bash
# Server health
curl http://localhost:3000/health

# Readiness
curl http://localhost:3000/ready

# Metrics
curl http://localhost:3000/metrics
```

### Logging

View structured logs:

```bash
journalctl -u minerva -f              # Live tail
journalctl -u minerva --since "1 hour ago"  # Last hour
journalctl -u minerva -p err          # Errors only
```

### Performance Monitoring

Monitor from metrics endpoint:

```bash
while true; do
  curl -s http://localhost:3000/metrics | jq '.requests.rps'
  sleep 5
done
```

### Prometheus Integration

Add scrape config to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'minerva'
    static_configs:
      - targets: ['localhost:3000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## Performance Tuning

### 1. Worker Threads

Adjust based on CPU count:

```env
MINERVA_WORKERS=8  # For 8-core system
```

### 2. Rate Limiting

Production settings:

```env
MINERVA_RATE_LIMIT_BURST=100    # Allow 100 requests burst
MINERVA_RATE_LIMIT_RPS=10       # 10 requests per second sustained
```

### 3. Model Caching

```toml
[models]
max_loaded = 3          # Keep 3 models in memory
auto_unload_idle_minutes = 300  # Unload after 5 minutes idle
```

### 4. GPU Optimization

```toml
[gpu]
enabled = true
max_memory_gb = 16
enable_tensor_cores = true
optimization_level = "aggressive"
```

## Troubleshooting

### Server won't start

```bash
# Check port availability
lsof -i :3000

# Check configuration
minerva config validate

# Run with debug logging
MINERVA_LOG_LEVEL=debug minerva serve
```

### High memory usage

```bash
# Check loaded models
curl http://localhost:3000/metrics | jq '.models'

# Reduce max_loaded count in config
# Or manually unload models
curl -X DELETE http://localhost:3000/v1/models/model-id
```

### Slow responses

```bash
# Check response time metrics
curl http://localhost:3000/metrics | jq '.response_times'

# Reduce concurrent requests or increase workers
# Scale horizontally with multiple instances
```

### Rate limiting too strict

Adjust in configuration:

```toml
[rate_limiting]
burst_capacity = 200    # Increase from 100
refill_rate = 20        # Increase from 10
```

## Backup and Recovery

### Model Backup

```bash
# Backup models directory
tar -czf minerva-models-backup.tar.gz /var/lib/minerva/models/

# Restore
tar -xzf minerva-models-backup.tar.gz -C /var/lib/minerva/
```

### Configuration Backup

```bash
# Backup configuration
cp /etc/minerva/config.toml /etc/minerva/config.toml.backup

# Restore
cp /etc/minerva/config.toml.backup /etc/minerva/config.toml
```

## Scaling

### Horizontal Scaling

Deploy multiple instances behind a load balancer:

```
Load Balancer (nginx/HAProxy)
    |
    ├── Minerva Instance 1 (port 3001)
    ├── Minerva Instance 2 (port 3002)
    └── Minerva Instance 3 (port 3003)
```

### Shared Model Storage

Use network storage for models:

```bash
# Mount NFS with models
sudo mount -t nfs 192.168.1.100:/models /var/lib/minerva/models
```

## Updates and Maintenance

### Rolling Updates

```bash
# Build new version
pnpm build:backend

# Create backup
sudo systemctl stop minerva
sudo cp -r /var/lib/minerva /var/lib/minerva.backup

# Update binary
sudo cp target/release/minerva /usr/local/bin/

# Restart service
sudo systemctl start minerva
```

### Database Migrations (Future)

```bash
minerva migrate --version latest
```

## Support

For issues or questions:

1. Check logs: `journalctl -u minerva -f`
2. Check metrics: `curl http://localhost:3000/metrics`
3. Run health check: `curl http://localhost:3000/health`
4. Review configuration: `cat /etc/minerva/config.toml`
5. Report issues with detailed logs and metrics
