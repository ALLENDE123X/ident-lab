# Docker Environment

This directory contains Docker configuration for reproducible development and testing.

## Quick Start

```bash
# Start Docker Desktop first, then:

# Build the image
docker-compose build

# Run tests
docker-compose run test

# Interactive development shell
docker-compose run dev

# Generate dataset
docker-compose run dataset

# Run quick benchmark
docker-compose run benchmark
```

## Manual Docker Commands

```bash
# Build image
docker build -t weakident-selector .

# Run tests
docker run --rm weakident-selector pytest tests/ -v

# Interactive shell
docker run -it --rm -v $(pwd):/app weakident-selector bash

# Test PySINDy import
docker run --rm weakident-selector python -c "import pysindy; print('PySINDy:', pysindy.__version__)"

# Run benchmark with output
docker run --rm -v $(pwd)/artifacts:/app/artifacts weakident-selector \
    python scripts/run_benchmark.py --cfg config/default.yaml --quick --parallel 4
```

## Services (docker-compose)

| Service | Description |
|---------|-------------|
| `dev` | Interactive development shell with mounted volume |
| `test` | Run pytest on all tests |
| `benchmark` | Run quick benchmark with 4 parallel workers |
| `dataset` | Generate dataset with default config |

## Mounted Volumes

- `.:/app` - Project directory (for development)
- `./artifacts:/app/artifacts` - Dataset and model outputs

## Dependencies

The `requirements-docker.txt` contains pinned versions:
- numpy==1.26.4
- scipy==1.11.4
- scikit-learn==1.3.0
- pysindy==1.7.5

These are tested to work together without conflicts.

## Troubleshooting

**Docker daemon not running:**
```bash
# macOS: Start Docker Desktop app
# Linux: sudo systemctl start docker
```

**Permission denied:**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
```

**Rebuild after code changes:**
```bash
docker-compose build --no-cache
```
