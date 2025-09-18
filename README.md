# Spekk

## Installation and Development Setup

To install the project for development and run tests:

```bash
# Create and activate virtual environment and install dev dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
make install-dev
uv pip install -e .

# Run tests
make test
```

## Requirements
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager