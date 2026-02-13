# Contributing to openeo-core

Thank you for your interest in contributing to openeo-core. We welcome contributions from everyone.

## Getting Started

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/openeo-core.git
   cd openeo-core
   ```

2. **Set up the development environment**:
   ```bash
   uv sync --extra dev
   ```

3. **Run the tests** to ensure everything works:
   ```bash
   uv run pytest tests/ -v
   ```

## Development Workflow

- Create a new branch for your changes: `git checkout -b feature/your-feature-name`
- Make your changes and add tests where appropriate
- Run the test suite: `uv run pytest tests/ -v`
- Run tests with coverage: `uv run pytest tests/ -v --cov=openeo_core`
- Commit your changes with clear, descriptive commit messages

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for public functions and classes

## Pull Requests

1. Open a pull request against the `dev` branch
2. Describe your changes clearly in the PR description
3. Ensure all tests pass

## Questions?

Open an [issue](https://github.com/PondiB/openeo-core/issues) or reach out to the maintainers. We're happy to help.
