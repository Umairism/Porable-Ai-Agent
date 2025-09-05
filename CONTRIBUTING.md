# Contributing to Portable AI Agent

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## How to Contribute

### Reporting Bugs

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/portable-ai-agent/issues/new); it's that easy!

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain which behavior you expected to see instead**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

### Development Setup

1. **Clone your fork:**
   ```bash
   git clone https://github.com/yourusername/portable-ai-agent.git
   cd portable-ai-agent
   ```

2. **Set up development environment:**
   ```bash
   python -m venv dev_env
   source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

4. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

## Code Style

- We use [black](https://github.com/psf/black) for code formatting
- We use [flake8](https://flake8.pycqa.org/) for linting
- We use [mypy](http://mypy-lang.org/) for type checking

Run these tools before submitting:

```bash
black .
flake8 .
mypy .
```

## Project Structure

```
portable-ai-agent/
├── core/                 # AI engine and neural networks
├── knowledge/            # Knowledge base management
├── memory/              # Conversation memory system
├── interface/           # User interfaces (CLI, Web)
├── utils/               # Utility functions
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Usage examples
└── scripts/             # Build and deployment scripts
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Include both unit tests and integration tests
- Test with different Python versions (3.8+)

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=core --cov=knowledge --cov=memory

# Run specific test file
python -m pytest tests/test_ai_engine.py
```

## Documentation

- Update README.md if needed
- Add docstrings to new functions and classes
- Update API documentation
- Include examples for new features

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask questions by opening an issue or reaching out to the maintainers.

Thank you for contributing! 🎉
