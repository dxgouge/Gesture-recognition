# Contributing to Gesture Recognition Project

Thank you for your interest in contributing to the Gesture Recognition project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide detailed information**:
   - System specifications (OS, Python version, etc.)
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Screenshots or error messages if applicable

### Suggesting Enhancements

1. **Check existing feature requests** first
2. **Describe the enhancement** clearly
3. **Explain the use case** and benefits
4. **Consider implementation complexity**

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following the coding standards
4. **Test your changes** thoroughly
5. **Commit with clear messages** (`git commit -m 'Add amazing feature'`)
6. **Push to your branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

## 📋 Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- A code editor (VS Code, PyCharm, etc.)

### Setup Steps

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/gesture-recognition.git
   cd gesture-recognition
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## 🎨 Coding Standards

### Python Code Style

- **Follow PEP 8** style guidelines
- **Use type hints** where appropriate
- **Write docstrings** for all functions and classes
- **Keep functions small** and focused
- **Use meaningful variable names**

### Code Formatting

We use **Black** for code formatting:

```bash
black .
```

### Linting

We use **Flake8** for linting:

```bash
flake8 .
```

### Testing

- **Write tests** for new functionality
- **Run existing tests** before submitting:
  ```bash
  pytest
  ```

## 📁 Project Structure Guidelines

### Adding New Features

1. **Choose the appropriate directory**:
   - `Classification/` - For classification-based approaches
   - `ObjectRecognition/` - For object detection approaches
   - `MediapipesGestureRecognition/` - For MediaPipe-based approaches

2. **Follow naming conventions**:
   - Use snake_case for file names
   - Use descriptive names that indicate functionality
   - Include the approach type in the name (e.g., `train_classification_*.py`)

3. **Update documentation**:
   - Update README.md if adding new features
   - Add docstrings to new functions
   - Update requirements.txt if adding new dependencies

### File Organization

```
YourNewFeature/
├── train_your_feature.py      # Training script
├── your_feature_webcam.py     # Real-time detection
├── test_your_feature.py       # Testing script
├── dataset.yaml               # Dataset configuration
└── your_feature_training/     # Training outputs
    └── weights/
        ├── best.pt
        └── last.pt
```

## 🧪 Testing Guidelines

### Unit Tests

- **Test individual functions** and methods
- **Use descriptive test names**
- **Test edge cases** and error conditions
- **Mock external dependencies** when appropriate

### Integration Tests

- **Test complete workflows**
- **Test with sample data**
- **Verify output formats**

### Manual Testing

- **Test with real webcam input**
- **Verify different lighting conditions**
- **Test on different operating systems**

## 📝 Documentation

### Code Documentation

- **Use docstrings** for all public functions and classes
- **Follow Google docstring format**:
  ```python
  def example_function(param1: str, param2: int) -> bool:
      """Brief description of the function.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: Description of when this exception is raised
      """
  ```

### README Updates

- **Update installation instructions** if adding new dependencies
- **Add usage examples** for new features
- **Update the features list**
- **Add troubleshooting information** if needed

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
2. **Run code formatting** (`black .`)
3. **Check for linting errors** (`flake8 .`)
4. **Update documentation** if needed
5. **Test your changes** thoroughly

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version numbers** in setup.py and README.md
2. **Update CHANGELOG.md** with new features and fixes
3. **Create a release tag**
4. **Test the release** thoroughly

## 🆘 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Pull Request Comments**: For code review discussions

### Code Review Process

1. **All PRs require review** before merging
2. **Be constructive** in feedback
3. **Ask questions** if something is unclear
4. **Be responsive** to review comments

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to the Gesture Recognition project! 🎉
