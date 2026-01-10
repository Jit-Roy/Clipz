# Contributing to Viral Clip Extractor

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/clip-extract.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it and install dependencies: `pip install -r requirements.txt`
5. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test your changes thoroughly
4. Commit with clear, descriptive messages
5. Push to your fork and submit a pull request

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

Example:
```python
def compute_audio_scores(self, audio_path, use_cache=True):
    """
    Compute excitement scores from audio analysis.
    
    Args:
        audio_path: Path to audio file
        use_cache: Whether to use cached results
        
    Returns:
        Tuple of (timestamps, scores)
    """
    # Implementation
```

## Testing

Before submitting a pull request:

1. Test your changes with various video files
2. Ensure no regressions in existing functionality
3. Add tests for new features if applicable

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- [ ] Improved laughter/applause detection using trained models
- [ ] Performance optimization for large videos
- [ ] Better error handling and user feedback
- [ ] Support for additional video formats

### Medium Priority
- [ ] Multi-language transcription support
- [ ] Real-time processing mode
- [ ] Web interface for clip extraction
- [ ] Docker containerization

### Low Priority
- [ ] Additional LLM provider integrations
- [ ] Advanced visualization tools
- [ ] Batch processing improvements
- [ ] Cloud deployment guides

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update requirements.txt if you add dependencies
3. Ensure your code follows the project's coding style
4. Provide a clear description of the changes in your PR
5. Link any related issues

## Reporting Bugs

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs
- Sample video (if possible and not too large)

## Feature Requests

For feature requests:

- Clearly describe the feature and its benefits
- Explain use cases
- Consider implementation complexity
- Check if similar features exist or have been requested

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

Feel free to open an issue for questions or discussions about:
- Implementation details
- Design decisions
- Best practices
- Usage help

Thank you for contributing! 🎉
