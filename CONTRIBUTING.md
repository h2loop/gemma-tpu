# Contributing to gemma-tpu

We welcome contributions! This project aims to make TPU fine-tuning accessible to the ML community.

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/h2loop/gemma-tpu/issues) for bug reports and feature requests
- Include your TPU type, JAX version, and error messages
- For training issues, include your config and the last few log lines

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test on at least one TPU type if possible
5. Submit a PR with a clear description

### Areas We'd Love Help With

- **Dataset adapters**: Add support for more dataset formats (ShareGPT, Alpaca, custom JSONL)
- **Model variants**: Test and document Gemma 9B/12B configurations
- **TPU types**: Benchmark on v5e, v4, or other TPU configurations
- **Inference**: Improve vLLM-TPU setup documentation
- **Docker**: Help create reproducible container images

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/gemma-tpu.git
cd gemma-tpu

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r training/requirements.txt
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/qwix
```

## Code Style

- Use clear variable names over comments
- Keep functions focused and under 50 lines where possible
- Add type hints for function signatures
- Test with `python -m py_compile <file>` before submitting

## Questions?

Open an issue with the `question` label or start a discussion.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
