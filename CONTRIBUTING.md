# Contributing to AgentNate

Thanks for your interest in contributing! AgentNate is an actively developed project and contributions are welcome.

## Getting Started

1. Fork the repository
2. Clone your fork and run `install.bat`
3. Make your changes
4. Test locally with `launcher.bat`
5. Submit a pull request

## Development Setup

AgentNate is 100% portable. After cloning:

```batch
install.bat          :: Downloads Python, Node.js, all dependencies
launcher.bat         :: Starts the app (browser mode)
launcher.bat --server  :: API server only (for development)
```

The codebase uses:
- **Backend**: Python 3.14 + FastAPI (in `backend/`)
- **Frontend**: Vanilla JavaScript ES modules (in `ui/`)
- **Providers**: LLM provider integrations (in `providers/`)
- **Orchestrator**: Request queue and health monitoring (in `orchestrator/`)

## What to Contribute

- **Bug fixes** -- Always welcome. Please include steps to reproduce.
- **New providers** -- Add support for additional LLM backends (see `providers/base_provider.py`).
- **New tools** -- Agent tools live in `backend/tools/`. Each tool is a function with a schema.
- **New workflow templates** -- n8n node builders in `backend/workflow_templates.py`.
- **UI improvements** -- Frontend modules in `ui/js/`. No build step, no framework.
- **Documentation** -- Manual source in `manual/AgentNate-Manual.md`.

## Code Style

- Python: Follow existing patterns. No strict linter enforced, but keep it clean.
- JavaScript: Vanilla ES modules. No TypeScript, no React, no build tools.
- Avoid adding external dependencies unless absolutely necessary.
- Keep it portable -- no system-level installs, no Docker requirements.

## Pull Requests

- Keep PRs focused on a single change
- Describe what changed and why
- Include screenshots for UI changes
- Test on Windows 10 or 11 before submitting

## Reporting Issues

Use the [issue templates](https://github.com/rookiemann/AgentNate/issues/new/choose) for bug reports and feature requests. For questions and discussion, use [Discussions](https://github.com/rookiemann/AgentNate/discussions).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
