# Contributing

## Development workflow

1. Create a focused branch from the default branch.
2. Create and activate a Python 3.10+ virtual environment.
3. Install the development profile with `pip install -e ".[dev]"`.
4. Install hooks with `pre-commit install`.
5. Add tests for behavioral changes.
6. Run the checks below before opening a pull request.

```bash
ruff check src tests loadtest dashboard
ruff format --check src tests loadtest dashboard
pytest --cov=support_search --cov-report=term-missing
support-search build-index
support-search evaluate
support-search quality-gate
```

Pull requests should explain the problem, the design decision, validation evidence, and any operational or
model-quality impact. Changes to retrieval behavior should include before-and-after ranking metrics.

