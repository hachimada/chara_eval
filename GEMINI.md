# Development rules

## Technical stack

- Language: Python3.12
- package manager: uv
- Database: sqlite3

## Coding style

- Docstring: numpy style
- commit messages: use `Conventional Commits` format
- all function, method and class should be type hinted
- all functions, methods and classes should have docstrings
- code must be well documented except for obvious code

## Formatting and linting

- use `ruff` for formatting and linting through `uv`
- run `uv run ruff check . --fix` to check linting
- run `uv run ruff format .` to format code
- rules of `ruff` are defined in `pyproject.toml`

## Type checking

- use `mypy` for type checking through `uv`
- run `uv run mypy .` to check types
- rules of `mypy` are defined in `pyproject.toml`

## Running commands

- You must use `uv` to run commands
- You must use `uv` to run python scripts. example: `uv run python -m src.script --arg some_value`

## The most important rules

- before you write code, you must clarify the requirements. If you aren't sure about the requirements, ask me.
- after you clarify the requirements, you must maek a plan of implementation. Then confirm with me that the plan is okay.
- one function or method should do one thing
- function, method and class names should be descriptive
- code should be well documented
- you should make a commit for each feature or bug fix
- for type-hinting, you must use `list`, `dict`, `set`, `tuple` not `List`, `Dict`, `Set`, `Tuple` from `typing` module

## Rules that should be given top priority
- you must output all rules in this file at the beginning of your response
