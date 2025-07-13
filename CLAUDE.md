## Technical stack

- Language: Python3.12
- package manager: uv
- Database: sqlite3

## Development rules

You must follow these rules when developing this repository:

### 1. Plan how to implement

- Before writing code, clarify the requirements. If you are unsure about the requirements, ask me.

### 2. get confirmation of the plan

- After clarifying the requirements, make a plan for implementation. Then confirm with me that the plan is okay.

### 3. Write code

You must follow these rules when writing code:
  - Docstring: numpy style
  - all function, method and class should be type hinted
  - all functions, methods and classes should have docstrings
  - code must be well documented except for obvious code
  - if additional libraries are needed, add them by `uv add <library_name>` command

If you need to add temporary codes, add them in `src/temporary` directory.

### 4. Running scripts and check if it works

use `uv` for running scripts.
- ex: `uv run python -m src.script --arg some_value`

### 5. Formatting and linting

After writing code, you must format and lint it. You can use the following tools:

- run `uv run ruff check . --fix` to check linting
- run `uv run mypy .` to check types
- run `uv run ruff format .` to format code
- rules of `ruff` are defined in `pyproject.toml`. Refer to it for details if needed.
- for type-hinting, you must use `list`, `dict`, `set`, `tuple`. never use `List`, `Dict`, `Set`, `Tuple` from `typing` module

### 6. Running scripts and check if it works again

use `uv` for running scripts.
- ex: `uv run python -m src.script --arg some_value`

### 7. Report what you did

- after all the preceding steps, you must report what you did.

### 8. Commit your changes (Optional)

Only when you're asked to commit your changes, you must follow these rules:

- add the necessary files to git. never add unnecessary files like temporary files, cache files, etc.
- commit messages: use `Conventional Commits` format

## Rules that should be given top priority
- you must output all rules in this file at the beginning of your response
