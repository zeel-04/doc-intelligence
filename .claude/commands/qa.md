Run all quality checks for this project in sequence: tests, linting, formatting, and type checking. Fix any issues found.

## Steps

Run these four commands **in order**, stopping to fix failures before moving to the next step:

### 1. Tests
```
uv run pytest tests/ $ARGUMENTS
```
Flags you can pass: a path like `tests/pdf/` to scope, or `-k pattern` to filter.

### 2. Lint
```
uv run ruff check .
```

### 3. Format
```
uv run ruff format --check .
```

### 4. Type check
```
uv run pyrefly check .
```

---

## Fix Rules

**If tests fail:**
- Read the failing test(s) and the code they exercise.
- Fix the implementation, not the tests (unless the test itself is wrong).
- Re-run only the failing test file first: `uv run pytest <file> -x`.

**If ruff check fails:**
- Run `uv run ruff check . --fix` to auto-fix safe issues.
- For remaining violations, edit the source manually.

**If ruff format fails:**
- Run `uv run ruff format .` to auto-format everything.

**If pyrefly fails:**
- Look at each error location and fix the type annotation or add a `# type: ignore[<rule>]` comment with a brief inline explanation only when a legitimate suppression is needed (e.g., known bad-override on a subclass field).
- Prefer fixing the root type (e.g., changing a return type on an abstract base) over scattering ignores.

After all fixes, re-run the full sequence to confirm everything is green before reporting done.
