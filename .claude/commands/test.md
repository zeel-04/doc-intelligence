Run the test suite and fix any failures. Optionally scope to a path or filter.

```
uv run pytest tests/ $ARGUMENTS -x
```

Pass a path (`tests/pdf/`) or a filter (`-k formatter`) as `$ARGUMENTS` to narrow scope.

## Fix Rules

- **Read the traceback** fully before touching any code.
- **Fix the implementation**, not the test — unless the test assertion itself is provably wrong.
- **One failure at a time**: use `-x` (already included) to stop on first failure, fix it, then re-run.
- **Fixture errors** (`fixture not found`) → add the missing fixture to `tests/conftest.py` following the existing patterns.
- **Import errors** → check that all new modules are importable and that `__init__.py` files exist where needed.
- **Assertion errors** → read what was expected vs. what was returned; trace back to the function under test.

After fixing, run the full suite `uv run pytest tests/` (no `-x`) to make sure nothing regressed.
Target: all tests pass, zero warnings about unraisable exceptions.
