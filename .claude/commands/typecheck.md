Run pyrefly type checking and fix all errors found.

```
uv run pyrefly check .
```

## Fix Rules

- **bad-override on return type** → change the abstract base's return type to the broader shared type (e.g., `Document` instead of a subclass).
- **bad-override on a mutable Pydantic field** (e.g., narrowing `BaseModel | None` → `PDF | None`) → add `# type: ignore[override]` on that field line with a one-line comment explaining why.
- **Missing attribute** (e.g., accessing a subclass field through a base-class variable) → use `typing.cast(SubClass, variable)` at the call site rather than changing the base class.
- **None not subscriptable** (e.g., `result["key"]` where return type is `dict | None`) → remove the `| None` from the return type if the function never actually returns `None`, or add a guard.
- **Unnecessary `# type: ignore`** → remove the suppression.

Prefer fixing root causes over adding ignores. Only use `# type: ignore` for genuine limitations (mutable field overrides, third-party stubs).

Re-run `uv run pyrefly check .` after each fix to confirm. Target: `0 errors`.
