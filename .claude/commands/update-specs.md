Explore the current codebase and update all spec files in `specs/` so they are the single source of truth for what is actually implemented.

## Inputs

- `$ARGUMENTS` — optional: a brief description of what changed (e.g., "added OCR pipeline"). If empty, discover changes by exploring the code.

## Steps

### 1. Explore the codebase

Thoroughly explore the current implementation to build a complete picture:

- **Package structure:** recursively list all Python files under `doc_intelligence/` and `tests/`. Note any new packages, modules, or files not reflected in specs.
- **Public API surface:** for each module, identify public classes, functions, factory methods, and their signatures.
- **Schemas & models:** read all Pydantic models, enums, and type aliases. Note fields, defaults, and inheritance.
- **Dependencies:** check `pyproject.toml` for current dependencies and optional extras.
- **Tests:** list test files and test classes to understand coverage scope.
- **Integration tests:** check `tests_integration/` if it exists.
- **Git history since last spec update:** run `git log --oneline --since="$(grep -m1 'Last updated' specs/prd.md | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')" -- doc_intelligence/ tests/` to find changes since the last spec update date.

### 2. Read current specs

Read all three spec files completely:
- `specs/prd.md`
- `specs/engineering_design.md`
- `specs/project_status.md`

### 3. Diff reality vs specs

Compare what you found in step 1 against what the specs say. Identify:

- **Missing from specs:** new modules, classes, functions, schemas, config fields, test files, or dependencies that exist in code but are not documented in specs.
- **Stale in specs:** things described in specs that no longer exist, have been renamed, moved, or have different signatures/behavior.
- **Incorrect phase status:** phases marked as "not started" that have partial or complete implementations, or phases marked "done" that are incomplete.
- **Architecture drift:** data flows, component relationships, or design decisions that have changed.

### 4. Update specs

Apply all necessary changes to bring specs in line with the code. Follow these rules:

**`specs/project_status.md`:**
- Update the phase summary table statuses.
- Check/uncheck individual items based on what actually exists and passes tests.
- Add any new items that were implemented but not listed.
- Remove items that were abandoned or moved to a different phase.
- Update the "Updated:" date to today.

**`specs/engineering_design.md`:**
- Update the package layout tree to match the actual file structure.
- Update class/function signatures to match the actual code.
- Update data flow diagrams if the pipeline has changed.
- Update "Files changed/created" tables for completed phases.
- Mark completed sections clearly (e.g., "Complete" in the section header or a note).
- Add any new design decisions or architectural changes discovered in the code.
- Bump the version number (minor increment, e.g., 0.1.5 -> 0.1.6).
- Update "Last updated" date to today.

**`specs/prd.md`:**
- Update feature descriptions to match what is actually implemented.
- Update code examples if the API has changed.
- Update status markers (e.g., "Complete", "In Progress", "Planned") on feature sections.
- Update non-functional requirements if new patterns have emerged.
- Bump the version number (same as engineering_design.md).
- Update "Last updated" date to today.

**General rules:**
- Do NOT invent features or behaviors — only document what the code actually does.
- Do NOT remove planned/future phases — keep them but ensure their status is accurate.
- Preserve the existing document structure and formatting style.
- Keep descriptions concise — the specs should be a reliable map, not a novel.

### 5. Verify consistency

After updating, verify:
- Phase statuses are consistent across all three spec files.
- Version numbers match between `prd.md` and `engineering_design.md`.
- The package layout in `engineering_design.md` matches reality (`ls -R doc_intelligence/`).
- All "Files changed/created" tables for completed phases are accurate.

### 6. Summary report

Print a summary report in this format:

```
## Spec Update Report

### Changes Made
- [list each spec file and what was changed]

### Key Findings
- [any notable discrepancies found between code and specs]
- [any items that were implemented but not tracked]
- [any tracked items that don't exist in code]

### Phase Status
| Phase | Previous Status | Current Status |
|-------|----------------|----------------|
| ...   | ...            | ...            |

### Spec Versions
- Previous: X.Y → Current: X.Z
- Updated date: YYYY-MM-DD
```
