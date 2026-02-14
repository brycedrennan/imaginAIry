# imaginAIry

AI image generation library. Python >=3.11.

## Quick reference

| What | Command |
|------|---------|
| Setup | `make init` |
| Check all | `make check` (format + lint + typecheck + fast tests) |
| Format | `make af` |
| Lint | `make lint` |
| Typecheck | `make typecheck` |
| Test (no GPU) | `make test-fast` |
| Test (all) | `make test` |

## Stack

- **Deps/env**: uv, pyproject.toml, dependency-groups
- **Lint/fmt**: ruff
- **Types**: ty
- **Test**: pytest (`-m "not gputest"` for non-GPU)
- **Build**: `uv build` / `uv publish`

## Layout

```
imaginairy/          # main package
  cli/               # click CLI (imagine, aimg commands)
  vendored/          # vendored third-party code (excluded from lint/type-check)
  configs/           # YAML configs
tests/               # pytest tests (mirror source layout)
```

## Notes

- GPU tests require the bd box (4090). Files sync via `make sync` (unison).
- `imaginairy/vendored/` is excluded from linting and type-checking.
- ALWAYS run `make test` on the bd box after changes to confirm things are working