# CLAUDE.md

## Security Audits

Run vulnerability audits with:

```bash
uv run --with pip-audit pip-audit
```


## Git Commits

- The user makes all commits. Never run `git commit`, `git push` or anything else that changes history.
- Commit messages must never mention Claude or any AI tool: no co-author trailers, no "generated with" lines.
- **`gmtcc`** means "give me the commit command". Reply with a single command the user can paste, in this format:

  ```
  git commit -m "x" -m "y"
  ```

  where `x` is a succinct commit title and `y` is a detailed description of what changed and why, written as a list so it reads clearly on GitHub: one change per line, each starting with a short bracketed title, e.g.

  ```
  git commit -m "Add Melaka SE-16 results" -m "- [Headline results] Official results for the 16th Melaka state election, 1 row per candidate.
  - [PENDING] Notes the tender data's quirks."
  ```

## Sessions

- **`gmid`** means "give me your id". Run `ListAgents` and reply with this session's name only, e.g. `vetro-data-89`.