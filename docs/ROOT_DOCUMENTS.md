# Root Document Roles

The repository root intentionally keeps only a small set of Markdown documents. Avoid adding new root-level guides unless they serve a distinct audience.

- `README.md`: project overview, setup, primary workflows, and links into deeper documentation.
- `AGENTS.md`: canonical contributor and AI-agent operating guide for this repository.
- `CONTRIBUTING.md`: human collaboration, branch, PR, and experiment-record expectations.
- `CHANGELOG.md`: concise release/change history.
- `CLAUDE.md`: Claude Code entry point. Scoped to what an agent needs before touching this repository — the guarded operations, the artifact contract, and the CV-LB gap rule. It does not restate `AGENTS.md`.

Removed/retired roles:

- `DEVELOPER.md` was removed from the root because it overlapped with `AGENTS.md` and contained stale workflow commands. If another tool requires a provider-specific instruction file, keep it short and make it point back to `AGENTS.md` instead of duplicating operational rules.
