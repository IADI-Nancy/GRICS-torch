# Agent instructions

Do not create a Git commit unless a human explicitly asks you to commit. Requests to edit files, fix code, transfer changes between branches, or prepare work do not authorize a commit. Leave completed changes uncommitted for human review unless a commit is explicitly requested.

When adding a feature, name new variables consistently with the style used by nearby code and existing variables in the same module. Preserve established naming conventions across related code.

Never use a fallback that is hidden from the user. If a fallback is needed, make it explicit in the code or user-facing output, and explain when and why it is used.

Do not modify README files unless a human explicitly asks you to modify them. A request to add a feature, fix code, or update documentation in general does not authorize README changes.

Use the minimum number of lines for function arguments or outputs.

Do not keep the legacy support unless explicitly asked; ensure the new functions are used in all calls.