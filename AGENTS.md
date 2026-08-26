# AGENTS.md

## Git / GitHub workflow (this machine)

The repo is `https://github.com/sshamay/GeoNames` (branch `main`, remote named
`origin`). Authentication is via a GitHub PAT, NOT SSH:

- The PAT is stored in a file at `~/.ssh/githubmac` (the only file there; there
  is no SSH key registered with GitHub, so SSH push fails with `Permission
  denied (publickey)`).
- The PAT is registered with the macOS keychain under `github.com` (host
  `github.com`, username `shsamay`) via `git credential-osxkeychain`. Ordinary
  `git push` / `git pull` work without prompts.

To push to a NEW machine / if credentials are lost, store the PAT in the
keychain without putting the token in shell history or command args:

```sh
printf "protocol=https\nhost=github.com\nusername=shsamay\npassword=$(cat ~/.ssh/githubmac)\n" | git credential-osxkeychain store
```

Commits use identity `shsamay <48283773+sshamay@users.noreply.github.com>`
(GitHub noreply format), set as repo-local git config.

IMPORTANT: Always ASK the user for confirmation before committing and before
pushing. Never commit or push automatically; wait for explicit approval.

## Repository conventions

- `config/config.yaml` is gitignored (local secrets); `config/config.example.yaml`
  is the committed template. Never commit secrets.
- `reports/` (generated AQuA run JSON + `latest.json` + `history.jsonl`) is
  currently committed as-is. May be gitignored later if it becomes noise.

## Toolchain / Python layout (this machine, macOS arm64)

- **Canonical Python**: Homebrew `python3` -> 3.14.6 at `/opt/homebrew/bin/python3`
  (`pip3` -> 3.14). Apple's system `/usr/bin/python3` (3.9.6) must NEVER be
  treated as the dev Python; it exists only for macOS system scripts.
- **Shell alignment**: `~/.zshrc` and `~/.zprofile` both eval
  `brew shellenv`, so every interactive zsh resolves `python3` to 3.14. GUI
  apps / cron / non-shell contexts do NOT source these; in the IDE always pick
  the interpreter explicitly.
- **Project venvs are per-project and version-frozen**: `GeoNames/.venv` and
  `smartspend/.venv` are on Python 3.9.6 (Apple CLT). Do not assume they match
  the global 3.14. If a task needs a modern interpreter, use
  `/opt/homebrew/bin/python3 -m venv .venv`.
- **Global CLI tools**: managed via Homebrew (`node`, `python`, ...) and pipx
  (`playwright` at `~/.local/bin`, isolated venv on 3.14). pipx-installed tools
  run in their own venv, separate from project venvs.
- **PEP 668**: Apple Python and Homebrew Python are externally-managed; bare
  `pip3 install` into them FAILS with "externally-managed-environment". Install
  into a venv, use `pipx` for CLIs, or `pip install --break-system-packages`
  only as a last resort. The earlier `pip3 install --user playwright` attempt
  failed for this reason.
