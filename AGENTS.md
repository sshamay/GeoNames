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
