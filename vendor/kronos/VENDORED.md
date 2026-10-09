# Vendored: Kronos

- Upstream: https://github.com/shiyu-coder/Kronos
- Commit: 67b630e67f6a18c9e9be918d9b4337c960db1e9a (2026-04-13)
- License: MIT (see LICENSE)
- Contents: the upstream `model/` package only, **unmodified**.

Upstream imports itself as a top-level package named `model` (`from model.module import *`),
so `candlecast.kronos_loader` puts `vendor/kronos` on `sys.path` before importing it.
To update: replace `model/` from a newer upstream checkout and update the commit above.
