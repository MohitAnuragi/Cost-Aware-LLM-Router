"""`python -m costrouter ...` works everywhere, including where the `router` launcher is blocked."""

from costrouter.cli import app

app(prog_name="router")
