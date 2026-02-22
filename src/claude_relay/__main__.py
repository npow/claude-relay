"""Entry point for `python -m claude_relay` and the `claude-relay` CLI command."""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="claude-relay", description="OpenAI-compatible API server for Claude Code")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve"], help="Command to run (default: serve)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8082, help="Bind port (default: 8082)")
    args = parser.parse_args()

    import uvicorn

    from .server import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
