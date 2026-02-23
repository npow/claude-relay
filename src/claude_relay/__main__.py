"""Entry point for `python -m claude_relay` and the `claude-relay` CLI command."""

import argparse
import os


def _cpu_count() -> int:
    """Return the number of usable CPUs (respects cgroup / affinity masks)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(prog="claude-relay", description="OpenAI-compatible API server for Claude Code")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve"], help="Command to run (default: serve)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8082, help="Bind port (default: 8082)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent subprocess requests per worker (default: 10)")
    parser.add_argument("--request-timeout", type=float, default=300, help="Per-request timeout in seconds (default: 300)")
    parser.add_argument("--workers", type=int, default=_cpu_count(), help="Number of uvicorn workers (default: CPU count)")
    args = parser.parse_args()

    import uvicorn

    # Set env vars so worker processes inherit the configuration.
    os.environ["CLAUDE_RELAY_MAX_CONCURRENT"] = str(args.max_concurrent)
    os.environ["CLAUDE_RELAY_REQUEST_TIMEOUT"] = str(args.request_timeout)

    uvicorn.run(
        "claude_relay.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
