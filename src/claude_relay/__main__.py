"""Entry point for `python -m claude_relay` and the `agent-relay` CLI command."""

import argparse
import os
import sys

DEFAULT_PORT = 18082


def _cpu_count() -> int:
    """Return the number of usable CPUs (respects cgroup / affinity masks)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(prog="agent-relay", description="OpenAI-compatible API server for agent CLIs")
    sub = parser.add_subparsers(dest="command")

    # --- serve (default) ---
    serve_p = sub.add_parser("serve", help="Start the relay server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})")
    serve_p.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent subprocess requests per worker (default: 10)")
    serve_p.add_argument("--request-timeout", type=float, default=300, help="Per-request timeout in seconds (default: 300)")
    serve_p.add_argument("--workers", type=int, default=_cpu_count(), help="Number of uvicorn workers (default: CPU count)")
    serve_p.add_argument(
        "--backend",
        choices=["claude", "codex"],
        default=os.environ.get("AGENT_RELAY_BACKEND", "claude"),
        help="Backend CLI to execute (default: claude). Codex is reserved for future adapter support.",
    )

    # --- service management ---
    svc_p = sub.add_parser("service", help="Manage background service (macOS launchd)")
    svc_sub = svc_p.add_subparsers(dest="action")
    install_p = svc_sub.add_parser("install", help="Install and start the launchd service")
    install_p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port for the service (default: {DEFAULT_PORT})")
    install_p.add_argument("--host", default="127.0.0.1", help="Bind host for the service (default: 127.0.0.1)")
    install_p.add_argument(
        "--backend",
        choices=["claude", "codex"],
        default=os.environ.get("AGENT_RELAY_BACKEND", "claude"),
        help="Backend CLI to execute (default: claude).",
    )
    svc_sub.add_parser("restart", help="Restart the launchd service")
    svc_sub.add_parser("uninstall", help="Stop and remove the launchd service")
    svc_sub.add_parser("status", help="Show service status")

    args = parser.parse_args()

    # Default to serve when no subcommand given.
    if args.command is None:
        args = parser.parse_args(["serve"])

    if args.command == "serve":
        import uvicorn

        os.environ["CLAUDE_RELAY_MAX_CONCURRENT"] = str(args.max_concurrent)
        os.environ["CLAUDE_RELAY_REQUEST_TIMEOUT"] = str(args.request_timeout)
        os.environ["AGENT_RELAY_BACKEND"] = args.backend

        uvicorn.run(
            "claude_relay.server:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
        )

    elif args.command == "service":
        from claude_relay.service import service_install, service_restart, service_status, service_uninstall

        if args.action == "install":
            service_install(host=args.host, port=args.port, backend=args.backend)
        elif args.action == "restart":
            service_restart()
        elif args.action == "uninstall":
            service_uninstall()
        elif args.action == "status":
            service_status()
        else:
            svc_p.print_help()
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
