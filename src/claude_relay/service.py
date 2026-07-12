"""Background service management for agent-relay."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "com.agent-relay.server"
LEGACY_LABEL = "com.claude-relay.server"
SYSTEMD_UNIT = "agent-relay.service"


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _systemd_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def _find_executable() -> str:
    """Find the relay executable path."""
    invoked = Path(sys.argv[0])
    if invoked.name in {"agent-relay", "claude-relay"} and invoked.exists():
        return str(invoked.resolve())
    exe = shutil.which("agent-relay") or shutil.which("claude-relay")
    if exe:
        return exe
    # Fallback: use the Python that's running us + module invocation.
    return sys.executable


def _generate_plist(
    host: str,
    port: int,
    backend: str = "claude",
    model: str = "sonnet",
) -> str:
    exe = _find_executable()
    log_dir = Path.home() / "Library" / "Logs" / "agent-relay"

    # If we found the actual claude-relay binary, use it directly.
    # Otherwise, invoke via python -m.
    if exe.endswith("agent-relay") or exe.endswith("claude-relay"):
        program_args = f"""\
    <array>
        <string>{exe}</string>
        <string>serve</string>
        <string>--port</string>
        <string>{port}</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--backend</string>
        <string>{backend}</string>
        <string>--model</string>
        <string>{model}</string>
        <string>--workers</string>
        <string>1</string>
    </array>"""
    else:
        program_args = f"""\
    <array>
        <string>{exe}</string>
        <string>-m</string>
        <string>claude_relay</string>
        <string>serve</string>
        <string>--port</string>
        <string>{port}</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--backend</string>
        <string>{backend}</string>
        <string>--model</string>
        <string>{model}</string>
        <string>--workers</string>
        <string>1</string>
    </array>"""

    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    <key>ProgramArguments</key>
{program_args}
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir / "stdout.log"}</string>
    <key>StandardErrorPath</key>
    <string>{log_dir / "stderr.log"}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}</string>
        <key>AGENT_RELAY_BACKEND</key>
        <string>{backend}</string>
        <key>AGENT_RELAY_ROUTING_DEFAULT_MODEL</key>
        <string>{model}</string>
    </dict>
</dict>
</plist>
"""


def _generate_systemd_unit(
    host: str,
    port: int,
    backend: str = "claude",
    model: str = "sonnet",
    working_directory: Path | None = None,
) -> str:
    exe = _find_executable()
    if exe.endswith("agent-relay") or exe.endswith("claude-relay"):
        command = exe
    else:
        command = f"{exe} -m claude_relay"
    path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    working_directory = (working_directory or Path.cwd()).resolve()
    return f"""\
[Unit]
Description=Agent Relay ({backend} / {model})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={command} serve --host {host} --port {port} --backend {backend} --model {model} --workers 1
Restart=on-failure
RestartSec=2
Environment="PATH={path}"
Environment="AGENT_RELAY_BACKEND={backend}"
Environment="AGENT_RELAY_ROUTING_DEFAULT_MODEL={model}"
WorkingDirectory={working_directory}

[Install]
WantedBy=default.target
"""


def _env_lines(
    host: str,
    port: int,
    backend: str = "claude",
    model: str = "sonnet",
) -> list[str]:
    """Return the export lines needed for SDK autodiscovery."""
    base = f"http://{host}:{port}"
    lines = [
        f'export ANTHROPIC_BASE_URL="{base}"',
        f'export OPENAI_BASE_URL="{base}/v1"',
    ]
    if backend == "codex":
        lines.extend([
            "unset ANTHROPIC_API_KEY",
            "unset ANTHROPIC_AUTH_TOKEN",
            f'export ANTHROPIC_MODEL="{model}"',
            f'export ANTHROPIC_CUSTOM_MODEL_OPTION="{model}"',
            'export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY="1"',
        ])
    return lines


def _shell_rc() -> Path:
    """Return the user's shell rc file."""
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return Path.home() / ".zshrc"
    return Path.home() / ".bashrc"


def _setup_env(host: str, port: int, backend: str, model: str) -> None:
    """Offer to append SDK env vars to the user's shell rc file."""
    lines = _env_lines(host, port, backend=backend, model=model)
    rc = _shell_rc()
    marker = "# agent-relay"

    # Check if already present.
    if rc.exists() and marker in rc.read_text():
        print(f"  Environment variables already in {rc}")
        return

    print()
    print("To let SDKs auto-discover the relay, add to your shell profile:")
    print()
    for line in lines:
        print(f"  {line}")
    print()

    try:
        answer = input(f"Append to {rc}? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if answer in ("", "y", "yes"):
        with open(rc, "a") as f:
            f.write(f"\n{marker}\n")
            for line in lines:
                f.write(f"{line}\n")
        print(f"  Added to {rc}. Run `source {rc}` or open a new terminal.")
    else:
        print("  Skipped. You can add them manually later.")


def service_install(
    host: str,
    port: int,
    backend: str = "claude",
    model: str = "sonnet",
) -> None:
    system = platform.system()
    if system not in {"Darwin", "Linux"}:
        print("Error: service management is only supported on macOS and Linux.", file=sys.stderr)
        sys.exit(1)

    if system == "Linux":
        unit = _systemd_path()
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(_generate_systemd_unit(host, port, backend=backend, model=model))
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT], check=True)
        print("Service installed and started.")
        print(f"  Listening on http://{host}:{port}")
        print(f"  Unit:    {unit}")
        print(f"  Logs:    journalctl --user -u {SYSTEMD_UNIT}")
        _setup_env(host, port, backend, model)
        return

    plist = _plist_path()
    log_dir = Path.home() / "Library" / "Logs" / "agent-relay"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Unload existing service if present.
    if plist.exists():
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)

    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(_generate_plist(host, port, backend=backend, model=model))

    subprocess.run(["launchctl", "load", str(plist)], check=True)
    print(f"Service installed and started.")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Plist:  {plist}")
    print(f"  Logs:   {log_dir}/")

    _setup_env(host, port, backend, model)

    print()
    print("The service will auto-start on login. To stop it:")
    print("  agent-relay service uninstall")


def service_restart() -> None:
    if platform.system() == "Linux":
        subprocess.run(["systemctl", "--user", "restart", SYSTEMD_UNIT], check=True)
        print("Service restarted.")
        return
    if platform.system() != "Darwin":
        print("Error: service management is only supported on macOS and Linux.", file=sys.stderr)
        sys.exit(1)

    plist = _plist_path()
    if not plist.exists():
        print("Service is not installed. Run: agent-relay service install")
        sys.exit(1)

    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    subprocess.run(["launchctl", "load", str(plist)], check=True)
    print("Service restarted.")


def service_uninstall() -> None:
    if platform.system() == "Linux":
        unit = _systemd_path()
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT],
            capture_output=True,
        )
        if unit.exists():
            unit.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        print("Service stopped and removed.")
        return
    if platform.system() != "Darwin":
        print("Error: service management is only supported on macOS and Linux.", file=sys.stderr)
        sys.exit(1)

    plist = _plist_path()
    if not plist.exists():
        print("Service is not installed.")
        return

    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    plist.unlink()
    legacy = Path.home() / "Library" / "LaunchAgents" / f"{LEGACY_LABEL}.plist"
    if legacy.exists():
        subprocess.run(["launchctl", "unload", str(legacy)], capture_output=True)
        legacy.unlink()
    print("Service stopped and removed.")


def service_status() -> None:
    if platform.system() == "Linux":
        unit = _systemd_path()
        if not unit.exists():
            print("Service is not installed.")
            print("  Run: agent-relay service install")
            return
        subprocess.run(
            ["systemctl", "--user", "status", SYSTEMD_UNIT, "--no-pager"],
            check=False,
        )
        return
    if platform.system() != "Darwin":
        print("Error: service management is only supported on macOS and Linux.", file=sys.stderr)
        sys.exit(1)

    plist = _plist_path()
    if not plist.exists():
        print("Service is not installed.")
        print(f"  Run: agent-relay service install")
        return

    result = subprocess.run(
        ["launchctl", "list", LABEL],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Service is installed and running.")
        # Parse PID from output.
        for line in result.stdout.strip().splitlines():
            if "PID" in line or line.strip().startswith('"PID"'):
                print(f"  {line.strip()}")
    else:
        print("Service is installed but not running.")

    print(f"  Plist: {plist}")
    log_dir = Path.home() / "Library" / "Logs" / "agent-relay"
    print(f"  Logs:  {log_dir}/")
