"""macOS launchd service management for claude-relay."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "com.claude-relay.server"


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _find_executable() -> str:
    """Find the claude-relay executable path."""
    exe = shutil.which("claude-relay")
    if exe:
        return exe
    # Fallback: use the Python that's running us + module invocation.
    return sys.executable


def _generate_plist(host: str, port: int) -> str:
    exe = _find_executable()
    log_dir = Path.home() / "Library" / "Logs" / "claude-relay"

    # If we found the actual claude-relay binary, use it directly.
    # Otherwise, invoke via python -m.
    if exe.endswith("claude-relay"):
        program_args = f"""\
    <array>
        <string>{exe}</string>
        <string>serve</string>
        <string>--port</string>
        <string>{port}</string>
        <string>--host</string>
        <string>{host}</string>
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
    </dict>
</dict>
</plist>
"""


def _env_lines(host: str, port: int) -> list[str]:
    """Return the export lines needed for SDK autodiscovery."""
    base = f"http://{host}:{port}"
    return [
        f'export ANTHROPIC_BASE_URL="{base}"',
        f'export OPENAI_BASE_URL="{base}/v1"',
    ]


def _shell_rc() -> Path:
    """Return the user's shell rc file."""
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return Path.home() / ".zshrc"
    return Path.home() / ".bashrc"


def _setup_env(host: str, port: int) -> None:
    """Offer to append SDK env vars to the user's shell rc file."""
    lines = _env_lines(host, port)
    rc = _shell_rc()
    marker = "# claude-relay"

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


def service_install(host: str, port: int) -> None:
    if platform.system() != "Darwin":
        print("Error: service management is only supported on macOS.", file=sys.stderr)
        sys.exit(1)

    plist = _plist_path()
    log_dir = Path.home() / "Library" / "Logs" / "claude-relay"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Unload existing service if present.
    if plist.exists():
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)

    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(_generate_plist(host, port))

    subprocess.run(["launchctl", "load", str(plist)], check=True)
    print(f"Service installed and started.")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Plist:  {plist}")
    print(f"  Logs:   {log_dir}/")

    _setup_env(host, port)

    print()
    print("The service will auto-start on login. To stop it:")
    print("  claude-relay service uninstall")


def service_restart() -> None:
    if platform.system() != "Darwin":
        print("Error: service management is only supported on macOS.", file=sys.stderr)
        sys.exit(1)

    plist = _plist_path()
    if not plist.exists():
        print("Service is not installed. Run: claude-relay service install")
        sys.exit(1)

    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    subprocess.run(["launchctl", "load", str(plist)], check=True)
    print("Service restarted.")


def service_uninstall() -> None:
    if platform.system() != "Darwin":
        print("Error: service management is only supported on macOS.", file=sys.stderr)
        sys.exit(1)

    plist = _plist_path()
    if not plist.exists():
        print("Service is not installed.")
        return

    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    plist.unlink()
    print("Service stopped and removed.")


def service_status() -> None:
    if platform.system() != "Darwin":
        print("Error: service management is only supported on macOS.", file=sys.stderr)
        sys.exit(1)

    plist = _plist_path()
    if not plist.exists():
        print("Service is not installed.")
        print(f"  Run: claude-relay service install")
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
    log_dir = Path.home() / "Library" / "Logs" / "claude-relay"
    print(f"  Logs:  {log_dir}/")
