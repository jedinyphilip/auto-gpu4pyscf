"""Print colour, prompts, spinners and the small formatters."""
import contextlib
import itertools
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone


def _use_colour():
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":  # ask the console host for ANSI; older cmd.exe refuses
        try:
            import ctypes

            kernel = ctypes.windll.kernel32
            kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


COLOUR = _use_colour()


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOUR else text


def bold(text):
    return c(text, "1")


def dim(text):
    return c(text, "2")


def green(text):
    return c(text, "32")


def yellow(text):
    return c(text, "33")


def red(text):
    return c(text, "31")


def clear():
    if not sys.stdout.isatty():  # piped or redirected: nothing to clear
        return
    if COLOUR:
        sys.stdout.write("\033[2J\033[H")
    else:
        os.system("cls" if os.name == "nt" else "clear")


def rule(title=""):
    width = min(shutil.get_terminal_size((78, 24)).columns, 78)
    if title:
        print(dim("-- " + title + " " + "-" * max(0, width - len(title) - 4)))
    else:
        print(dim("-" * width))


def ask(prompt, default=""):
    try:
        answer = input(bold(prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"
    return answer or default


def confirm(prompt, default=False):
    answer = ask(f"{prompt} {'[Y/n]' if default else '[y/N]'} ").lower()
    return default if not answer else answer.startswith("y")


def pause():
    ask(dim("\npress Enter to go back "))


@contextlib.contextmanager
def spinner(label):
    """Show a spinner while the block runs.

    Progress goes to stderr so that ``status --json`` stays pipeable.
    """
    if not sys.stdout.isatty():
        print("  " + dim(label + " ..."), file=sys.stderr)
        yield
        return
    stop = threading.Event()

    def spin():
        for frame in itertools.cycle("|/-\\"):
            if stop.is_set():
                return
            sys.stderr.write(f"\r  {dim(label)} {frame} ")
            sys.stderr.flush()
            time.sleep(0.12)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=0.5)
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


def fmt_dur(seconds):
    """Format seconds as m:ss, never negative."""
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def human(size):
    """Format bytes as GB or MB, whichever reads better."""
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


def when(iso, now=None):
    """Format an ISO timestamp as '2026-08-27 14:12 (3 hours ago)'."""
    if not iso:
        return "unknown"
    try:
        stamp = datetime.fromisoformat(iso.replace("Z", "+00:00")[:32])
    except ValueError:
        return iso
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    delta = (now or datetime.now(timezone.utc)) - stamp
    if delta.days > 1:
        ago = f"{delta.days} days ago"
    elif delta.days == 1 or delta.seconds >= 3600:
        hours = delta.days * 24 + delta.seconds // 3600
        ago = f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.seconds >= 60:
        ago = f"{delta.seconds // 60} min ago"
    else:
        ago = "just now"
    return f"{stamp.astimezone().strftime('%Y-%m-%d %H:%M')} ({ago})"
