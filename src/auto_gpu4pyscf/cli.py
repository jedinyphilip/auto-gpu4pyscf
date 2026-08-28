"""Drive the menu and the non-interactive commands."""
import argparse
import json
import os
import shlex
import string
import subprocess
import sys
import time

from . import cluster, gpumap, paths, progress, slurm, system, upstream
from .backends import get as get_backend
from .backends.docker import available as docker_available
from .config import Settings
from .ui import (
    ask,
    bold,
    clear,
    confirm,
    dim,
    fmt_dur,
    green,
    human,
    pause,
    red,
    rule,
    spinner,
    when,
    yellow,
)

PAGE = 18
WINDOWS = os.name == "nt"


# --- status

def status_lines(backend, settings):
    """Build the panel above the menu: what is installed, and how old."""
    lines = [f"{dim('backend   '):<14}{backend.name:<8}{backend.location()}"]
    cluster_line = cluster.status_line(settings)
    if cluster_line:
        lines.append(cluster_line)
    if not backend.installed():
        lines.append(red("  not built yet") + dim("   use Rebuild"))
        return lines, False
    lines.append(f"{dim('built     '):<14}{when(backend.built_at())}   {human(backend.size())}")

    info = backend.build_info()
    gpus = system.host_gpus()
    # Every machine this runs on teaches it one more card.
    gpumap.learn_local(gpus)
    if info:
        lines.append(
            f"{dim('gpu4pyscf '):<14}{info.get('gpu4pyscf', '?')}"
            f"   {dim(info.get('git_short', ''))} {dim(info.get('git_date', '')[:10])}"
        )
        archs = " ".join(info.get("archs", [])) or info.get("cuda_arch", "?")
        note = ""
        if gpus and info.get("archs"):
            want = system.sm_name(gpus[0][1])
            note = (
                green("   native for this GPU")
                if want in info["archs"]
                else yellow("   NOT native -- kernels will JIT")
            )
        cuda = info.get("cuda_version") or "?"
        lines.append(f"{dim('compiled  '):<14}{archs}{note}   {dim('cuda ' + cuda)}")
        lines.append(
            f"{dim('stack     '):<14}pyscf {info.get('pyscf', '?')}"
            f"   cupy {info.get('cupy', '?')}   cutensor {info.get('cutensor', '?')}"
        )
    else:
        lines.append(
            yellow("  no build info recorded -- rebuild to show version,")
            + yellow(" commit and architectures")
        )
        cuda = backend.image_env("CUDA_VERSION") if backend.name == "docker" else ""
        if cuda:
            lines.append(f"{dim('cuda      '):<14}{cuda}")
    if gpus:
        lines.append(
            f"{dim('gpu       '):<14}"
            + ", ".join(f"{name} ({system.sm_name(cap)})" for name, cap in gpus)
        )
    else:
        lines.append(f"{dim('gpu       '):<14}" + yellow("no NVIDIA GPU visible on the host"))
    free = disk_free(backend)
    if free is not None:
        lines.append(f"{dim('disk      '):<14}{free} GB free")
    return lines, True


def disk_free(backend):
    candidates = []
    if backend.name == "docker":
        candidates.append(backend.data_root())
    candidates += [str(paths.repo_root()), os.path.expanduser("~")]
    return system.disk_free_gb(*candidates)


# --- rebuild

def run_build(backend, settings, ref=None, keep_cache=False):
    cmd, env = backend.build_command(ref=ref, keep_cache=keep_cache)
    if cmd is None:
        print(red(f"  {env}"))
        return 1
    cmd, env = cluster.wrap_build(cmd, env, settings)
    kind = backend.progress_kind
    bar = progress.CLASSES[kind](progress.load_profile(kind))
    log_path = paths.build_log()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(paths.repo_root()), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    with open(log_path, "w") as log:
        try:
            for line in proc.stdout:
                log.write(line)
                text = line.rstrip("\n")
                if bar.feed(text) and text.strip():
                    bar.clear_line()
                    print(text)
        except KeyboardInterrupt:
            proc.terminate()
            bar.clear_line()
            print(yellow("\n  interrupted"))
            return 130
    code = proc.wait()
    took = bar.finish()
    if code == 0:
        progress.save_profile(kind, bar.durations)
        print(dim(f"  took {fmt_dur(took)}; full log in {log_path}"))
    else:
        print(red(f"  see {log_path} for the full output"))
        for line in log_path.read_text().splitlines()[-15:]:
            print(dim("  " + line[:100]))
    return code


def screen_rebuild(backend, settings):
    clear()
    rule("rebuild")
    info = backend.build_info()
    if info:
        print(
            f"  installed   {info.get('gpu4pyscf', '?')}   "
            f"{dim(info.get('git_short', ''))} {dim(info.get('git_date', '')[:10])}"
        )
    else:
        print("  " + dim("nothing recorded as built yet"))

    with spinner("checking github and pypi"):
        payloads = upstream.fetch()
    verdict = upstream.compare(info, payloads)
    ref = "master"
    if verdict is None:
        print("  " + yellow("could not reach github or pypi -- building master anyway"))
    else:
        _print_upstream(verdict, info)
        print()
        if not verdict["newer"] and info:
            print("  " + green("already up to date."))
            if not confirm("  Rebuild anyway?", default=False):
                return
        print("  Build which ref?")
        print("    " + bold("m") + "  master (latest commit)")
        if verdict["tag"]:
            print("    " + bold("t") + f"  {verdict['tag']} (latest release)")
        print("    " + bold("o") + "  other (type a tag, branch or SHA)")
        print("    " + bold("q") + "  cancel")
        choice = ask("  > ").lower()
        if choice == "t" and verdict["tag"]:
            ref = verdict["tag"]
        elif choice == "o":
            ref = ask("  ref: ") or "master"
        elif choice != "m":
            return

    problems = backend.preflight()
    if cluster.active(settings):
        if backend.name == "docker":
            problems.append("docker cannot build on a cluster: it needs root. "
                            "Switch to the env backend in Settings.")
        elif not cluster.build_arch(settings):
            problems.append("the target architecture is unknown: choose a partition "
                            "with GPUs, or probe one, in Cluster.")
    if problems:
        print()
        for problem in problems:
            print("  " + red(problem))
        if backend.name == "env":
            print("  " + dim("Debian/Ubuntu: sudo apt install cmake gfortran git python3-venv"))
        pause()
        return

    keep = False
    print()
    if backend.name == "docker":
        free = disk_free(backend)
        print(f"  The build peaks near 25 GB; {free if free is not None else '?'} GB free.")
        print("  Afterwards the ~8 GB build cache can be kept or thrown away:")
        print("    " + dim("keep    ") + "a rebuild reuses it and takes ~3 min")
        print("    " + dim("discard ") + "reclaims the space; a rebuild starts from scratch")
        keep = confirm("  Keep the build cache?", default=False)
    else:
        print(f"  Building natively into {backend.location()}")
    if not confirm("  Start the build now?", default=True):
        return
    code = run_build(backend, settings, ref=ref, keep_cache=keep)
    backend.forget()
    print()
    print(green("  build finished") if code == 0 else red(f"  build failed (exit {code})"))
    pause()


def _print_upstream(verdict, info):
    if verdict["tag"]:
        print(f"  release     {verdict['tag']}   {dim(when(verdict['released_at']))}")
    if verdict["pypi_version"]:
        installed = info.get("gpu4pyscf")
        flag = ""
        if installed and verdict["pypi_version"] != installed:
            flag = yellow("  <- newer than what you built")
        print(f"  pypi        {verdict['pypi_version']}{flag}")
    if verdict["master"]:
        current = green("   this is what you built") if verdict["is_current"] else ""
        print(f"  master      {verdict['master'][:7]} {dim(verdict['master_date'][:10])}{current}")
        if verdict["ahead_by"]:
            print("  " + yellow(f"master is {verdict['ahead_by']} commits ahead"))
        if verdict["master_subject"] and not verdict["is_current"]:
            print("  " + dim("head commit ") + verdict["master_subject"][:58])


# --- uninstall

def screen_uninstall(backend):
    clear()
    rule("uninstall")
    targets = backend.uninstall_targets()
    if not targets:
        print("  " + dim("nothing to remove."))
        pause()
        return
    print("  This will remove:")
    for name, size, _ in targets:
        print(f"    {name}   {dim(human(size) if size else 'several GB')}")
    note = (
        "Other images, and the buildkit helper image shared with other projects,\n"
        "  are left alone."
        if backend.name == "docker"
        else "Nothing outside that directory is touched."
    )
    print("\n  " + dim(note) + "\n")
    if not confirm("  Remove them?", default=False):
        return
    for name, _, do in targets:
        with spinner(f"removing {name}"):
            ok = do()
        print(f"    {green('removed') if ok else red('failed ')}  {name}")
    backend.forget()
    pause()


# --- run

def entries_for(path):
    """Return (directories, python files) at path.

    An empty path means the Windows drive list.
    """
    if WINDOWS and path == "":
        drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        return drives, []
    dirs, scripts = [], []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            try:
                if entry.is_dir():
                    dirs.append(entry.name)
                elif entry.name.endswith(".py"):
                    scripts.append(entry.name)
            except OSError:
                continue
    return sorted(dirs, key=str.lower), sorted(scripts, key=str.lower)


def parent_of(path):
    if WINDOWS:
        return os.path.dirname(path.rstrip("\\/")) if len(path.rstrip("\\/")) > 2 else ""
    return os.path.dirname(path.rstrip("/")) or "/"


def browse(start=None):
    """Browse for a .py file, starting at the project root."""
    root = str(paths.repo_root())
    current = start or root
    page = 0
    error = ""
    while True:
        clear()
        rule("run -- pick a python script")
        print("  " + bold(current or "This PC"))
        try:
            dirs, scripts = entries_for(current)
        except (PermissionError, OSError) as exc:
            error, current = f"cannot open: {exc}", parent_of(current)
            continue
        items = [(name, True) for name in dirs] + [(name, False) for name in scripts]
        pages = max(1, (len(items) + PAGE - 1) // PAGE)
        page = min(page, pages - 1)
        window = items[page * PAGE : (page + 1) * PAGE]
        if not items:
            print(dim("  (no subdirectories or .py files here)"))
        for number, (name, is_dir) in enumerate(window, start=page * PAGE + 1):
            print(f"  {number:>3}  " + (dim(name + os.sep) if is_dir else green(name)))
        print()
        if pages > 1:
            print(dim(f"  page {page + 1}/{pages}   n next   p prev"))
        print(dim("  number select   .. up   r project root   ~ home"
                  "   / type a path   q back"))
        if error:
            print("  " + red(error))
            error = ""
        choice = ask("  > ")
        if choice in ("q", ""):
            return None
        if choice == "..":
            current, page = parent_of(current), 0
        elif choice == "~":
            current, page = os.path.expanduser("~"), 0
        elif choice == "r":
            current, page = root, 0
        elif choice == "n" and page + 1 < pages:
            page += 1
        elif choice == "p" and page > 0:
            page -= 1
        elif choice == "/":
            typed = os.path.expanduser(ask("  path: "))
            if os.path.isdir(typed):
                current, page = typed, 0
            elif os.path.isfile(typed):
                return typed
            else:
                error = f"no such path: {typed}"
        elif choice.isdigit() and 1 <= int(choice) <= len(items):
            name, is_dir = items[int(choice) - 1]
            target = name if (WINDOWS and current == "") else os.path.join(current, name)
            if is_dir:
                current, page = target, 0
            else:
                return target
        else:
            error = "not a choice on this page"


def run_script(backend, script, settings):
    args = shlex.split(ask("  arguments (optional): "))
    if cluster.active(settings):
        return submit_script(backend, script, args, settings)
    cmd, env, cwd = backend.run_command(script, args)
    if cmd is None:
        print(red(f"  {env}"))
        pause()
        return
    if (backend.name == "docker" and not WINDOWS and backend.snap_docker()
            and not cwd.startswith(("/home", "/media"))):
        print(yellow(f"  note: docker is a snap and may refuse to mount {cwd};"))
        print(yellow("        if it does, copy the script under your home directory."))
    print(dim("\n  " + " ".join(cmd) + "\n"))
    started = time.time()
    code = subprocess.call(cmd, env=env, cwd=cwd)
    took = fmt_dur(time.time() - started)
    name = os.path.basename(script)
    print()
    print(
        green(f"  {name} finished in {took}")
        if code == 0
        else red(f"  {name} exited {code} after {took}")
    )
    pause()


def screen_run(backend, settings):
    if not backend.installed():
        clear()
        rule("run")
        print("  " + yellow("Nothing built yet -- use Rebuild first."))
        pause()
        return
    script = browse()
    if script:
        run_script(backend, script, settings)


def submit_script(backend, script, args, settings):
    """Submit the script to Slurm instead of running it here."""
    with spinner("sbatch"):
        job_id, message = cluster.submit_run(backend, script, args, settings)
    print()
    if job_id is None:
        print("  " + red(message))
        pause()
        return
    print("  " + green(f"submitted job {job_id}") + dim(f"  {message}"))
    print("  " + dim(f"output: {os.path.dirname(os.path.abspath(script))}/slurm-{job_id}.out"))
    print("  " + dim("watch it under Cluster > my queue"))
    pause()


# --- settings

BACKEND_BLURB = {
    "docker": "portable: Linux and Windows, brings its own CUDA toolkit",
    "env": "native build beside the repo -- Linux only, no container",
}


def screen_settings(settings, backend):
    while True:
        clear()
        rule("settings")
        current = settings.backend
        print("  " + bold("backend"))
        for name, blurb in BACKEND_BLURB.items():
            mark = green(" * ") if name == current else "   "
            print(f"  {mark}{name:<8}{dim(blurb)}")
        from .backends.env import EnvBackend

        missing = EnvBackend(settings).missing_tools() if not WINDOWS else ["(windows)"]
        print("\n  " + dim("env needs on the host: ")
              + (green("all present") if not missing else red("missing " + ", ".join(missing))))
        print("  " + dim("(nvcc is not required -- a matching CUDA toolkit is installed"
                         " from pip when the host has none)"))
        print()
        other = "env" if current == "docker" else "docker"
        print(f"  {bold('1')}  switch backend to {other}")
        print(f"  {bold('2')}  image tag        {settings.image}")
        print(f"  {bold('3')}  environment dir  {settings.env_dir}")
        print(f"  {bold('4')}  where settings live: {dim(str(settings.path))}")
        print(f"  {bold('q')}  back")
        choice = ask("\n  > ").lower()
        if choice in ("q", ""):
            return
        if choice == "1":
            if other == "env" and WINDOWS:
                print("  " + red("the env backend is Linux only; docker is the portable one."))
                pause()
                continue
            if other == "env" and missing:
                print("  " + yellow("note: " + ", ".join(missing)
                                    + " missing -- Rebuild will refuse until installed"))
            settings.backend = other
            settings.save()
            return  # the caller rebuilds the backend object
        if choice == "2":
            settings.image = ask(f"  image tag [{settings.image}]: ", settings.image)
            settings.save()
            backend.forget()
        elif choice == "3":
            typed = ask(f"  environment dir [{settings.env_dir}]: ", settings.env_dir)
            settings.env_dir = os.path.abspath(os.path.expanduser(typed))
            settings.save()
            backend.forget()


# --- scripted use

def command_status(backend, settings, as_json=False):
    if as_json:
        print(json.dumps({
            "backend": backend.name,
            "launcher": settings.launcher,
            "location": backend.location(),
            "installed": backend.installed(),
            "built_at": backend.built_at(),
            "gpus": system.host_gpus(),
            **backend.build_info(),
        }, indent=2))
        return 0
    for line in status_lines(backend, settings)[0]:
        print("  " + line)
    return 0


def command_build(backend, settings, ref=None, keep_cache=False):
    problems = backend.preflight()
    if problems:
        for problem in problems:
            print(red(problem), file=sys.stderr)
        return 1
    return run_build(backend, settings, ref=ref, keep_cache=keep_cache)


def command_run(backend, settings, script, args):
    """Run a script with no prompts, as the example wrappers do."""
    if not os.path.isfile(script):
        print(red(f"no such script: {script}"), file=sys.stderr)
        return 2
    if not backend.installed():
        print(red(f"nothing built yet for the {backend.name} backend; "
                  "run 'auto-gpu4pyscf build' first"), file=sys.stderr)
        return 1
    if cluster.active(settings):
        job_id, message = cluster.submit_run(backend, script, args, settings)
        if job_id is None:
            print(red(message), file=sys.stderr)
            return 1
        print(f"submitted job {job_id}")
        print(f"output: {os.path.dirname(os.path.abspath(script))}/slurm-{job_id}.out")
        return 0
    cmd, env, cwd = backend.run_command(script, args)
    if cmd is None:
        print(red(env), file=sys.stderr)
        return 1
    return subprocess.call(cmd, env=env, cwd=cwd)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="auto-gpu4pyscf",
        description="Build and run gpu4pyscf compiled for the GPU you actually have.",
    )
    parser.add_argument("--backend", choices=("docker", "env"),
                        help="override the configured backend for this command")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--local", action="store_true", help="run here, not through slurm")
    where.add_argument("--slurm", action="store_true", help="submit through slurm")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("menu", help="the interactive menu (the default)")

    status = sub.add_parser("status", help="what is built, for which GPU")
    status.add_argument("--json", action="store_true", dest="as_json")

    build = sub.add_parser("build", help="build for this machine")
    build.add_argument("--ref", default=None, help="branch, tag or SHA (default master)")
    build.add_argument("--keep-cache", action="store_true", dest="keep_cache")

    run = sub.add_parser("run", help="run a python script against the built gpu4pyscf")
    run.add_argument("script")
    run.add_argument("args", nargs=argparse.REMAINDER)

    return parser.parse_args(argv)


# --- main

def unavailable(built, slurm_ok):
    """Why each menu entry cannot be used, keyed by its number.

    An entry absent from the result is available.
    """
    reasons = {}
    if not built:
        reasons["2"] = "nothing built yet"
        reasons["5"] = "nothing to remove"
    if not slurm_ok:
        reasons["4"] = "no slurm on this host"
    # Cluster is not gated on a build: it is where the partition is chosen, and
    # Rebuild needs that to know which architecture to compile for.
    return reasons


MENU = [
    ("1", "Rebuild", "check upstream, then build for this machine"),
    ("2", "Run", "browse for a .py script and run it"),
    ("3", "Settings", "docker or a native env, image tag, paths"),
    ("4", "Cluster", "run builds and jobs through slurm"),
    ("5", "Uninstall", "remove what this tool installed"),
]


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings.load()
    if args.backend:
        settings.backend = args.backend
    if args.slurm:
        settings.launcher = "slurm"
    elif args.local:
        settings.launcher = "local"

    if args.command in ("status", "build", "run"):
        backend = get_backend(settings)
        if args.command == "status":
            return command_status(backend, settings, as_json=args.as_json)
        if args.command == "build":
            return command_build(backend, settings, ref=args.ref,
                                 keep_cache=args.keep_cache)
        return command_run(backend, settings, args.script, args.args)

    if settings.backend == "docker" and not docker_available():
        print(red("docker is not available."))
        print("Install Docker (Linux) or Docker Desktop with the WSL2 backend (Windows),")
        print(f"or switch to the env backend in {settings.path}.")
        return 1
    backend = get_backend(settings)
    while True:
        clear()
        print(bold("  gpu4pyscf"), dim("-- pyscf on the GPU, compiled for this machine"))
        rule()
        lines, built = status_lines(backend, settings)
        for line in lines:
            print("  " + line)
        rule()
        greyed = unavailable(built, slurm.available())
        for key, name, blurb in MENU:
            reason = greyed.get(key)
            if reason:
                print(dim(f"  {key}  {name:<11}{blurb}   ({reason})"))
            else:
                print(f"  {bold(key)}  {name:<11}{dim(blurb)}")
        print(f"  {bold('q')}  Quit")
        choice = ask("\n  > ").lower()
        if choice in ("q", "quit", "exit"):
            return 0
        if choice in greyed:
            continue
        if choice == "1":
            screen_rebuild(backend, settings)
        elif choice == "2":
            screen_run(backend, settings)
        elif choice == "3":
            screen_settings(settings, backend)
            backend = get_backend(settings)
        elif choice == "4":
            cluster.screen_cluster(settings)
        elif choice == "5":
            screen_uninstall(backend)


if __name__ == "__main__":
    sys.exit(main())
