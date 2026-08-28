"""Decide what to submit, and drive the cluster screen.

slurm.py is the mechanism; this is the policy around it.
"""
import os
import shlex

from . import gpumap, paths, slurm
from .ui import ask, bold, clear, confirm, dim, green, pause, red, rule, spinner, yellow


def options_for(settings):
    return slurm.Options.from_dict(settings.slurm)


def save_options(settings, options):
    settings.slurm = options.to_dict()
    settings.save()


def active(settings):
    return settings.launcher == "slurm"


def status_line(settings):
    """Return one line for the panel, or None when running locally."""
    if not active(settings):
        return None
    options = options_for(settings)
    where = options.partition or dim("no partition chosen")
    return (f"{dim('launcher  '):<14}slurm    {where}   {options.gres}"
            f"   {options.cpus} cpus   {options.time}")


def target_capability(settings):
    """Return the capability to compile for, and where it came from.

    Never the login node's own GPU: it usually has none, and when it has one it
    is often not the model the jobs land on.
    """
    options = options_for(settings)
    recorded = settings.slurm.get("capability")
    if recorded:
        return int(recorded), "probed"
    for partition in slurm.partitions():
        if partition.name == options.partition and partition.capability:
            return partition.capability, f"{partition.gpu_name} in {partition.name}"
    return None, ""


def build_arch(settings):
    """Return CUDA_ARCH for a cluster build, or '' to let the script decide."""
    capability, _ = target_capability(settings)
    return slurm.arch_flag(capability) if capability else ""


def wrap_build(command, env, settings):
    """Wrap a build command in srun.

    Login nodes are shared, and the compile wants cores and a GPU for the smoke
    test at the end.
    """
    if not active(settings):
        return command, env
    options = options_for(settings)
    arch = build_arch(settings)
    if arch:
        env = dict(env, CUDA_ARCH=arch)
    return slurm.srun_command(options, command), env


def run_body(backend, script, args, settings):
    """Return the shell a submitted job executes."""
    quoted = " ".join(shlex.quote(a) for a in args)
    script_path = shlex.quote(os.path.abspath(script))
    if backend.name == "env":
        env_sh = shlex.quote(str(backend.root / "env.sh"))
        return f"source {env_sh}\npython {script_path} {quoted}".strip()
    # Clusters run apptainer, not docker, so say so rather than guessing.
    return None


def submit_run(backend, script, args, settings):
    """Submit a script as a job and return (job_id, message)."""
    options = options_for(settings)
    body = run_body(backend, script, args, settings)
    if body is None:
        return None, (
            "the docker backend cannot run on a cluster: docker needs root, so "
            "clusters use apptainer instead. Switch to the env backend."
        )
    workdir = os.path.dirname(os.path.abspath(script))
    name = os.path.splitext(os.path.basename(script))[0][:20] or slurm.JOB_NAME
    text = slurm.batch_script(
        slurm.replace(options, job_name=name),
        body,
        output="slurm-%j.out",
        workdir=workdir,
    )
    return slurm.submit(text, paths.state_dir() / "jobs" / f"{name}.sbatch")


def screen_cluster(settings):
    partitions = None
    while True:
        clear()
        rule("cluster")
        if not slurm.available():
            print("  " + yellow("no slurm on this host") + dim("  (sinfo, sbatch, squeue"
                                                              " are not on PATH)"))
            print("  " + dim("This screen is for submitting work to a cluster from a"))
            print("  " + dim("login node. Everything else keeps working locally."))
            pause()
            return
        if partitions is None:
            with spinner("asking sinfo about partitions"):
                partitions = slurm.partitions()
        options = options_for(settings)
        gpu_map = gpumap.default()
        print(f"  {dim('slurm     ')}{slurm.version() or 'detected'}")
        print(f"  {dim('launcher  ')}{'slurm' if active(settings) else 'local'}")
        print(f"  {dim('partition ')}{options.partition or red('not chosen')}"
              f"   {options.gres}   {options.cpus} cpus   {options.time}"
              + (f"   account {options.account}" if options.account else ""))
        capability, source = target_capability(settings)
        if capability:
            print(f"  {dim('compiling ')}sm_{capability}   {dim('from ' + source)}")
        else:
            print("  " + yellow("target architecture unknown -- choose a partition or probe"))
        print(f"  {dim('gpu map   ')}{gpu_map.summary()}")
        unknown = gpu_map.unknown_models([p.gpu_name for p in partitions])
        if unknown:
            print("  " + yellow("unrecognised gpus: " + ", ".join(unknown))
                  + dim("  -- probe to learn them"))
        if options.modules:
            print(f"  {dim('modules   ')}{' '.join(options.modules)}")
        print()
        toggle = "run locally instead" if active(settings) else "use slurm for builds and runs"
        print(f"  {bold('1')}  {toggle}")
        print(f"  {bold('2')}  choose partition")
        print(f"  {bold('3')}  gpus, cpus, time, account")
        print(f"  {bold('4')}  module loads")
        print(f"  {bold('5')}  probe the gpu with a short job   {dim('(learns the mapping)')}")
        print(f"  {bold('6')}  my queue")
        print(f"  {bold('q')}  back")
        choice = ask("\n  > ").lower()
        if choice in ("q", ""):
            return
        if choice == "1":
            if active(settings):
                settings.launcher = "local"
                settings.save()
            else:
                _enable_slurm(settings)
        elif choice == "2":
            _choose_partition(settings, partitions)
            partitions = None
        elif choice == "3":
            _choose_resources(settings)
        elif choice == "4":
            _choose_modules(settings)
        elif choice == "5":
            _probe(settings)
            partitions = None
        elif choice == "6":
            _queue()


def _enable_slurm(settings):
    """Turn the launcher on, offering to leave the docker backend.

    Docker is not available to users on a shared cluster, so staying on it
    would only fail later.
    """
    settings.launcher = "slurm"
    settings.save()
    if settings.backend == "docker":
        print()
        print("  " + yellow("docker cannot run on a shared cluster: it needs root,"))
        print("  " + yellow("which is why clusters use apptainer or plain modules."))
        print("  " + dim("the env backend builds gpu4pyscf into a directory instead,"))
        print("  " + dim("with no container and no privileges."))
        if confirm("  Switch to the env backend now?", default=True):
            settings.backend = "env"
            settings.save()
        else:
            print("  " + dim("left on docker; Rebuild will refuse until you switch"))


def _choose_partition(settings, found=None):
    if found is None:
        with spinner("asking sinfo about partitions"):
            found = slurm.partitions()
    if not found:
        print("  " + red("sinfo returned nothing"))
        pause()
        return
    print()
    for number, partition in enumerate(found, start=1):
        marker = green(" *") if partition.default else "  "
        print(f"  {number:>3}{marker} {partition.name:<16}{partition.describe():<28}"
              f"{dim(str(partition.nodes) + ' nodes')}")
    print(dim("\n  * after an architecture means it was learned from a probe or"))
    print(dim("    from a machine this tool has seen, not guessed from the name"))
    choice = ask("\n  partition number (or Enter to keep): ")
    if not choice.isdigit() or not 1 <= int(choice) <= len(found):
        return
    partition = found[int(choice) - 1]
    options = options_for(settings)
    gres = options.gres
    if partition.gpu_name:
        count = gres.split(":")[-1] if ":" in gres else "1"
        gres = f"gpu:{partition.gpu_name}:{count}"
    updated = slurm.replace(options, partition=partition.name, gres=gres)
    save_options(settings, updated)
    # A different partition means a different card.
    settings.values["slurm"].pop("capability", None)
    settings.save()


def _choose_resources(settings):
    options = options_for(settings)
    print()
    gpus = ask(f"  gpus per job [{options.gres}]: ", options.gres)
    cpus = ask(f"  cpus per task [{options.cpus}]: ", str(options.cpus))
    walltime = ask(f"  time [{options.time}]: ", options.time)
    account = ask(f"  account [{options.account or 'none'}]: ", options.account)
    memory = ask(f"  memory, e.g. 32G [{options.memory or 'default'}]: ", options.memory)
    try:
        cpus = int(cpus)
    except ValueError:
        cpus = options.cpus
    save_options(
        settings,
        slurm.replace(
            options,
            gres=gpus if ":" in gpus else f"gpu:{gpus}",
            cpus=cpus,
            time=walltime,
            account="" if account.lower() in ("none", "-") else account,
            memory="" if memory.lower() in ("default", "-") else memory,
        ),
    )


def _choose_modules(settings):
    options = options_for(settings)
    print()
    print(dim("  Modules loaded before the build or run, e.g. 'cuda/12.8 gcc/13'."))
    print(dim("  The env backend needs a CUDA toolkit; on a cluster it usually"))
    print(dim("  comes from a module rather than pip."))
    typed = ask(f"  modules [{' '.join(options.modules) or 'none'}]: ",
                " ".join(options.modules))
    modules = () if typed.lower() in ("none", "-") else tuple(typed.split())
    save_options(settings, slurm.replace(options, modules=modules))


def _probe(settings):
    """Ask a compute node what it has, and remember the answer.

    sinfo names the card and nvidia-smi knows the architecture; a job is what
    puts the two in the same output.
    """
    options = options_for(settings)
    if not options.partition:
        print("  " + yellow("choose a partition first"))
        pause()
        return
    print()
    print(dim("  Submitting a one-GPU, five-minute job to read the card's name and"))
    print(dim("  compute capability from a real compute node. This queues like any job."))
    if not confirm("  Submit it?", default=True):
        return
    with spinner(f"waiting for a probe job on {options.partition}"):
        pairs, error = slurm.probe_gpus(options)
    if not pairs:
        print("  " + red(error or "the probe failed"))
        pause()
        return

    parts = options.gres.split(":")
    alias = parts[1] if len(parts) >= 3 else ""
    gpu_map = gpumap.default()
    corrections = []
    for name, capability in pairs:
        corrections += gpu_map.record_alias(
            alias, name, capability, source=f"probe on {options.partition}"
        )
    gpu_map.save()
    settings.values["slurm"]["capability"] = pairs[0][1]
    settings.save()

    print()
    for name, capability in pairs:
        print("  " + green(f"{name} is sm_{capability}"))
    if alias:
        print("  " + dim(f"recorded for '{alias}' too, so sinfo lookups hit it directly"))
    for candidate, guess in corrections:
        print("  " + yellow(f"the built-in table had {candidate} as sm_{guess}; "
                            "the probe wins and is now remembered"))
    pause()


def _queue():
    with spinner("squeue"):
        jobs = slurm.queue()
    print()
    if not jobs:
        print("  " + dim("nothing queued or running"))
        pause()
        return
    for job in jobs:
        state = green(job.state) if job.running else yellow(job.state)
        print(f"  {job.job_id:>10}  {job.name:<20}{state:<12}{job.elapsed:>8}  {dim(job.reason)}")
    choice = ask("\n  job id to cancel (Enter to skip): ")
    if choice.strip():
        ok = slurm.cancel(choice.strip())
        print("  " + (green("cancelled") if ok else red("scancel failed")))
        pause()
