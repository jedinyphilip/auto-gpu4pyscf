"""Check that the build works on this GPU, and was compiled for it.

Kernels built for another architecture still run, by JIT-ing from PTX, so the
architecture report matters as much as the energy.
"""
import os
import re
import subprocess
import sys
import time

import cupy
import gpu4pyscf
import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.lib import cutensor
from pyscf import gto

dev = cupy.cuda.Device()
props = cupy.cuda.runtime.getDeviceProperties(dev.id)
major, minor = props['major'], props['minor']
print(f"GPU        : {props['name'].decode()} (sm_{major}{minor}, "
      f"{props['totalGlobalMem'] / 1e9:.0f} GB)")
print(f"gpu4pyscf  : {gpu4pyscf.__version__}")
print(f"pyscf      : {pyscf.__version__}")
print(f"cupy       : {cupy.__version__}")
print(f"contraction: {'cutensor' if cutensor.cutensor is not None else 'cupy (cutensor missing!)'}")

# Which architectures are actually baked into the binaries? cuobjdump is part of
# the toolkit and the slim image does not carry it, so fall back to the list
# recorded during the build.
def compiled_archs():
    lib = os.path.join(os.path.dirname(gpu4pyscf.lib.__file__), 'libgint.so')
    try:
        elf = subprocess.run(['cuobjdump', '--list-elf', lib],
                             capture_output=True, text=True, check=True).stdout
        return sorted({int(m) for m in re.findall(r'\.sm_(\d+)\.', elf)})
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    recorded = os.path.join(sys.prefix, 'share', 'gpu4pyscf', 'archs.txt')
    if os.path.exists(recorded):
        with open(recorded) as fh:
            return sorted({int(m) for m in re.findall(r'sm_(\d+)', fh.read())})
    return []


archs = compiled_archs()
if archs:
    native = int(f"{major}{minor}") in archs
    print(f"compiled   : {', '.join('sm_%d' % a for a in archs)}"
          f" -> {'native' if native else 'NOT native, kernels will JIT from PTX'}")
else:
    print("compiled   : unknown (no cuobjdump, no recorded arch list)")

mol = gto.M(atom='''O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692;
                    H 0.0000 -0.7572 -0.4692''',
            basis='def2-tzvpp', verbose=0)
t0 = time.perf_counter()
mf = rks.RKS(mol, xc='B3LYP').density_fit()
mf.grids.level = 3
e = mf.kernel()
g = mf.nuc_grad_method().kernel()
dt = time.perf_counter() - t0

print(f"\nDF-B3LYP/def2-TZVPP water: E = {e:.9f} Ha  |grad|max = {abs(g).max():.2e}"
      f"  ({dt:.1f} s incl. warm-up)")
ok = mf.converged and -76.6 < e < -76.3
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
