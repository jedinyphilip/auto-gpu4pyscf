# Examples

Six examples taken **verbatim from gpu4pyscf upstream**, each with a shell
wrapper that runs it against your built gpu4pyscf without the menu.

```sh
./00-h2o.sh                              # the configured backend
BOX_ARGS="--backend env"    ./00-h2o.sh  # force the native environment
BOX_ARGS="--backend docker" ./00-h2o.sh  # force the image
BOX_ARGS="--slurm"          ./00-h2o.sh  # submit it as a slurm job
./00-h2o.sh --some-flag                  # anything else goes through to python
```

| | what it does | upstream |
|---|---|---|
| `00-h2o.sh` | energy, gradient, Hessian and thermochemistry for water; start here | `dft/00-h2o.py` |
| `02-h2o_geomopt.sh` | geometry optimisation with geomeTRIC, energy printed each step | `geometry_optimization/02-h2o_geomopt.py` |
| `16-smd_solvent.sh` | solvation free energy in water with SMD: gas phase, then solvated | `solvent/16-smd_solvent.py` |
| `20-dfmp2.sh` | density-fitted MP2 correlation energy on top of Hartree-Fock | `post_HF/20-dfmp2.py` |
| `22-resp_charge.sh` | ESP and RESP atomic charges from a converged density | `properties/22-resp_charge.py` |
| `26-tddft_and_gradient.sh` | five TDDFT excited states and the gradient of one | `tddft/26-tddft_and_gradient.py` |

The `.py` files are unmodified copies carrying their original Apache-2.0
headers: they belong to the PySCF developers, not to this repository. Diff
them against
[pyscf/gpu4pyscf/examples](https://github.com/pyscf/gpu4pyscf/tree/master/examples)
if you want to be sure. Many more live there; anything in that directory runs
here the same way.

Some write a `pyscf.log` beside themselves (that is upstream's `output=` in the
example, not something the wrapper does), so most of the detail lands there
rather than on the terminal. Those files are gitignored.

The wrappers all defer to `_run.sh`, which is a thin call to
`auto-gpu4pyscf run <script>`, the same non-interactive interface you can use
for your own scripts:

```sh
./menu.sh run my_calculation.py           # or auto-gpu4pyscf run, once installed
./menu.sh status --json                   # what is built, machine readable
./menu.sh build --ref v1.8.1              # build a specific upstream tag
```
