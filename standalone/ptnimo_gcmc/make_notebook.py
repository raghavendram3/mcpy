"""Regenerate PtNiMo_GCMC_colab.ipynb from ptnimo_gcmc.py (keeps them in sync).

    python make_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


script = open(os.path.join(HERE, "ptnimo_gcmc.py")).read()

cells = [
    md("""
# GCMC of a PtNiMo octahedral nanoparticle under an oxygen chemical potential

Grand canonical Monte Carlo of a ~500-atom Pt–Ni–Mo octahedron (default:
489 atoms, Mo-Pt₃Ni composition Pt:Ni:Mo = 73.4 : 25.0 : 1.6, Mo on vertex/edge
sites) exchanging O with an O₂ reservoir at fixed

$$\\mu_\\mathrm{O} = \\tfrac12 E_{\\mathrm{O_2}} + \\Delta\\mu_\\mathrm{O}(T, p_{\\mathrm{O_2}})$$

with the MACE-MP foundation potential on the Colab GPU. Moves: O insertion,
O deletion, O displacement and metal–metal swaps (Pt↔Ni, Pt↔Mo, Ni↔Mo), each
followed by a local FIRE relaxation of the trial structure.

**Before you start:** *Runtime → Change runtime type → GPU* (T4 works; L4/A100 are faster).

Results go to Google Drive and the run checkpoints itself. If Colab disconnects,
re-run all cells: it continues from the last checkpoint.
"""),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md("## 1. Install"),
    code("""
import os
# N changes on every accepted move; avoid CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
!pip -q install mace-torch

# Optional: cuEquivariance kernels, ~2x faster MACE on GPU. Then set enable_cueq=True below.
INSTALL_CUEQ = False
if INSTALL_CUEQ:
    !pip -q install cuequivariance-torch cuequivariance-ops-torch-cu12
"""),
    md("## 2. Output folder on Google Drive"),
    code("""
from google.colab import drive
drive.mount("/content/drive")
# one folder per (T, Delta mu_O)
RUN_DIR = "/content/drive/MyDrive/ptnimo_gcmc/dmuO_-0.50_T300"
os.makedirs(RUN_DIR, exist_ok=True)
"""),
    md("## 3. The GCMC code\n\n"
       "The stand-alone script, written to `ptnimo_gcmc.py` in the Colab VM."),
    code("%%writefile ptnimo_gcmc.py\n" + script),
    md("""
## 4. Settings

The main physical knobs are `temperature` and `delta_mu_O` (eV, relative to ½E(O₂)
from the same potential). Leave `delta_mu_O=None` to compute it from `temperature` and
`p_O2_atm` (Reuter & Scheffler JANAF table, 100–1000 K). `mu_O_correction` lets you
shift μ_O, e.g. for the GGA O₂ overbinding error.

Time per step is set mostly by `relax_steps` × (time per force call, ~0.05–0.2 s for
500 atoms on a T4 in float32). Run section 5 with `steps=20` first to measure it.
"""),
    code("""
import importlib, ptnimo_gcmc
importlib.reload(ptnimo_gcmc)
from ptnimo_gcmc import Config, GCMC

cfg = Config(
    # nanoparticle
    octahedron_length=9, octahedron_cutoff=0,          # 489 atoms
    composition={"Pt": 0.734, "Ni": 0.250, "Mo": 0.016},
    mo_placement="surface",
    # thermodynamics
    temperature=300.0,
    delta_mu_O=-0.50,          # None -> from (temperature, p_O2_atm)
    p_O2_atm=1.0,
    mu_O_correction=0.0,
    # potential
    model="medium-mpa-0", dtype="float32", enable_cueq=False,
    # trial relaxation
    relax_steps=40, relax_fmax=0.10, relax_radius=6.0,
    # moves
    move_weights={"insert": 1.0, "delete": 1.0, "displace": 0.5, "swap": 0.5},
    # run
    steps=5000, equilibration=1000,
    traj_every=10, checkpoint_every=25, print_every=10,
    seed=2024, out_dir=RUN_DIR,
)
"""),
    md("## 5. Run (re-run this cell after a disconnect to resume)"),
    code("""
sim = GCMC(cfg)
summary = sim.run()
"""),
    md("## 6. Results"),
    code("""
import numpy as np, matplotlib.pyplot as plt
d = np.genfromtxt(f"{RUN_DIR}/log.csv", delimiter=",", names=True, dtype=None, encoding=None)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(d["step"], d["N_O"]); ax[0].set_xlabel("MC step"); ax[0].set_ylabel("N_O")
ax[1].plot(d["step"], d["Omega_eV"]); ax[1].set_xlabel("MC step")
ax[1].set_ylabel("Ω = E − μ_O N_O (eV)")
for a in ax: a.axvline(cfg.equilibration, color="gray", ls="--", lw=1)
plt.tight_layout(); plt.show()
print({k: v for k, v in summary.items() if k.startswith(("mean", "err"))})
"""),
    code("""
from ase.io import read
from ase.visualize.plot import plot_atoms
best = read(f"{RUN_DIR}/best.extxyz")
print(best.get_chemical_formula(), best.info)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plot_atoms(best, ax[0], radii=0.8, rotation="0x,0y,0z")
plot_atoms(best, ax[1], radii=0.8, rotation="45x,45y,0z")
for a in ax: a.axis("off")
plt.show()
"""),
    md("""
## 7. Scanning Δμ_O

One run per chemical potential, each in its own folder (Colab GPU time is limited, so
run them in separate sessions or reduce `steps`):

```python
for dmu in [-1.0, -0.75, -0.5, -0.25]:
    c = Config(**{**cfg.__dict__, "delta_mu_O": dmu,
                  "out_dir": f"/content/drive/MyDrive/ptnimo_gcmc/dmuO_{dmu:+.2f}_T300"})
    GCMC(c).run()
```
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
with open(os.path.join(HERE, "PtNiMo_GCMC_colab.ipynb"), "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote PtNiMo_GCMC_colab.ipynb")
