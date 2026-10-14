"""Regenerate the Colab notebooks from ptnimo_gcmc.py (keeps them in sync).

    python make_notebook.py

writes PtNiMo_GCMC_colab.ipynb (~500 atoms, gas-phase O2) and
PtNiMo_5nm_ORR_colab.ipynb (5 nm edge, 4579 atoms, ORR potential).
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

small_cells = [
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

five_nm_cells = [
    md("""
# GCMC of a 5 nm PtNiMo octahedron under oxygen reduction (ORR) conditions

Grand canonical Monte Carlo of a Pt–Ni–Mo octahedron with a **4.93 nm edge**
(`ase.cluster.Octahedron` length 19: **4579 atoms**, 1298 of them on the surface)
at **Pt:Ni:Mo = 72 : 14 : 14**, in equilibrium with oxygen at the conditions under
which PtNi(Mo) octahedra are benchmarked experimentally for the ORR
(O₂-saturated 0.1 M HClO₄, room temperature, activity quoted at **0.90 V vs RHE**;
e.g. Huang et al., *Science* 348, 1230 (2015)).

O is in equilibrium with water at that potential (computational hydrogen electrode):

$$\\mu_\\mathrm{O} = G_{\\mathrm{H_2O}} - G_{\\mathrm{H_2}} + 2eU_\\mathrm{RHE}$$

which, with experimental thermochemistry, is
Δμ_O = Δμ_O(298 K, 1 atm) + 2e(U − 1.229 V) ≈ **−0.93 eV** relative to ½E(O₂).
The code evaluates both routes with MACE and prints them; see section 4.

Changes from the ~500-atom notebook, from what that run showed:

* **displacements nearly off** (weight 0.05): they were accepted ~95% of the time but
  changed E by ~0.01 eV, costing a full relaxation each;
* **swaps only in the surface and first subsurface layer** (`swap_depth=3.5` Å): they
  drive the Mo/Ni segregation that lowered Ω most; interior swaps change nothing O can see;
* **swap weight raised** to 1 (insert = delete = 1, which the acceptance rules require);
* **Mo placed randomly**: 14 % Mo is 641 atoms, more than the 210 edge and vertex sites,
  so the "Mo on low-coordination sites" start would make half the surface Mo;
* cheaper trial relaxations (20 FIRE steps, fmax 0.15 eV/Å, radius 5 Å).

**GPU:** 4579 atoms is ~9× the cost per force call of the 489-atom run. Use an
**A100 or L4** (*Runtime → Change runtime type*); a T4 may run out of memory and is
~10× slower. Results go to Google Drive with checkpoints: after a disconnect, re-run
all cells and it continues.
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

# cuEquivariance kernels: ~2x faster MACE on GPU, worth it at this size.
# The kernel package must match the CUDA version PyTorch was built with.
# If they cannot load, CUEQ_OK stays False, and the calculator also checks
# itself and falls back to the standard kernels, so a bad install never crashes.
INSTALL_CUEQ = True
CUEQ_OK = False
if INSTALL_CUEQ:
    import torch
    CUDA_MAJOR = torch.version.cuda.split(".")[0]      # e.g. "12"
    print("PyTorch CUDA:", torch.version.cuda)
    !pip -q install cuequivariance-torch cuequivariance-ops-torch-cu{CUDA_MAJOR}
    try:
        import cuequivariance_ops_torch  # noqa: F401  (the CUDA kernels themselves)
        CUEQ_OK = True
    except Exception as exc:
        print("cuEquivariance kernels unavailable, running without them:", exc)
"""),
    md("## 2. Output folder on Google Drive"),
    code("""
from google.colab import drive
drive.mount("/content/drive")
# one folder per (size, composition, T, potential)
RUN_DIR = "/content/drive/MyDrive/ptnimo_gcmc/5nm_Pt72Ni14Mo14_U0.90_T298"
os.makedirs(RUN_DIR, exist_ok=True)
"""),
    md("## 3. The GCMC code\n\n"
       "The stand-alone script, written to `ptnimo_gcmc.py` in the Colab VM."),
    code("%%writefile ptnimo_gcmc.py\n" + script),
    md("""
## 4. Settings

**Oxygen chemical potential.** `potential_V_RHE=0.90` sets μ_O from the ORR potential.
`mu_O_reference` chooses how:

* `"H2O"` (default): μ_O = G(H₂O) − G(H₂) + 2eU with MACE energies and the standard
  ZPE − TS corrections. It avoids the O₂ molecule, which GGA-trained potentials
  overbind.
* `"O2"`: μ_O = ½E(O₂) + Δμ_O(T, 1 atm) + 2e(U − 1.229 V), the same reference as the
  ~500-atom runs.

The log prints both. If they differ by more than ~0.3 eV, the potential's water
formation energy is off; pick the route you trust, or set `delta_mu_O` directly
(eV relative to ½E(O₂); −0.93 is the experimental-thermochemistry value at 0.90 V).

**What the model leaves out.** At 0.9 V, Pt surfaces in acid also carry OH and water;
this simulation exchanges O only, in vacuum (no solvent, no field). Read N_O as
the oxygen uptake those conditions drive, not the full adsorbate layer.

**Speed.** Run section 5 once with `steps=10` and read the `s/step` column before
committing to a long run.
"""),
    code("""
import importlib, ptnimo_gcmc
importlib.reload(ptnimo_gcmc)
from ptnimo_gcmc import Config, GCMC

cfg = Config(
    # nanoparticle: 5 nm edge -> Octahedron length 19, 4579 atoms
    octahedron_edge_nm=5.0, octahedron_cutoff=0,
    composition={"Pt": 0.72, "Ni": 0.14, "Mo": 0.14},
    mo_placement="random",
    # thermodynamics: ORR conditions
    temperature=298.15,
    potential_V_RHE=0.90,      # V vs RHE
    mu_O_reference="H2O",      # or "O2"; both are printed
    delta_mu_O=None,           # set a number (eV vs E(O2)/2) to override
    mu_O_correction=0.0,
    # potential
    model="medium-mpa-0", dtype="float32", enable_cueq=CUEQ_OK,
    # trial relaxation (cheaper than the 489-atom run)
    relax_steps=20, relax_fmax=0.15, relax_radius=5.0,
    initial_relax_steps=200, initial_relax_fmax=0.05,
    # moves: insert = delete (required); displacements nearly off;
    # swaps only in the surface + first subsurface layer
    move_weights={"insert": 1.0, "delete": 1.0, "swap": 1.0, "displace": 0.05},
    swap_depth=3.5,
    volume_points=1_000_000,
    # run
    steps=3000, equilibration=1000,
    traj_every=10, checkpoint_every=5, print_every=5,
    seed=2024, out_dir=RUN_DIR,
)
"""),
    md("## 5. Run (re-run this cell after a disconnect to resume)"),
    code("""
sim = GCMC(cfg)
summary = sim.run()
"""),
    md("""
## 6. Results

Average only once Ω has stopped drifting, swap acceptance has dropped well below its
early value, and N_O fluctuates around a constant. Move the `equilibration` line to
where that happens and re-run `sim.summary()`.
"""),
    code("""
import numpy as np, matplotlib.pyplot as plt
d = np.genfromtxt(f"{RUN_DIR}/log.csv", delimiter=",", names=True, dtype=None, encoding=None)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].plot(d["step"], d["N_O"]); ax[0].set_ylabel("N_O")
ax[1].plot(d["step"], d["Omega_eV"]); ax[1].set_ylabel("Ω = E − μ_O N_O (eV)")
swaps = d["move"] == "swap"
win = 50
rate = np.convolve(d["accepted"][swaps], np.ones(win) / win, mode="valid")
ax[2].plot(d["step"][swaps][win - 1:], rate); ax[2].set_ylabel(f"swap acceptance ({win}-swap avg)")
for a in ax:
    a.set_xlabel("MC step"); a.axvline(cfg.equilibration, color="gray", ls="--", lw=1)
plt.tight_layout(); plt.show()
print("O per surface metal atom:", d["N_O"][-1] / 1298)
print({k: v for k, v in summary.items() if k.startswith(("mean", "err", "surface"))})
"""),
    code("""
from ase.io import read
from ase.visualize.plot import plot_atoms
best = read(f"{RUN_DIR}/best.extxyz")
print(best.get_chemical_formula(), best.info)
print("surface composition:", ptnimo_gcmc.surface_composition(best))
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plot_atoms(best, ax[0], radii=0.8, rotation="0x,0y,0z")
plot_atoms(best, ax[1], radii=0.8, rotation="45x,45y,0z")
for a in ax: a.axis("off")
plt.show()
"""),
    md("""
## 7. Scanning the potential

One run per potential, each in its own folder:

```python
for U in [0.70, 0.80, 0.90, 1.00]:
    c = Config(**{**cfg.__dict__, "potential_V_RHE": U, "delta_mu_O": None,
                  "out_dir": f"/content/drive/MyDrive/ptnimo_gcmc/5nm_Pt72Ni14Mo14_U{U:.2f}_T298"})
    GCMC(c).run()
```
"""),
]


def save(name, cells, gpu):
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": gpu},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    for c in nb["cells"]:
        lines = c["source"].split("\n")
        c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    with open(os.path.join(HERE, name), "w") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", name)


save("PtNiMo_GCMC_colab.ipynb", small_cells, "T4")
save("PtNiMo_5nm_ORR_colab.ipynb", five_nm_cells, "A100")
