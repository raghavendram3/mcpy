# PtNiMo nanoparticle GCMC under an oxygen chemical potential (GPU / Colab)

A stand-alone grand canonical Monte Carlo code for one job: oxygen adsorption on,
and O-driven segregation in, a ~500-atom Pt–Ni–Mo octahedral nanoparticle at a
fixed oxygen chemical potential, using the MACE-MP machine-learned potential
on a single GPU (Google Colab).

It does **not** import `mcpy`. All it needs is `ase`, `numpy`, `scipy` and
`mace-torch`.

| File | What it is |
| --- | --- |
| `ptnimo_gcmc.py` | The whole code, about 600 lines. Run it as a script or import `Config` and `GCMC`. |
| `PtNiMo_GCMC_colab.ipynb` | A Colab notebook. It installs MACE, mounts Google Drive, writes the script into the VM, runs it and plots the results. |
| `make_notebook.py` | Regenerates the notebook from the script after you edit the script. |

## Running it on Colab

1. Open `PtNiMo_GCMC_colab.ipynb` in Colab: upload it, or use *File → Open notebook → GitHub*.
2. Select *Runtime → Change runtime type → GPU*.
3. Set `RUN_DIR`, `temperature` and `delta_mu_O`, then run all cells.

The run writes a checkpoint to Drive every `checkpoint_every` steps. When
Colab disconnects, which it will, re-run all cells with the same `RUN_DIR` and
the run continues where it stopped.

## Running it on any machine with a GPU

```sh
pip install mace-torch
python ptnimo_gcmc.py --delta-mu-O -0.5 --temperature 300 --steps 5000 --out run1
python ptnimo_gcmc.py --config my_settings.json --out run1   # any Config field in the JSON
```

## The model

- **Particle.** `ase.cluster.Octahedron(length=9, cutoff=0)`, which has 489 atoms.
  The default composition is Pt:Ni:Mo = 73.4 : 25.0 : 1.6, the composition of the
  Mo-Pt3Ni octahedra of Huang et al., *Science* 348, 1230 (2015).
  - With `mo_placement='surface'`, Mo goes onto the lowest-coordinated sites,
    vertices first and then edges.
  - Ni is placed randomly.
  - To start from your own structure, set `initial_structure`.
- **Reservoir.** μ_O = ½E(O₂) + Δμ_O + `mu_O_correction`.
  - E(O₂) is computed with the same potential, so it matches the energies used in the run.
  - Give Δμ_O directly. Alternatively set `delta_mu_O=None` and it is computed
    from `temperature` and `p_O2_atm`, using the JANAF-based table of Reuter &
    Scheffler, *PRB* 65, 035406 (2001), valid for 100–1000 K.
- **Moves.** One trial move per MC step, chosen by `move_weights`:
  - O insertion
  - O deletion
  - O displacement
  - metal swaps: Pt↔Ni, Pt↔Mo and Ni↔Mo, so O can pull Ni or Mo to the surface
- **Relaxed trials.** Every trial structure is FIRE-relaxed before it is
  accepted or rejected. This is the hybrid scheme of Senftle et al.
  - By default the relaxation is local: only atoms within `relax_radius` (6 Å)
    of the changed site move.
  - `relax_steps` sets the cost of a step. Each relaxation step is one force call
    on about 500 atoms.
- **Acceptance.** Here N is the number of O atoms in the adsorption shell, V is
  the free volume of the shell, and Λ is the thermal wavelength of O.
  - Insertion is accepted with probability min(1, V/(Λ³(N+1))·e^(−β(ΔE−μ_O))).
  - Deletion is accepted with probability min(1, Λ³N/V′·e^(−β(ΔE+μ_O))), where
    V′ is the free volume after the deletion.
  - Displacement and swap moves use the Metropolis rule.
- **Adsorption shell.** This is where O is inserted and deleted.
  - It extends from the convex hull of the metal atoms up to `shell_height`
    (2.6 Å) above its facets. Set `shell_depth > 0` to allow subsurface O.
  - Points within `r_excl_metal` of a metal atom or `r_excl_O` of an O atom are excluded.
  - V is measured on a fixed set of sample points, so it is an exact function of the structure.

## Output files (in `out_dir`)

- `log.csv`: one row per step, with the step, move, whether it was accepted,
  N_O, E, Ω = E − μ_O·N_O, V_free and the time per step.
- `traj.extxyz`: a frame every `traj_every` steps.
- `best.extxyz`: the structure with the lowest Ω seen so far.
- `summary.json`: written at the end. It holds ⟨N_O⟩, ⟨E⟩ and ⟨Ω⟩ after
  `equilibration`, with block-averaged errors, plus acceptance ratios and the
  surface composition.
- `checkpoint.extxyz` and `checkpoint.json`: the restart point, including the
  random-number state.

## Speed and settings

- **Time per step.** On a T4 in float32, expect roughly
  `relax_steps × 0.05–0.15 s` for a 500-atom particle. Run 20 steps first and
  read the `s/step` column.
- **Precision.** Keep `dtype='float32'` on T4 and L4 GPUs, which are slow in
  float64.
- **Faster kernels.** cuEquivariance (`enable_cueq=True`, after installing it in
  the notebook) gives about another 2×.
- **Cheaper relaxations.** Lower `relax_steps`, or raise `relax_fmax`.
  Converged relaxations are better but cost more.
- **How long to run.** One run is one point of the uptake curve. Coverage
  usually needs a few thousand steps to equilibrate at a fixed Δμ_O. Check the
  N_O-vs-step plot before you trust the averages.

## Caveats

- **Relaxed trials are approximate.** With relaxation, the sampling is not in
  strict detailed balance. This is the standard approximation of the hybrid
  scheme. Set `relax_steps=0` for exact GCMC with unrelaxed trials, which
  accepts far fewer moves.
- **Reference error.** MACE-MP is trained on PBE(+U) data. PBE overbinds O₂,
  which shifts μ_O. Use `mu_O_correction` if you have a correction you trust,
  and check the Mo–O energetics against DFT for your system.
- **EMT is for testing only.** `model='emt'` runs on CPU but has no Mo
  parameters, and its oxygen chemistry is not physical.
