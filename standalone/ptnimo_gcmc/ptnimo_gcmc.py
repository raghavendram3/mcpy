#!/usr/bin/env python
"""Grand canonical Monte Carlo of a PtNiMo octahedral nanoparticle in O2.

Stand-alone: needs only ASE, NumPy, SciPy and mace-torch (for the GPU
potential). Written for one GPU on Google Colab; see
``PtNiMo_GCMC_colab.ipynb`` next to this file.

What it samples
---------------
A ~500-atom PtNiMo octahedron (default: ase.cluster.Octahedron length 9,
489 atoms, Mo-Pt3Ni composition Pt:Ni:Mo = 73.4:25.0:1.6) in contact with
an O2 reservoir at fixed oxygen chemical potential

    mu_O = E(O2)/2 + DELTA_MU_O + MU_O_CORRECTION

E(O2) is computed with the same potential, so only DELTA_MU_O(T, p_O2) is
a physical knob. Give it directly, or let it be computed from (T, p_O2) with
the JANAF-based table of Reuter & Scheffler, PRB 65, 035406 (2001).

For electrochemical conditions (e.g. the oxygen reduction reaction) give
``potential_V_RHE`` instead. O is then in equilibrium with water at that
potential (computational hydrogen electrode, Norskov et al., JPCB 108,
17886 (2004)), and ``mu_O_reference`` picks how mu_O is evaluated:

    'H2O': mu_O = G(H2O) - G(H2) + 2 e U      (no O2 energy needed)
    'O2' : mu_O = E(O2)/2 + Delta mu_O(T, 1 atm) + 2 e (U - 1.229 V)

Both are computed and logged; they differ by the potential's error on the
water formation energy, chiefly the GGA O2 overbinding.

Moves (one trial per MC step, chosen by weight):

* ``insert``  - O at a uniform random point of the free adsorption shell
* ``delete``  - a random O in the shell
* ``displace``- random O shifted by up to ``displace_max`` A
* ``swap``    - exchange two metal atoms of different species (Pt<->Ni,
                Pt<->Mo, Ni<->Mo), so O can drive surface segregation.
                With ``swap_depth`` set, only metal atoms within that depth
                below the surface take part (interior swaps change nothing
                O can see and cost a full trial each).

The ``insert`` and ``delete`` weights must be equal: the acceptance rules
below assume both are proposed equally often.

Every trial structure is relaxed (FIRE) before its energy is used: the
hybrid relaxed-trial GCMC of Senftle et al. Relaxation is local by default:
only atoms within ``relax_radius`` of the modified site move, which keeps a
trial at a few dozen force calls on a 500-atom particle.

Acceptance (beta = 1/kT, Lambda = thermal de Broglie wavelength of O,
N = O atoms in the shell, V = free volume of the shell):

    insert : min(1, V / (Lambda^3 (N+1)) exp(-beta (dE - mu_O)))
    delete : min(1, Lambda^3 N / V'     exp(-beta (dE + mu_O)))
    others : min(1, exp(-beta dE))

The shell is the region outside the convex hull of the metal atoms (down to
``shell_depth`` below its facets, to allow subsurface O) and at most
``shell_height`` above it, minus spheres of radius ``r_excl_metal`` /
``r_excl_O`` around every atom. V is measured on a fixed set of sample
points, so it is a deterministic function of the configuration.

Output (in ``out_dir``): ``log.csv`` (one row per step), ``traj.extxyz``,
``best.extxyz`` (lowest grand potential Omega = E - mu_O N_O),
``checkpoint.extxyz`` + ``checkpoint.json`` (restart: just run again with
the same ``out_dir``), ``summary.json``.

Usage
-----
    python ptnimo_gcmc.py --delta-mu-O -0.5 --temperature 300 --steps 5000 \\
        --out /content/drive/MyDrive/ptnimo_run1

or from Python::

    from ptnimo_gcmc import Config, GCMC
    GCMC(Config(delta_mu_O=-0.5, steps=5000, out_dir='run1')).run()
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.build import molecule
from ase.cluster import Octahedron
from ase.constraints import FixAtoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE
from scipy.spatial import ConvexHull, cKDTree

METALS = ("Pt", "Ni", "Mo")

# Delta mu_O(T, p0 = 1 atm) in eV, from Reuter & Scheffler, PRB 65, 035406
# (2001), Table I (JANAF thermochemical tables).
DELTA_MU_O_TABLE = {
    100: -0.08, 200: -0.17, 300: -0.27, 400: -0.38, 500: -0.50,
    600: -0.61, 700: -0.73, 800: -0.85, 900: -0.98, 1000: -1.10,
}


# Computational hydrogen electrode: ZPE - TS (eV) at 298 K, Norskov et al.,
# JPCB 108, 17886 (2004). H2O(l) is taken as the gas at 0.035 bar.
ZPE_MINUS_TS = {"H2O": 0.56 - 0.67, "H2": 0.27 - 0.41}
EQUILIBRIUM_POTENTIAL_V = 1.229   # O2 + 4(H+ + e-) <-> 2 H2O


def delta_mu_O_from_pressure(temperature: float, p_O2_atm: float) -> float:
    """Delta mu_O(T, p) = Delta mu_O(T, p0) + 1/2 kT ln(p / p0)."""
    ts = sorted(DELTA_MU_O_TABLE)
    if not ts[0] <= temperature <= ts[-1]:
        raise ValueError(f"T = {temperature} K outside the table "
                         f"({ts[0]}-{ts[-1]} K); give delta_mu_O directly")
    base = float(np.interp(temperature, ts, [DELTA_MU_O_TABLE[t] for t in ts]))
    return base + 0.5 * units.kB * temperature * math.log(p_O2_atm)


@dataclass
class Config:
    # --- nanoparticle -------------------------------------------------------
    octahedron_length: int = 9          # (9, 0) -> 489 atoms
    octahedron_edge_nm: Optional[float] = None  # set: overrides length (19 -> 4.93 nm)
    octahedron_cutoff: int = 0          # corner truncation
    lattice_constant: float = 3.87      # A, ~Pt3Ni; relaxed away at start
    composition: dict = field(default_factory=lambda: {
        "Pt": 0.734, "Ni": 0.250, "Mo": 0.016})
    mo_placement: str = "surface"       # 'surface' (lowest CN first) | 'random'
    initial_structure: Optional[str] = None  # file to start from instead

    # --- thermodynamics -----------------------------------------------------
    temperature: float = 300.0          # K
    delta_mu_O: Optional[float] = None  # eV; None -> from potential or (T, p_O2_atm)
    p_O2_atm: float = 1.0
    potential_V_RHE: Optional[float] = None  # electrochemical: O in equilibrium with H2O at U
    mu_O_reference: str = "H2O"         # with potential_V_RHE: 'H2O' or 'O2' route
    mu_O_correction: float = 0.0        # eV per O, e.g. GGA O2 overbinding

    # --- potential ----------------------------------------------------------
    model: str = "medium-mpa-0"         # any mace_mp model name or path; 'emt' = test only
    device: str = "auto"                # 'auto' | 'cuda' | 'cpu'
    dtype: str = "float32"              # float32: fast on T4/L4; float64 on A100 if wanted
    enable_cueq: bool = False           # needs cuequivariance-torch installed

    # --- relaxation of trial structures ------------------------------------
    relax_steps: int = 40
    relax_fmax: float = 0.10            # eV/A
    relax_radius: Optional[float] = 6.0  # A; None = relax all atoms
    initial_relax_steps: int = 300
    initial_relax_fmax: float = 0.03

    # --- moves --------------------------------------------------------------
    move_weights: dict = field(default_factory=lambda: {
        "insert": 1.0, "delete": 1.0, "displace": 0.5, "swap": 0.5})
    displace_max: float = 0.7           # A
    swap_depth: Optional[float] = None  # A below the surface; None = whole particle
    shell_height: float = 2.6           # A above the metal hull facets
    shell_depth: float = 0.0            # A below them (>0: subsurface O)
    r_excl_metal: float = 1.6           # A, no O inserted closer to a metal
    r_excl_O: float = 2.2               # A, ... or to another O
    min_distance: float = 1.0           # A, displaced O closer than this: reject
    volume_points: int = 200_000

    # --- run ----------------------------------------------------------------
    steps: int = 5000
    equilibration: int = 1000           # steps excluded from averages
    traj_every: int = 10
    checkpoint_every: int = 25
    print_every: int = 10
    seed: int = 2024
    out_dir: str = "ptnimo_gcmc_run"


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def make_calculator(cfg: Config):
    if cfg.model == "emt":  # for testing the machinery only; EMT has no Mo
        from ase.calculators.emt import EMT
        return EMT()
    from mace.calculators import mace_mp
    kwargs = dict(model=cfg.model, device=resolve_device(cfg.device),
                  default_dtype=cfg.dtype)
    if cfg.enable_cueq:
        # The cuEquivariance CUDA kernels can be installed yet unloadable
        # (e.g. libnvrtc of another CUDA version); MACE then crashes on the
        # first forward pass. Probe with a tiny molecule and fall back.
        try:
            calc = mace_mp(**kwargs, enable_cueq=True)
            probe = molecule("H2O")
            probe.center(vacuum=5.0)
            probe.calc = calc
            probe.get_forces()
            print("cuEquivariance kernels active", flush=True)
            return calc
        except Exception as exc:
            print(f"cuEquivariance unusable ({type(exc).__name__}: {exc}); "
                  "using the standard MACE kernels", flush=True)
    return mace_mp(**kwargs)


def build_nanoparticle(cfg: Config, rng: np.random.Generator) -> Atoms:
    """Octahedron with the requested composition. Mo goes to the lowest
    coordinated sites (vertices, then edges) for ``mo_placement='surface'``,
    as found for Mo-Pt3Ni octahedra; Ni is distributed randomly."""
    length = cfg.octahedron_length
    if cfg.octahedron_edge_nm is not None:
        # an edge of `length` atoms spans (length - 1) nearest-neighbour gaps
        length = 1 + int(round(10 * cfg.octahedron_edge_nm
                               / (cfg.lattice_constant / math.sqrt(2))))
    atoms = Octahedron("Pt", length=length,
                       cutoff=cfg.octahedron_cutoff,
                       latticeconstant=cfg.lattice_constant)
    n = len(atoms)
    frac = {s: float(cfg.composition.get(s, 0.0)) for s in METALS}
    total = sum(frac.values())
    n_mo = int(round(frac["Mo"] / total * n))
    n_ni = int(round(frac["Ni"] / total * n))

    cn = np.bincount(neighbor_list("i", atoms, 0.85 * cfg.lattice_constant),
                     minlength=n)
    tiebreak = rng.random(n)
    if cfg.mo_placement == "surface":
        order = np.lexsort((tiebreak, cn))      # lowest CN first
    elif cfg.mo_placement == "random":
        order = rng.permutation(n)
    else:
        raise ValueError("mo_placement must be 'surface' or 'random'")
    symbols = np.array(["Pt"] * n, dtype=object)
    symbols[order[:n_mo]] = "Mo"
    rest = rng.permutation(order[n_mo:])
    symbols[rest[:n_ni]] = "Ni"
    atoms.set_chemical_symbols(list(symbols))
    atoms.center(vacuum=12.0)
    atoms.pbc = False
    return atoms


def save_frame(path: str, atoms: Atoms, append: bool = False, **info) -> None:
    """Write one extxyz frame with ``info`` (energy, step, ...) in its
    comment line."""
    frame = atoms.copy()
    frame.calc = None
    frame.info.update(info)
    write(path, frame, format="extxyz", append=append)


class Shell:
    """Adsorption region around the particle and its free volume."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        # Fixed sample points in the unit cube; scaled onto the current
        # bounding box, so V is a deterministic function of the structure.
        self._unit_points = rng.random((cfg.volume_points, 3))

    def update(self, atoms: Atoms) -> None:
        metal = np.isin(atoms.get_chemical_symbols(), METALS)
        pos = atoms.positions[metal]
        eq = ConvexHull(pos).equations          # n.x + d <= 0 inside
        self.planes = np.unique(np.round(eq, 4), axis=0)
        pad = self.cfg.shell_height + 0.5
        self.lo = pos.min(0) - pad
        self.hi = pos.max(0) + pad
        self.box_volume = float(np.prod(self.hi - self.lo))

    def height(self, points: np.ndarray) -> np.ndarray:
        """Signed height above the metal hull (negative inside)."""
        points = np.atleast_2d(points)
        return (points @ self.planes[:, :3].T + self.planes[:, 3]).max(axis=1)

    def in_shell(self, points: np.ndarray) -> np.ndarray:
        h = self.height(points)
        return (h >= -self.cfg.shell_depth) & (h <= self.cfg.shell_height)

    def _not_excluded(self, points: np.ndarray, atoms: Atoms) -> np.ndarray:
        ok = np.ones(len(points), dtype=bool)
        is_o = np.array(atoms.get_chemical_symbols()) == "O"
        for mask, r in ((~is_o, self.cfg.r_excl_metal), (is_o, self.cfg.r_excl_O)):
            if mask.any() and r > 0:
                d, _ = cKDTree(atoms.positions[mask]).query(
                    points, distance_upper_bound=r)
                ok &= ~np.isfinite(d)
        return ok

    def is_free(self, points: np.ndarray, atoms: Atoms) -> np.ndarray:
        points = np.atleast_2d(points)
        free = self.in_shell(points)
        if free.any():
            free[free] = self._not_excluded(points[free], atoms)
        return free

    def free_volume(self, atoms: Atoms) -> float:
        pts = self.lo + self._unit_points * (self.hi - self.lo)
        frac = np.count_nonzero(self.is_free(pts, atoms)) / len(pts)
        # floor at one sample point so a full shell never gives V = 0
        return max(frac, 1.0 / len(pts)) * self.box_volume

    def random_free_point(self, atoms, rng, max_tries=20000):
        for _ in range(max_tries // 200):
            pts = self.lo + rng.random((200, 3)) * (self.hi - self.lo)
            free = np.flatnonzero(self.is_free(pts, atoms))
            if len(free):
                return pts[free[0]]
        return None

    def oxygens_in_shell(self, atoms: Atoms) -> np.ndarray:
        idx = np.flatnonzero(np.array(atoms.get_chemical_symbols()) == "O")
        if len(idx) == 0:
            return idx
        return idx[self.in_shell(atoms.positions[idx])]


# ---------------------------------------------------------------------------
# GCMC driver
# ---------------------------------------------------------------------------

class GCMC:
    def __init__(self, cfg: Config, calculator=None):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.rng = np.random.default_rng(cfg.seed)
        self.calc = calculator if calculator is not None else make_calculator(cfg)
        self.shell = Shell(cfg, self.rng)

        T = cfg.temperature
        self.beta = 1.0 / (units.kB * T)
        mass = atomic_masses[atomic_numbers["O"]] * units._amu
        lam = units._hplanck / math.sqrt(2 * math.pi * mass * units._k * T) * 1e10
        self.lambda3 = lam ** 3                                   # A^3

        if cfg.move_weights.get("insert", 0) != cfg.move_weights.get("delete", 0):
            raise ValueError("move_weights: 'insert' and 'delete' must be equal "
                             "(the acceptance rules assume it)")
        self.moves = [m for m, w in cfg.move_weights.items() if w > 0]
        w = np.array([cfg.move_weights[m] for m in self.moves], dtype=float)
        self.move_p = w / w.sum()
        self.attempts = {m: 0 for m in self.moves}
        self.accepted = {m: 0 for m in self.moves}
        self.step_count = 0
        self.best_omega = math.inf

        if not self._load_checkpoint():
            self._fresh_start()
        self.shell.update(self.atoms)
        self.volume = self.shell.free_volume(self.atoms)

    # -- setup ---------------------------------------------------------------

    def _fresh_start(self):
        cfg = self.cfg
        self.half_E_O2 = 0.5 * self._molecule_energy("O2")
        if cfg.potential_V_RHE is not None and cfg.delta_mu_O is None:
            mu = self._electrochemical_mu_O(cfg.potential_V_RHE)
            cfg.delta_mu_O = mu - self.half_E_O2
        elif cfg.delta_mu_O is None:
            cfg.delta_mu_O = delta_mu_O_from_pressure(cfg.temperature, cfg.p_O2_atm)
        self.mu_O = self.half_E_O2 + cfg.delta_mu_O + cfg.mu_O_correction

        if cfg.initial_structure:
            atoms = read(cfg.initial_structure)
            atoms.pbc = False
        else:
            atoms = build_nanoparticle(cfg, self.rng)
        self._log(f"nanoparticle: {atoms.get_chemical_formula()} "
                  f"({len(atoms)} atoms); relaxing ...")
        t0 = time.perf_counter()
        self.E = self._relax(atoms, None, cfg.initial_relax_steps,
                             cfg.initial_relax_fmax)
        self.atoms = atoms
        self._log(f"initial E = {self.E:.4f} eV ({time.perf_counter() - t0:.1f} s)")
        write(os.path.join(cfg.out_dir, "initial.extxyz"), atoms)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as fh:
            json.dump(asdict(cfg), fh, indent=2)
        with open(self._csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["step", "move", "accepted", "N_O", "E_eV", "Omega_eV",
                 "V_free_A3", "seconds"])
        self._log(f"T = {cfg.temperature} K, Delta mu_O = {cfg.delta_mu_O:.4f} eV, "
                  f"E(O2)/2 = {self.half_E_O2:.4f} eV, mu_O = {self.mu_O:.4f} eV")

    def _molecule_energy(self, name: str) -> float:
        mol = molecule(name)
        mol.center(vacuum=8.0)
        return self._relax(mol, None, 200, 0.01)

    def _electrochemical_mu_O(self, potential: float) -> float:
        """mu_O for O in equilibrium with H2O at ``potential`` (V vs RHE)."""
        cfg = self.cfg
        g_h2o = self._molecule_energy("H2O") + ZPE_MINUS_TS["H2O"]
        g_h2 = self._molecule_energy("H2") + ZPE_MINUS_TS["H2"]
        via_h2o = g_h2o - g_h2 + 2.0 * potential
        via_o2 = (self.half_E_O2 + delta_mu_O_from_pressure(cfg.temperature, 1.0)
                  + 2.0 * (potential - EQUILIBRIUM_POTENTIAL_V))
        self._log(f"U = {potential:.3f} V_RHE: mu_O via H2O/H2 = {via_h2o:.4f} eV, "
                  f"via O2 = {via_o2:.4f} eV (difference {via_h2o - via_o2:+.3f} eV)")
        if abs(via_h2o - via_o2) > 0.3:
            self._log("WARNING: the two references disagree by more than 0.3 eV; "
                      "the potential's water formation energy is off. "
                      "Check mu_O_reference.")
        if cfg.mu_O_reference == "H2O":
            return via_h2o
        if cfg.mu_O_reference == "O2":
            return via_o2
        raise ValueError("mu_O_reference must be 'H2O' or 'O2'")

    @property
    def _csv_path(self):
        return os.path.join(self.cfg.out_dir, "log.csv")

    def _log(self, msg):
        print(msg, flush=True)

    # -- energy --------------------------------------------------------------

    def _relax(self, atoms: Atoms, centers, steps, fmax) -> float:
        """Relax in place (only atoms within relax_radius of ``centers`` move
        when given) and return the potential energy."""
        atoms.calc = self.calc
        if steps > 0:
            radius = self.cfg.relax_radius
            if centers is not None and radius is not None:
                c = np.atleast_2d(centers)
                d = np.linalg.norm(atoms.positions[:, None, :] - c[None], axis=2).min(1)
                atoms.set_constraint(FixAtoms(mask=d > radius))
            FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
            atoms.set_constraint()
        return float(atoms.get_potential_energy())

    def _trial_energy(self, trial, centers):
        return self._relax(trial, centers, self.cfg.relax_steps, self.cfg.relax_fmax)

    def _metropolis(self, log_acc: float) -> bool:
        return log_acc >= 0 or math.log(self.rng.random()) < log_acc

    def n_oxygen(self, atoms=None):
        atoms = self.atoms if atoms is None else atoms
        return atoms.get_chemical_symbols().count("O")

    def omega(self):
        return self.E - self.mu_O * self.n_oxygen()

    # -- moves: each returns (accepted, trial_atoms, trial_energy) -----------

    def _move_insert(self):
        point = self.shell.random_free_point(self.atoms, self.rng)
        if point is None:
            return False, None, None
        n = len(self.shell.oxygens_in_shell(self.atoms))
        trial = self.atoms.copy()
        trial += Atoms("O", positions=[point])
        E = self._trial_energy(trial, point)
        log_acc = (math.log(self.volume / (self.lambda3 * (n + 1)))
                   - self.beta * (E - self.E - self.mu_O))
        return self._metropolis(log_acc), trial, E

    def _move_delete(self):
        candidates = self.shell.oxygens_in_shell(self.atoms)
        if len(candidates) == 0:
            return False, None, None
        i = int(self.rng.choice(candidates))
        site = self.atoms.positions[i].copy()
        trial = self.atoms.copy()
        del trial[i]
        volume_after = self.shell.free_volume(trial)
        E = self._trial_energy(trial, site)
        log_acc = (math.log(self.lambda3 * len(candidates) / volume_after)
                   - self.beta * (E - self.E + self.mu_O))
        return self._metropolis(log_acc), trial, E

    def _move_displace(self):
        oxygens = np.flatnonzero(np.array(self.atoms.get_chemical_symbols()) == "O")
        if len(oxygens) == 0:
            return False, None, None
        i = int(self.rng.choice(oxygens))
        old = self.atoms.positions[i].copy()
        new = old + self.rng.uniform(-1, 1, 3) * self.cfg.displace_max
        others = np.delete(self.atoms.positions, i, axis=0)
        if np.linalg.norm(others - new, axis=1).min() < self.cfg.min_distance:
            return False, None, None
        trial = self.atoms.copy()
        trial.positions[i] = new
        E = self._trial_energy(trial, [old, new])
        return self._metropolis(-self.beta * (E - self.E)), trial, E

    def _move_swap(self):
        symbols = np.array(self.atoms.get_chemical_symbols())
        allowed = np.isin(symbols, METALS)
        if self.cfg.swap_depth is not None:
            # depth below the nearest hull facet; set by positions only, so
            # a swap does not change which atoms may be picked
            allowed &= self.shell.height(self.atoms.positions) >= -self.cfg.swap_depth
        present = [s for s in METALS if (allowed & (symbols == s)).any()]
        if len(present) < 2:
            return False, None, None
        pairs = [(a, b) for k, a in enumerate(present) for b in present[k + 1:]]
        a, b = pairs[self.rng.integers(len(pairs))]
        i = int(self.rng.choice(np.flatnonzero(allowed & (symbols == a))))
        j = int(self.rng.choice(np.flatnonzero(allowed & (symbols == b))))
        trial = self.atoms.copy()
        numbers = trial.numbers.copy()
        numbers[i], numbers[j] = numbers[j], numbers[i]
        trial.numbers = numbers
        E = self._trial_energy(trial, trial.positions[[i, j]])
        return self._metropolis(-self.beta * (E - self.E)), trial, E

    # -- loop ----------------------------------------------------------------

    def step(self):
        move = self.moves[self.rng.choice(len(self.moves), p=self.move_p)]
        t0 = time.perf_counter()
        accepted, trial, E = getattr(self, f"_move_{move}")()
        self.attempts[move] += 1
        if accepted:
            self.accepted[move] += 1
            self.atoms, self.E = trial, E
            self.atoms.calc = None
            self.shell.update(self.atoms)
            self.volume = self.shell.free_volume(self.atoms)
            omega = self.omega()
            if omega < self.best_omega:
                self.best_omega = omega
                save_frame(os.path.join(self.cfg.out_dir, "best.extxyz"),
                           self.atoms, energy=self.E, omega=omega,
                           step=self.step_count + 1)
        self.step_count += 1
        return move, accepted, time.perf_counter() - t0

    def run(self, steps: Optional[int] = None):
        cfg = self.cfg
        target = cfg.steps if steps is None else steps
        if self.step_count == 0:
            self.best_omega = self.omega()
            write(os.path.join(cfg.out_dir, "best.extxyz"), self.atoms)
        self._log(f"running steps {self.step_count} -> {target} "
                  f"(device: {resolve_device(cfg.device)})")
        with open(self._csv_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            while self.step_count < target:
                move, acc, dt = self.step()
                writer.writerow([self.step_count, move, int(acc), self.n_oxygen(),
                                 f"{self.E:.6f}", f"{self.omega():.6f}",
                                 f"{self.volume:.2f}", f"{dt:.3f}"])
                if self.step_count % cfg.traj_every == 0:
                    save_frame(os.path.join(cfg.out_dir, "traj.extxyz"),
                               self.atoms, append=True, energy=self.E,
                               omega=self.omega(), step=self.step_count)
                if self.step_count % cfg.print_every == 0:
                    rates = " ".join(f"{m}={self.accepted[m]}/{self.attempts[m]}"
                                     for m in self.moves)
                    self._log(f"step {self.step_count:6d}  N_O={self.n_oxygen():3d}  "
                              f"E={self.E:.4f}  Omega={self.omega():.4f}  "
                              f"{dt:.1f}s/step  acc: {rates}")
                if self.step_count % cfg.checkpoint_every == 0:
                    fh.flush()
                    self._save_checkpoint()
        self._save_checkpoint()
        return self.summary()

    # -- checkpoint / restart ------------------------------------------------

    def _save_checkpoint(self):
        d = self.cfg.out_dir
        write(os.path.join(d, "checkpoint.tmp.extxyz"), self.atoms)
        state = {
            "step": self.step_count, "E": self.E, "mu_O": self.mu_O,
            "half_E_O2": self.half_E_O2, "delta_mu_O": self.cfg.delta_mu_O,
            "attempts": self.attempts, "accepted": self.accepted,
            "best_omega": self.best_omega,
            "rng": self.rng.bit_generator.state,
        }
        with open(os.path.join(d, "checkpoint.tmp.json"), "w") as fh:
            json.dump(state, fh)
        os.replace(os.path.join(d, "checkpoint.tmp.extxyz"),
                   os.path.join(d, "checkpoint.extxyz"))
        os.replace(os.path.join(d, "checkpoint.tmp.json"),
                   os.path.join(d, "checkpoint.json"))

    def _load_checkpoint(self) -> bool:
        d = self.cfg.out_dir
        path = os.path.join(d, "checkpoint.json")
        if not os.path.exists(path):
            return False
        with open(path) as fh:
            state = json.load(fh)
        self.atoms = read(os.path.join(d, "checkpoint.extxyz"))
        self.atoms.pbc = False
        self.step_count = state["step"]
        self.E = state["E"]
        self.mu_O = state["mu_O"]
        self.half_E_O2 = state["half_E_O2"]
        self.cfg.delta_mu_O = state["delta_mu_O"]
        self.attempts.update(state["attempts"])
        self.accepted.update(state["accepted"])
        self.best_omega = state["best_omega"]
        self.rng.bit_generator.state = state["rng"]
        # drop log rows written after the checkpoint, so the log has no gaps
        # or duplicates when the run continues
        with open(self._csv_path) as fh:
            rows = list(csv.reader(fh))
        rows = [rows[0]] + [r for r in rows[1:] if int(r[0]) <= self.step_count]
        with open(self._csv_path, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        self._log(f"resumed from checkpoint at step {self.step_count} "
                  f"(N_O = {self.n_oxygen()}, mu_O = {self.mu_O:.4f} eV)")
        return True

    # -- analysis ------------------------------------------------------------

    def summary(self) -> dict:
        cfg = self.cfg
        data = np.genfromtxt(self._csv_path, delimiter=",", names=True,
                             dtype=None, encoding=None)
        data = np.atleast_1d(data)
        prod = data[data["step"] > cfg.equilibration]
        out = {
            "temperature_K": cfg.temperature, "delta_mu_O_eV": cfg.delta_mu_O,
            "mu_O_eV": self.mu_O, "steps": self.step_count,
            "acceptance": {m: (self.accepted[m] / self.attempts[m]
                               if self.attempts[m] else None) for m in self.moves},
            "final_N_O": self.n_oxygen(),
            "final_formula": self.atoms.get_chemical_formula(),
            "best_omega_eV": self.best_omega,
        }
        if len(prod) >= 20:
            for name in ("N_O", "E_eV", "Omega_eV"):
                mean, err = block_average(np.asarray(prod[name], dtype=float))
                out[f"mean_{name}"] = mean
                out[f"err_{name}"] = err
            out["n_production_steps"] = int(len(prod))
            out["surface_composition_final"] = surface_composition(self.atoms)
        with open(os.path.join(cfg.out_dir, "summary.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        self._log(json.dumps(out, indent=2))
        return out


def block_average(x: np.ndarray, n_blocks: int = 10):
    """Mean and its standard error from ``n_blocks`` contiguous blocks."""
    blocks = np.array_split(x, n_blocks)
    means = np.array([b.mean() for b in blocks if len(b)])
    return float(x.mean()), float(means.std(ddof=1) / math.sqrt(len(means)))


def surface_composition(atoms: Atoms, cutoff: float = 3.1) -> dict:
    """Metal composition of the outer layer (metal CN < 12)."""
    metal = np.flatnonzero(np.isin(atoms.get_chemical_symbols(), METALS))
    sub = atoms[metal]
    cn = np.bincount(neighbor_list("i", sub, cutoff), minlength=len(sub))
    surf = np.array(sub.get_chemical_symbols())[cn < 12]
    return {s: int((surf == s).sum()) for s in METALS}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", help="JSON file with Config fields")
    p.add_argument("--delta-mu-O", type=float)
    p.add_argument("--p-O2-atm", type=float)
    p.add_argument("--potential-V-RHE", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--equilibration", type=int)
    p.add_argument("--model")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", dest="out_dir")
    args = p.parse_args(argv)

    values = {}
    if args.config:
        with open(args.config) as fh:
            values.update(json.load(fh))
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            values[f.name] = v
    cfg = Config(**values)
    GCMC(cfg).run()


if __name__ == "__main__":
    main()
