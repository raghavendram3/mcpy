"""MPI-parallel GCMC of a Pt nanoparticle in an oxygen reservoir (CPU nodes).

Oxygen is inserted, deleted and displaced around a Pt octahedron held at a
fixed oxygen chemical potential

    mu_O = E(O2) / 2 + DELTA_MU_O

where E(O2) comes from the running potential (so the reference is consistent
with whatever potential is used) and DELTA_MU_O(T, p_O2) is the physical knob,
e.g. from the NIST-JANAF tables: Delta mu_O = 1/2 [Delta G_O2(T, p0) +
k_B T ln(p_O2 / p0)].

All MPI ranks are split by :class:`mcpy.ensembles.ParallelGCMC` into
N_WALKERS independent chains, and the ranks of each chain evaluate trial
moves in parallel (exactly; see ``mcpy/ensembles/parallel_gcmc.py``). Run it
on a 192-core node with ``examples/slurm_parallel_gcmc.sh``, or locally:

    mpirun -n 4 python examples/05_parallel_gcmc_nanoparticle.py

Requirements: ``pip install mpi4py`` (built against the cluster MPI), plus
``pip install -e .[mace]`` for POTENTIAL = 'mace'. POTENTIAL = 'emt' runs
anywhere and is meant for testing the setup, not for chemistry.

Output, per walker ``w``: ``np_O_walker{w}.out`` (step / N / E / acceptance)
and ``np_O_walker{w}.xyz``; overall: ``parallel_gcmc_summary.txt`` (oxygen
uptake <N_O> +/- standard error over walkers, energies, grand potential,
parallel efficiency) and ``global_minimum.xyz`` (lowest grand potential).
"""
import os

from ase.cluster import Octahedron
from mpi4py import MPI

from mcpy.calculators import BaseCalculator
from mcpy.cell import SphericalCell
from mcpy.ensembles import GrandCanonicalEnsemble, ParallelGCMC
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector
from mcpy.utils import derive_mu_gas
from mcpy.utils.logging import configure as configure_logging

# --- parallel layout -------------------------------------------------------
# MPI ranks = N_WALKERS * ranks_per_walker. ranks_per_walker trial moves are
# evaluated at once per walker; that pays off up to ~1 / acceptance ratio
# (check "trials/round" in the summary). On one 192-core node, 192 ranks =
# 8 walkers x 24 ranks, or 16 x 12, are good starting points.
N_WALKERS = 8

# --- potential -------------------------------------------------------------
POTENTIAL = 'mace'           # 'mace' (mace-mp foundation model) or 'emt'
MACE_MODEL = 'medium'        # named mace-mp model or a path to a .model file
MACE_DTYPE = 'float64'
RELAX_STEPS = 50             # LBFGS steps per trial structure
FMAX = 0.1                   # LBFGS force threshold (eV/A)

# --- system ----------------------------------------------------------------
METAL = 'Pt'
OCTAHEDRON = (6, 1)          # ase.cluster.Octahedron(length, cutoff): 140 atoms
VACUUM = 3.0                 # radius of the exchange sphere beyond the NP (A)
R_O_METAL = 2.0              # exclusion radius an inserted O sees (A)
MIN_INSERT = 0.5             # reject insertions closer than this (A)
MAX_O = None                 # optional cap on the number of O atoms

# --- thermodynamics --------------------------------------------------------
TEMPERATURE = 600.0          # K
DELTA_MU_O = -1.0            # eV, relative to E(O2)/2

# --- run -------------------------------------------------------------------
STEPS = 20_000               # GCMC steps per walker
EQUILIBRATION = 5_000        # steps discarded before averaging
SAMPLE_INTERVAL = 10
WRITE_INTERVAL = 10
SEED = 2024                  # every RNG on every rank derives from it


def make_calculator(comm):
    """One calculator per rank, each on its own OMP_NUM_THREADS threads."""
    if POTENTIAL == 'emt':
        from ase.calculators.emt import EMT
        return BaseCalculator(EMT(), RELAX_STEPS, FMAX)

    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    from mace.calculators import mace_mp

    # Rank 0 loads first so a named model is downloaded once, not by every
    # rank at the same time into the same cache directory.
    calc = None
    if comm.Get_rank() == 0:
        calc = mace_mp(model=MACE_MODEL, device='cpu', default_dtype=MACE_DTYPE)
    comm.Barrier()
    if calc is None:
        calc = mace_mp(model=MACE_MODEL, device='cpu', default_dtype=MACE_DTYPE)
    return BaseCalculator(calc, RELAX_STEPS, FMAX)


def main():
    comm = MPI.COMM_WORLD
    # Rank 0 narrates; walker files and the summary carry the results.
    configure_logging(mpi_rank=comm.Get_rank(),
                      level=20 if comm.Get_rank() == 0 else 30)

    calculator = make_calculator(comm)
    mu_o = derive_mu_gas(calculator, 'O2') + DELTA_MU_O

    def gcmc_factory(walker):
        # Every rank of a walker builds the same starting system; the driver
        # gives each rank its own random streams, so the seeds here only
        # need to be valid.
        atoms = Octahedron(METAL, *OCTAHEDRON)
        cell = SphericalCell(atoms, vacuum=VACUUM,
                             species_radii={METAL: R_O_METAL, 'O': 0.0})
        moves = MoveSelector(
            [1, 1, 1],
            [InsertionMove(cell, species=['O'], min_insert=MIN_INSERT,
                           max_atoms=MAX_O, seed=1),
             DeletionMove(cell, species=['O'], seed=2),
             DisplacementMove(species=['O'], max_displacement=0.5, seed=3)],
            seed=4,
        )
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[cell],
            calculator=calculator,
            mu={'O': mu_o},
            species=['O'],
            units_type='metal',
            temperature=TEMPERATURE,
            move_selector=moves,
            random_seed=5,
            outfile=f'np_O_walker{walker}.out',
            traj_file=f'np_O_walker{walker}.xyz',
            outfile_write_interval=WRITE_INTERVAL,
            trajectory_write_interval=WRITE_INTERVAL,
        )

    result = ParallelGCMC(
        gcmc_factory,
        steps=STEPS,
        # Fewer ranks than walkers (a laptop test): one rank per walker.
        n_walkers=min(N_WALKERS, comm.Get_size()),
        equilibration_steps=EQUILIBRATION,
        sample_interval=SAMPLE_INTERVAL,
        seed=SEED,
    ).run()

    if result is not None:  # world rank 0
        for pooled in result['pooled']:
            n_o, err = pooled['stats']['N_O']
            print(f"T={TEMPERATURE} K, mu_O={mu_o:.4f} eV "
                  f"(Delta mu_O={DELTA_MU_O}): <N_O> = {n_o:.2f} +/- {err:.2f}")


if __name__ == '__main__':
    main()
