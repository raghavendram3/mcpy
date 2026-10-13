"""MPI driver for tests/test_parallel_gcmc.py (not collected by pytest).

    mpiexec -n 4 python tests/_parallel_gcmc_ideal_gas.py N_WALKERS OUT.json

Ideal gas confined to a slab of the box: insertions are proposed over the
whole box volume V, but any atom outside the slab (a fraction ``FRAC`` of V)
costs +1000 kT, so ~90% of insertions are rejected. Detailed balance then
gives a Poisson distribution of N with mean ``FRAC * V * exp(beta * mu)``,
an exact target for the speculative parallel scheme to reproduce.
"""
import json
import sys

import numpy as np
from ase import Atoms

from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble, ParallelGCMC
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector

L = 100.0 ** (1.0 / 3.0)   # box side, V = 100
FRAC = 0.1
MEAN = 5.0                 # target <N> = FRAC * V * z
STEPS = 100000
MU = float(np.log(MEAN / (FRAC * L ** 3)))  # beta = 1 (LJ units, T = 1)


class SlabCalc:
    def get_potential_energy(self, atoms):
        if len(atoms) and atoms.positions[:, 0].max() > FRAC * L:
            return 1000.0
        return 0.0


def factory(walker):
    atoms = Atoms(cell=[L, L, L], pbc=True)
    cell = Cell(atoms)
    ms = MoveSelector([1, 1], [InsertionMove(cell, species=['H'], seed=1),
                               DeletionMove(cell, species=['H'], seed=2)],
                      seed=3)
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ', calculator=SlabCalc(),
        mu={'H': MU}, species=['H'], temperature=1.0, move_selector=ms,
        random_seed=4, traj_file=None, outfile=None)


def main():
    n_walkers, out = int(sys.argv[1]), sys.argv[2]
    driver = ParallelGCMC(factory, steps=STEPS, n_walkers=n_walkers,
                          equilibration_steps=500, seed=7,
                          summary_file=None, global_minimum_file=None)
    result = driver.run()
    if result is not None:
        with open(out, 'w') as fh:
            json.dump(result, fh, default=float)


if __name__ == '__main__':
    main()
