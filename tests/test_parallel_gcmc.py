"""ParallelGCMC: exactness of the speculative (first-accepted) parallel
trial scheme and the independent-walker split.

mpi4py is optional, so these skip when it (or ``mpiexec``) is missing.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.ensembles.parallel_gcmc import ParallelGCMC, reseed_ensemble
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector

HERE = os.path.dirname(__file__)


def _lj_gcmc(walker=0, outfile=None):
    atoms = Atoms(cell=[6.0, 6.0, 6.0], pbc=True)
    cell = Cell(atoms)
    ms = MoveSelector(
        [1, 1, 1],
        [InsertionMove(cell, species=['H'], seed=1),
         DeletionMove(cell, species=['H'], seed=2),
         DisplacementMove(species=['H'], max_displacement=0.3, seed=3)],
        seed=4)
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ',
        calculator=LennardJones(sigma=1.0, epsilon=1.0, rc=3.0, smooth=False),
        mu={'H': -3.0}, species=['H'], temperature=2.0, move_selector=ms,
        random_seed=5, traj_file=None, outfile=outfile,
        outfile_write_interval=10)


def test_reseed_replaces_every_stream_independently():
    a, b = _lj_gcmc(), _lj_gcmc()
    # acceptance + selector + 3 moves + 1 cell
    assert reseed_ensemble(a, 1) == 6
    reseed_ensemble(b, 2)
    draws_a = [m.rng.get_uniform() for m in a.move_selector.move_list]
    draws_b = [m.rng.get_uniform() for m in b.move_selector.move_list]
    assert len(set(draws_a)) == 3          # streams within a rank differ
    assert draws_a != draws_b              # and differ between ranks
    assert a.cells[0]._rng.random() != b.cells[0]._rng.random()


def test_single_rank_reproduces_serial_chain(tmp_path):
    """With one rank per walker the speculative round is one serial trial,
    so the run must match GrandCanonicalEnsemble.run bit for bit."""
    MPI = pytest.importorskip("mpi4py.MPI")
    serial_out = tmp_path / "serial.out"
    par_out = tmp_path / "parallel.out"

    serial = _lj_gcmc(outfile=str(serial_out))
    reseed_ensemble(serial, np.random.SeedSequence(9).spawn(1)[0].spawn(1)[0])
    serial.run(300)

    driver = ParallelGCMC(lambda walker: _lj_gcmc(walker, str(par_out)),
                          steps=300, seed=9, comm=MPI.COMM_SELF,
                          summary_file=None, global_minimum_file=None)
    driver.run()

    assert driver.gcmc.E_old == serial.E_old
    assert np.array_equal(driver.gcmc.atoms.positions, serial.atoms.positions)
    assert (driver.gcmc.move_selector.move_acceptance_total
            == serial.move_selector.move_acceptance_total)
    assert (driver.gcmc.move_selector.move_counter_total
            == serial.move_selector.move_counter_total)
    assert par_out.read_text() == serial_out.read_text()


@pytest.mark.parametrize("n_walkers", [1, 2])
def test_speculative_ranks_sample_exact_distribution(tmp_path, n_walkers):
    pytest.importorskip("mpi4py.MPI")
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if mpiexec is None:
        pytest.skip("no mpiexec")
    out = tmp_path / "result.json"
    script = os.path.join(HERE, "_parallel_gcmc_ideal_gas.py")
    proc = subprocess.run([mpiexec, "-n", "4", sys.executable, script,
                           str(n_walkers), str(out)],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 and not out.exists():
        pytest.skip(f"mpiexec unusable here: {proc.stderr[-500:]}")
    result = json.loads(out.read_text())

    assert len(result["walkers"]) == n_walkers
    for w in result["walkers"]:
        assert w["steps"] == 100000
        assert w["trials"] == 200000
        # 4 // n_walkers ranks speculate; mostly-rejected trials must be
        # batched several per round.
        if n_walkers == 1:
            assert w["trials"] / w["rounds"] > 2.0
        # Poisson(5): mean == variance == 5
        assert abs(w["means"]["N_H"] - 5.0) < 0.35
        assert abs(w["stds"]["N_H"] ** 2 - 5.0) < 1.0
    pooled = result["pooled"][0]
    assert pooled["n_walkers"] == n_walkers
    assert abs(pooled["stats"]["N_H"][0] - 5.0) < 0.3
