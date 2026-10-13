"""MPI-parallel grand canonical Monte Carlo at a fixed chemical potential.

A single Markov chain is sequential, so the cores of a node are put to work
on two levels:

* **Independent walkers.** ``COMM_WORLD`` is split into ``n_walkers`` groups.
  Each group runs its own chain from the same starting structure with its
  own random streams. Walkers are statistically independent, so pooling them
  gives honest error bars on <N_O>, <E> and the grand potential.

* **Speculative trial moves inside a walker.** The ``K`` ranks of a group all
  hold the same configuration ``x``. Each rank proposes one trial move from
  ``x`` with its own random stream, evaluates (and relaxes) it, and runs the
  acceptance test. Ranks are then read in order 0..K-1 and the *first*
  accepted proposal ``w`` wins: it becomes the new state on every rank and the
  proposals after it are thrown away.

  This is exact, not an approximation. A serial chain only reaches trial
  ``j`` after trials ``0..j-1`` were rejected, i.e. with the state still
  ``x`` -- which is exactly the state rank ``j`` proposed from. One parallel
  round therefore reproduces ``w + 1`` consecutive serial trials (``K`` when
  none were accepted), with the same proposal and acceptance distributions.
  Acceptance counters, step counting, output writes and averages are all
  replayed trial by trial in that order, so the outfile is that of a serial
  run. The expected number of serial trials per round is
  ``(1 - (1 - a)**K) / a`` for acceptance ratio ``a``: close to ``K`` when
  ``a << 1/K``, as in GCMC of oxide nanoparticles with relaxed trial moves.

Run with ``mpirun -n <n_walkers * K> python script.py``; see
``examples/05_parallel_gcmc_nanoparticle.py`` and
``examples/slurm_parallel_gcmc.sh``.
"""
import logging
import time
from collections import defaultdict
from random import Random

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from ..utils.random_number_generator import RandomNumberGenerator
from .base_ensemble import write_global_minimum

logger = logging.getLogger(__name__)


def _unique_cells(gcmc):
    """The ensemble's cells plus any separate cell a move samples from,
    each once, in construction order (identical on all ranks)."""
    cells, seen = [], set()
    for cell in list(gcmc.cells) + [getattr(m, 'cell', None)
                                    for m in gcmc.move_selector.move_list]:
        if cell is not None and id(cell) not in seen:
            seen.add(id(cell))
            cells.append(cell)
    return cells


def _rng_owners(gcmc):
    """Every object of ``gcmc`` that may carry a random stream."""
    return ([gcmc, gcmc.move_selector] + list(gcmc.move_selector.move_list)
            + _unique_cells(gcmc))


def reseed_ensemble(gcmc, seed_sequence) -> int:
    """Replace every random stream of ``gcmc`` (acceptance test, move
    selector, moves, cells) with an independent stream spawned from
    ``seed_sequence`` (a :class:`numpy.random.SeedSequence` or an int).

    Ranks of one walker must not share streams -- with identical seeds they
    would all propose the identical move and the speculation would be wasted.
    Returns the number of streams replaced.
    """
    if not isinstance(seed_sequence, np.random.SeedSequence):
        seed_sequence = np.random.SeedSequence(seed_sequence)
    slots = []
    for owner in _rng_owners(gcmc):
        for name in sorted(vars(owner)):
            value = getattr(owner, name)
            if isinstance(value, (RandomNumberGenerator, np.random.Generator)):
                slots.append((owner, name, value))
    for (owner, name, value), child in zip(slots,
                                           seed_sequence.spawn(len(slots))):
        if isinstance(value, RandomNumberGenerator):
            value.random = Random(int(child.generate_state(2, np.uint64)[0]))
        else:
            setattr(owner, name, np.random.default_rng(child))
    return len(slots)


class ParallelGCMC:
    """Run :class:`GrandCanonicalEnsemble` on many MPI ranks at fixed mu.

    Parameters
    ----------
    gcmc_factory : callable
