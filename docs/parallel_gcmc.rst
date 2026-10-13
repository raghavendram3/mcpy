Parallel GCMC at fixed chemical potential
=========================================

:class:`~mcpy.ensembles.parallel_gcmc.ParallelGCMC` runs one
:class:`GrandCanonicalEnsemble` setup (for example a nanoparticle in an
oxygen reservoir at fixed :math:`\mu_\mathrm{O}`) on many MPI ranks, such as
all 192 cores of a CPU node. One Markov chain is sequential, so the ranks
are used on two levels.

Independent walkers
-------------------

``COMM_WORLD`` is split into ``n_walkers`` groups. Every group runs its own
chain from the same starting structure with its own random streams (all
derived from one ``seed``). The walkers are independent, so at the end the
driver pools their averages of :math:`N_\mathrm{O}`, :math:`E` and
:math:`\Omega = E - \sum_s \mu_s N_s` and reports the standard error of the
mean over walkers. Each walker still has to equilibrate on its own.

Speculative trial moves inside a walker
---------------------------------------

The :math:`K` ranks of one walker all hold the same configuration :math:`x`.
Each proposes one trial move from :math:`x`, relaxes it, computes its energy
and runs the acceptance test. The proposals are then read in rank order and
the first accepted one becomes the new state on all ranks; the proposals
after it are thrown away.

This samples exactly the same distribution as the serial chain. A serial
chain only reaches trial :math:`j` when trials :math:`0 \dots j-1` were all
rejected, i.e. with the state still :math:`x`, which is the state rank
:math:`j` proposed from. A round therefore is :math:`w+1` serial trials when
rank :math:`w` wins, or :math:`K` trials when none accepts. Step counts,
acceptance statistics, outfile/trajectory writes and averages are replayed
trial by trial, so a walker's outfile reads like that of a serial run. With
one rank per walker the run is bit-for-bit identical to
``GrandCanonicalEnsemble.run`` (tested).

The expected number of serial trials done per round is

.. math::

   \frac{1 - (1 - a)^K}{a}

for acceptance ratio :math:`a`. It is close to :math:`K` while
:math:`K \lesssim 1/a`, which is typical for insertion/deletion with relaxed
trials on a nanoparticle, and saturates at :math:`1/a` beyond that. The
summary file reports the measured ``trials/round``; if it is well below
:math:`K`, give the ranks to more walkers instead.

Usage
-----

.. code-block:: python

   from mcpy.ensembles import GrandCanonicalEnsemble, ParallelGCMC

   def gcmc_factory(walker):
       ...  # build atoms, SphericalCell, moves, calculator
       return GrandCanonicalEnsemble(..., mu={'O': mu_o},
                                     outfile=f'walker{walker}.out',
                                     traj_file=f'walker{walker}.xyz')

   result = ParallelGCMC(gcmc_factory, steps=20000, n_walkers=8,
                         equilibration_steps=5000, sample_interval=10,
                         seed=2024).run()

and ``srun -n 192 python script.py``. The factory runs on every rank; all
ranks of a walker must build the same system. Only the first rank of each
walker writes the outfile and trajectory. World rank 0 writes
``parallel_gcmc_summary.txt`` and ``global_minimum.xyz`` and receives the
results dictionary from ``run()`` (other ranks get ``None``).

A full script is ``examples/05_parallel_gcmc_nanoparticle.py`` and a SLURM
job for 192-core nodes, with rank/thread layouts, is
``examples/slurm_parallel_gcmc.sh``. Always set ``OMP_NUM_THREADS`` to the
cores per rank, otherwise every rank's PyTorch spawns a thread per core.

Limits
------

- Every rank holds its own copy of the potential. For MACE on CPU that is a
  few hundred MB per rank.
- The random streams of the ensemble, its move selector, moves and cells are
  replaced per rank (:func:`~mcpy.ensembles.parallel_gcmc.reseed_ensemble`);
  seeds passed to them in the factory are ignored.
- Only the configuration (``atoms.arrays``, constraints, energy, cell free
  volumes) is shared after an accepted move. A custom move or calculator that
  keeps configuration-dependent state elsewhere breaks the scheme; the
  driver checks at the end that all ranks of a walker agree.
- For several chemical potentials at once (a phase diagram), give each
  walker its own ``mu`` in the factory; results are pooled per
  ``(T, mu)``. To *exchange* configurations between them use
  :class:`ReplicaExchange` instead.
