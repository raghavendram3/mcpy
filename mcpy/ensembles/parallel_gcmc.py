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
        ``gcmc_factory(walker=int) -> GrandCanonicalEnsemble``. Called once
        per rank; every rank of a walker must build the same system. Put
        ``walker`` in the outfile/trajectory names. Only the first rank of
        each walker (its *leader*) writes them; the driver disables the
        output files on the other ranks.
    steps : int
        GCMC steps per walker (one step = ``move_selector.n_moves`` trials,
        as in :meth:`GrandCanonicalEnsemble.run`).
    n_walkers : int
        Number of independent chains. ``comm.size`` must be a multiple of
        it; ``comm.size // n_walkers`` ranks evaluate trial moves of each
        walker in parallel.
    equilibration_steps : int
        Steps discarded before averages are accumulated.
    sample_interval : int
        Take one sample for the averages every this many steps.
    seed : int
        Root seed. All random streams of all ranks are derived from it, so
        a run is reproducible for a fixed rank layout.
    comm : MPI communicator, optional
        Defaults to ``MPI.COMM_WORLD``.
    summary_file : str or None
        Written by world rank 0: per-walker and pooled averages.
    global_minimum_file : str or None
        Lowest grand potential configuration over all walkers.
    """

    def __init__(self,
                 gcmc_factory,
                 steps,
                 n_walkers=1,
                 equilibration_steps=0,
                 sample_interval=1,
                 seed=12345,
                 comm=None,
                 summary_file='parallel_gcmc_summary.txt',
                 global_minimum_file='global_minimum.xyz'):
        if comm is None:
            if MPI is None:
                raise ImportError(
                    "mpi4py is required for ParallelGCMC. Install it with "
                    "`pip install mpi4py` against the cluster's MPI library.")
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        if n_walkers < 1 or self.size % n_walkers:
            raise ValueError(
                f"{self.size} MPI ranks cannot be split into {n_walkers} "
                "equal walker groups; use a rank count that is a multiple "
                "of n_walkers.")
        if sample_interval < 1:
            raise ValueError("sample_interval must be >= 1")

        self.n_walkers = n_walkers
        self.group_size = self.size // n_walkers
        self.walker = self.rank // self.group_size
        self.group = comm.Split(color=self.walker, key=self.rank)
        self.group_rank = self.group.Get_rank()
        self.is_leader = self.group_rank == 0

        self.steps = int(steps)
        self.equilibration_steps = int(equilibration_steps)
        self.sample_interval = int(sample_interval)
        self.summary_file = summary_file
        self.global_minimum_file = global_minimum_file
        self.logger = logging.getLogger(f"{__name__}.walker{self.walker}")

        self.gcmc = gcmc_factory(walker=self.walker)
        if getattr(self.gcmc, 'move_selector', None) is None:
            raise TypeError("gcmc_factory must return a GrandCanonicalEnsemble")
        if not self.is_leader:
            self.gcmc._outfile = None
            self.gcmc._traj_file = None
            self.gcmc._minima_file = None

        walker_seq = np.random.SeedSequence(seed).spawn(n_walkers)[self.walker]
        reseed_ensemble(self.gcmc,
                        walker_seq.spawn(self.group_size)[self.group_rank])

        self._cells = _unique_cells(self.gcmc)
        self._trials = 0
        self._rounds = 0
        self._samples = defaultdict(list)
        self._wall = 0.0

    # -- state synchronisation inside a walker group ------------------------

    def _payload(self):
        atoms = self.gcmc.atoms
        return ({k: v.copy() for k, v in atoms.arrays.items()},
                [c.copy() for c in atoms.constraints],
                self.gcmc.E_old,
                [getattr(c, 'volume', None) for c in self._cells])

    def _apply(self, payload):
        arrays, constraints, energy, volumes = payload
        gcmc = self.gcmc
        gcmc.atoms.arrays = arrays
        gcmc.atoms.set_constraint(constraints)
        gcmc.E_old = energy
        gcmc.n_atoms = len(gcmc.atoms)
        for cell, volume in zip(self._cells, volumes):
            if volume is not None:
                cell.volume = volume

    def _sync_from(self, root):
        payload = self._payload() if self.group_rank == root else None
        payload = self.group.bcast(payload, root=root)
        if self.group_rank != root:
            self._apply(payload)

    # -- one speculative round -----------------------------------------------

    def _propose_and_test(self):
        """Propose one trial from the shared state, evaluate and test it.
        Leaves the state unchanged. Returns ``(record, pending)`` where
        ``record = (move_index, failed, accepted)`` and ``pending`` holds the
        relaxed configuration and energy if it was accepted."""
        gcmc = self.gcmc
        ms = gcmc.move_selector
        proposal = gcmc._propose()
        idx = ms.to_use
        # MoveSelector counted this attempt locally; the group-wide tally
        # in _replay counts it again, only if the serial chain would reach it.
        ms.move_counter[idx] -= 1
        ms.move_counter_total[idx] -= 1
        if proposal is None:
            ms.move_failed_counter[idx] -= 1
            ms.move_failed_counter_total[idx] -= 1
            return (idx, True, False), None
        snapshot, meta = proposal
        E_new = gcmc.compute_energy(gcmc.atoms)
        accepted = bool(gcmc._is_accepted(E_new, meta))
        pending = None
        if accepted:
            pending = ({k: v.copy() for k, v in gcmc.atoms.arrays.items()},
                       [c.copy() for c in gcmc.atoms.constraints], E_new)
        gcmc._restore(snapshot)
        return (idx, False, accepted), pending

    def _round(self, n_active):
        record, pending = (None, None)
        if self.group_rank < n_active:
            record, pending = self._propose_and_test()
        records = self.group.allgather(record)[:n_active]
        winner = next((j for j, r in enumerate(records) if r[2]), None)
        self._rounds += 1
        self._replay(records, winner, pending)

    def _replay(self, records, winner, pending):
        """Apply the round as the serial chain would have seen it."""
        gcmc = self.gcmc
        ms = gcmc.move_selector
        covered = len(records) if winner is None else winner + 1
        for j in range(covered):
            idx, failed, accepted = records[j]
            ms.move_counter[idx] += 1
            ms.move_counter_total[idx] += 1
            if failed:
                ms.move_failed_counter[idx] += 1
                ms.move_failed_counter_total[idx] += 1
            if j == winner:
                ms.move_acceptance[idx] += 1
                ms.move_acceptance_total[idx] += 1
                if self.group_rank == winner:
                    arrays, constraints, E_new = pending
                    gcmc.atoms.arrays = arrays
                    gcmc.atoms.set_constraint(constraints)
                    gcmc._commit(E_new)
                self._sync_from(winner)
                gcmc._record_minimum(gcmc.atoms, gcmc.E_old)
            self._trials += 1
            if self._trials % ms.n_moves == 0:
                self._finish_step()

    def _finish_step(self):
        """Per-step bookkeeping of ``GrandCanonicalEnsemble._run``."""
        gcmc = self.gcmc
        gcmc._step += 1
        if gcmc._step % gcmc._outfile_write_interval == 0:
            gcmc.write_outfile()
            if self.is_leader:
                self.logger.info(
                    "walker=%d step=%d N=%d E=%.6f trials/round=%.2f",
                    self.walker, gcmc._step, gcmc.n_atoms, gcmc.E_old,
                    self._trials / max(self._rounds, 1))
        if gcmc._step % gcmc._trajectory_write_interval == 0:
            gcmc.write_coordinates(gcmc.atoms, gcmc.E_old)
        past_eq = gcmc._step - self.equilibration_steps
        if self.is_leader and past_eq > 0 and past_eq % self.sample_interval == 0:
            self._sample()

    def _sample(self):
        gcmc = self.gcmc
        for specie in gcmc._mu:
            self._samples[f"N_{specie}"].append(
                gcmc._species_count(gcmc.atoms, specie))
        self._samples["N_total"].append(len(gcmc.atoms))
        self._samples["E"].append(gcmc.E_old)
        self._samples["Omega"].append(
            gcmc._minimum_score(gcmc.atoms, gcmc.E_old))

    # -- driver ---------------------------------------------------------------

    def run(self):
        gcmc = self.gcmc
        total_trials = self.steps * gcmc.move_selector.n_moves
        try:
            self._sync_from(0)
            gcmc.initialize_run()
            if self.rank == 0:
                self._log_banner()
            t0 = time.perf_counter()
            try:
                while self._trials < total_trials:
                    self._round(min(self.group_size,
                                    total_trials - self._trials))
            finally:
                self._wall = time.perf_counter() - t0
                gcmc.finalize_run()
            self._check_in_sync()
        except Exception:
            self.logger.exception("rank %d failed; aborting all ranks",
                                  self.rank)
            if self.size > 1:
                self.comm.Abort(1)
            raise
        return self._gather_results()

    def _check_in_sync(self):
        """All ranks of a walker must end on the same configuration."""
        mine = (len(self.gcmc.atoms), self.gcmc.E_old, self.gcmc._step)
        if len(set(self.group.allgather(mine))) != 1:
            raise RuntimeError(
                f"walker {self.walker}: ranks diverged; a move or calculator "
                "keeps configuration-dependent state outside atoms.arrays")

    def _log_banner(self):
        g = self.gcmc
        lines = [
            "+-------------------------------------------------+",
            "| Parallel Grand Canonical Monte Carlo            |",
            "+-------------------------------------------------+",
            f"MPI ranks: {self.size}  walkers: {self.n_walkers}  "
            f"ranks per walker: {self.group_size}",
            f"T (K): {g._temperature}  mu (eV): {g._mu}",
            f"GCMC steps per walker: {self.steps} "
            f"({g.move_selector.n_moves} trials/step), "
            f"equilibration: {self.equilibration_steps}, "
            f"sample interval: {self.sample_interval}",
        ]
        for line in lines:
            self.logger.info(line)

    def _gather_results(self):
        gcmc = self.gcmc
        ms = gcmc.move_selector
        local = None
        if self.is_leader:
            local = {
                "walker": self.walker,
                "mu": dict(gcmc._mu),
                "temperature": gcmc._temperature,
                "steps": gcmc._step,
                "trials": self._trials,
                "rounds": self._rounds,
                "wall_seconds": self._wall,
                "move_names": list(ms.move_list_names),
                "acceptance": ms.total_ratios(),
                "means": {k: float(np.mean(v)) for k, v in self._samples.items()},
                "stds": {k: float(np.std(v)) for k, v in self._samples.items()},
                "n_samples": len(self._samples.get("E", [])),
                "final_N": len(gcmc.atoms),
                "final_E": gcmc.E_old,
                "best": (None if gcmc._best_atoms is None else
                         (gcmc._best_score, gcmc._best_energy,
                          gcmc._best_atoms)),
            }
        gathered = self.comm.gather(local, root=0)
        if self.rank != 0:
            return None
        walkers = [w for w in gathered if w is not None]
        walkers.sort(key=lambda w: w["walker"])
        self._write_global_minimum(walkers)
        pooled = self._pool(walkers)
        self._write_summary(walkers, pooled)
        for w in walkers:
            w.pop("best")
        return {"walkers": walkers, "pooled": pooled}

    @staticmethod
    def _pool(walkers):
        """Mean over walkers sharing (T, mu) and the standard error of that
        mean from the spread of the independent walker means."""
        groups = defaultdict(list)
        for w in walkers:
            key = (w["temperature"], tuple(sorted(w["mu"].items())))
            groups[key].append(w)
        pooled = []
        for (temperature, mu), members in groups.items():
            members = [m for m in members if m["n_samples"] > 0]
            if not members:
                continue
            stats = {}
            for name in members[0]["means"]:
                values = np.array([m["means"][name] for m in members])
                sem = (float(values.std(ddof=1) / np.sqrt(len(values)))
                       if len(values) > 1 else float('nan'))
                stats[name] = (float(values.mean()), sem)
            pooled.append({"temperature": temperature, "mu": dict(mu),
                           "n_walkers": len(members), "stats": stats})
        return pooled

    def _write_global_minimum(self, walkers):
        if self.global_minimum_file is None:
            return
        best = [(w["walker"], w["best"]) for w in walkers if w["best"]]
        if not best:
            return
        walker, (score, energy, atoms) = min(best, key=lambda kv: kv[1][0])
        write_global_minimum(self.global_minimum_file, atoms, energy, score,
                             walker, self.logger)

    def _write_summary(self, walkers, pooled):
        lines = ["Parallel GCMC summary",
                 f"MPI ranks: {self.size}, walkers: {self.n_walkers}, "
                 f"ranks per walker: {self.group_size}", ""]
        for w in walkers:
            acc = ", ".join(
                f"{n}={r * 100:.1f}%" if not np.isnan(r) else f"{n}=N/A"
                for n, r in zip(w["move_names"], w["acceptance"]))
            means = ", ".join(
                f"<{k}>={w['means'][k]:.4f} (sd {w['stds'][k]:.4f})"
                for k in w["means"])
            lines += [
                f"walker {w['walker']}: T={w['temperature']} mu={w['mu']}",
                f"  steps={w['steps']} trials={w['trials']} "
                f"rounds={w['rounds']} trials/round="
                f"{w['trials'] / max(w['rounds'], 1):.2f} "
                f"wall={w['wall_seconds']:.1f}s",
                f"  acceptance: {acc}",
                f"  samples={w['n_samples']} {means}",
                f"  final N={w['final_N']} E={w['final_E']:.6f}",
            ]
        lines.append("")
        for p in pooled:
            lines.append(f"pooled over {p['n_walkers']} walkers: "
                         f"T={p['temperature']} mu={p['mu']}")
            for name, (mean, sem) in p["stats"].items():
                lines.append(f"  <{name}> = {mean:.6f} +/- {sem:.6f}")
        text = "\n".join(lines) + "\n"
        for line in lines:
            self.logger.info(line)
        if self.summary_file is not None:
            try:
                with open(self.summary_file, "w") as fh:
                    fh.write(text)
            except OSError:
                self.logger.exception("Error writing %s", self.summary_file)
