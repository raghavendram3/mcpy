import logging
import time
from typing import Optional, List, Dict

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .base_ensemble import BaseEnsemble
from ..utils.random_number_generator import RandomNumberGenerator
from ..utils.set_unit_constant import SetUnits
from ..moves.move_selector import MoveSelector
from ..moves.molecule_utils import find_molecules
from ..cell import Cell

logger = logging.getLogger(__name__)

_RATIO_COL_PER_MOVE = 8  # width per "100.0%, " entry, generous


class GrandCanonicalEnsemble(BaseEnsemble):
    def __init__(self,
                 atoms: Atoms,
                 cells: List[Cell],
                 units_type: str,
                 calculator: Calculator,
                 mu: Dict[str, float],
                 species: List[str],
                 temperature: float,
                 move_selector: MoveSelector,
                 molecules: Optional[Dict[str, Atoms]] = None,
                 random_seed: Optional[int] = None,
                 traj_file: Optional[str] = 'trajectory.xyz',
                 traj_mode: str = 'w',
                 trajectory_write_interval: Optional[int] = 1,
                 outfile: Optional[str] = 'outfile.out',
                 outfile_mode: str = 'w',
                 outfile_write_interval: Optional[int] = 1,
                 minima_file: Optional[str] = None,
                 minima_mode: str = 'a') -> None:

        super().__init__(atoms=atoms,
                         cells=cells,
                         calculator=calculator,
                         random_seed=random_seed,
                         traj_file=traj_file,
                         traj_mode=traj_mode,
                         trajectory_write_interval=trajectory_write_interval,
                         outfile=outfile,
                         outfile_mode=outfile_mode,
                         outfile_write_interval=outfile_write_interval,
                         minima_file=minima_file,
                         minima_mode=minima_mode)

        self.E_old = self.compute_energy(self.atoms)

        self.units = SetUnits(units_type,
                              temperature=temperature,
                              species=species,
                              molecules=molecules)

        self.initial_atoms = len(self.atoms)
        self.n_atoms = len(self.atoms)
        self._temperature = temperature
        self._mu = mu

        self.move_selector = move_selector
        self.calculate_cells_volume(self.atoms)
        self._step = 0
        self._last_logged_step = None
        self.exchange_attempts = 0
        self.exchange_successes = 0
        self.rng_acceptance = RandomNumberGenerator(seed=self._random_seed + 2)
        # PBC is fixed for the run; precompute so we can skip ``wrap()`` for
        # cluster systems and on rejected trials.
        self._wrap_on_accept = bool(np.any(np.asarray(self.atoms.pbc)))

        # Width of the acceptance-ratios column in the outfile, sized to the
        # number of moves so >4 moves don't shred the alignment.
        self._ratio_col_width = max(20, _RATIO_COL_PER_MOVE * len(move_selector.move_list))
        self._table_width = 10 + 1 + 10 + 1 + 15 + 1 + self._ratio_col_width

    def get_state(self) -> Dict[str, any]:
        return {
            "atoms": self.atoms,
            "n_atoms": self.n_atoms,
            "energy": self.E_old,
            "mu": self._mu,
            "temperature": self._temperature,
            "beta": self.units.beta,
            "lambda_dbs": self.units.lambda_dbs,
            "step": self._step,
            "exchange_attempts": self.exchange_attempts,
            "exchange_successes": self.exchange_successes,
        }

    def set_state(self, state: Dict[str, any]) -> None:
        self.atoms = state["atoms"]
        self.E_old = state["energy"]
        self.n_atoms = state["n_atoms"]
