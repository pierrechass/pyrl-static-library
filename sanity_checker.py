from typing import Tuple

import einops
import numpy as np
from scipy.optimize import linprog

from sycabot_cirl.config.discrete_config import Config
from sycabot_cirl.environments.discrete import Gridworld
from sycabot_cirl.methods.discrete.entropy_regularized import occ2policy


class LP:
    @staticmethod
    def solve(env: Gridworld, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the CMDP using LP.

        Args:
            MDP (GridworldMDP): Markov Decision Process.

        Returns:
            policy, occupancy(Tuple[np.ndarray,np.ndarray]): shape(n,m)
        """
        E_T = np.hstack([np.eye(env.observation_space.n) for _ in range(env.action_space.n)])
        P_T = einops.rearrange(env.P.copy(), "s a k -> k (a s)")
        r = einops.rearrange(env.r.copy(), "s a -> (a s)")
        Psi_T = einops.rearrange(env.Psi.copy(), "s a k -> k (a s)")
        nu = env.nu_0.copy()
        b = env.b.copy()
        sol = linprog(
            c=-r / (1 - cfg.gamma),
            A_eq=E_T - cfg.gamma * P_T,
            b_eq=(1 - cfg.gamma) * nu,
            A_ub=Psi_T / (1 - cfg.gamma),
            b_ub=b,
            bounds=(0, None),
        )
        try:
            occ = einops.rearrange(sol.x, "(a s) -> s a", a=env.action_space.n)
        except RuntimeError as e:
            raise RuntimeWarning(f"Sanity check didn't pass : {sol.message}") from e

        return occ2policy(occ)
