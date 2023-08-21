import math

import numpy as np

from . import entropy_regularized as soft
from .utilities import Project

S, A, S_next, D, T, R, Psi = 0, 1, 2, 3, 4, 5, 6


def model_based_w_update(
    env: Gridworld,
    Phi: np.ndarray,
    w: np.ndarray,
    policy: np.ndarray,
    sigma_E: np.ndarray,
    gamma: float,
    cfg: DiscreteConfig.inverseRlConfig,
):
    """
    Perform gradient update for w.

    :param env: Environment.
    :param Phi: Reward feature matrix.
    :param w: Reward feature vector
    :param policy: Current policy, shape (n,m).
    :param sigma_E: Estimate of the expert occupancy measure.
    :param gamma: Discouting factor.
    :param cfg: Config of the agent.
    :return: Updated feature vector, shape (n,m).
    """

    grad_w = np.einsum("nmd,nm->d", Phi, soft.policy2occ(policy, env.nu_0, env.P, gamma)) - sigma_E
    w_int = w - cfg.eta_w * grad_w

    if cfg.proj_type == "l1_ball":
        return Project.l1_ball(w_int, cfg.ball_radius), grad_w
    elif cfg.proj_type == "l2_ball":
        return Project.l2_ball(w_int, cfg.ball_radius), grad_w
    else:
        return np.clip(w_int, 0, 1), grad_w


def model_free_w_update(
    Phi: np.ndarray,
    w: np.ndarray,
    sigma_E: np.ndarray,
    buffer: Buffer,
    gamma: float,
    cfg: DiscreteConfig.inverseRlConfig,
):
    """
    Perform gradient update for w.

    :param Phi: Reward feature matrix.
    :param w: Reward feature vector
    :param policy: Current policy, shape (n,m).
    :param sigma_E: Estimate of the expert occupancy measure.
    :param buffer: Replay buffer containing transitions.
    :param gamma: Discouting factor.
    :param cfg: Config of the agent.
    :return: Updated feature vector, shape (n,m).
    """
    s, a, _, _, _, _ = buffer.extract_datas()
    gammas = buffer.get_gamma_vec(gamma)

    grad_w = np.sum(Phi[(s, a)] * gammas[:, None], axis=0) / buffer.n_traj - sigma_E

    w = w - cfg.eta_w * grad_w
    if cfg.proj_type == "l1_ball":
        return Project.l1_ball(w, cfg.ball_radius)
    elif cfg.proj_type == "l2_ball":
        return Project.l2_ball(w, cfg.ball_radius)
    else:
        return np.clip(w, 0, 1)
