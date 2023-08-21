# TODO: Add methods for constrained rl which are not "classic" to any rl problem
import numpy as np

from . import entropy_regularized as soft
from .entropy_regularized import pi_values

S, A, S_next, D, T, R, Psi = 0, 1, 2, 3, 4, 5, 6


def cost(env: Gridworld, policy: np.ndarray, xi: np.ndarray, gamma: float):
    mu = soft.policy2occ(policy, env.nu_0, env.P, gamma)
    return np.sum(env.r * mu) - constraints_cost(env, xi, mu)


def constraints_cost(env: Gridworld, xi: np.ndarray, mu: np.ndarray):
    return xi.T @ (env.b_values - np.einsum("nmk,nm->k", env.Psi, mu))


def model_based_xi_update(
    env: Gridworld,
    xi: np.ndarray,
    policy: np.ndarray,
    c_values: np.ndarray,
    gamma: float,
    cfg: DiscreteConfig.safeRlConfig,
) -> np.ndarray:
    """Perform gradient update for Xi.

    Args:
        MDP (GridworldMDP): Markov Decision Process
        xi (np.ndarray): Dual variable of the constrained problem, shape (n_constraints,).
        policy (np.ndarray): Policy used to perform the update, shape (n,m).
        c_values (np.ndarray): Current constraints value, shape (n,).
        eta_xi (float): Step size.
        max_iter (Optional[int], optional): Maximum number of iteration. Defaults to 50.
        tol (Optional[float], optional): Stopping criterion. Defaults to 1e-9.
        threshold (Optional[float], optional): Threshold for projection. Defaults to float("inf").

    Returns:
        np.ndarray: Updated xi.
    """

    new_c_values = soft.approx_value_eval(
        env.P, c_values, env.Psi, policy, beta=0, gamma=gamma, max_iters=10000, tol=cfg.tol
    )
    grad_xi = env.b - np.einsum("i,ij->j", env.nu_0, new_c_values)
    return xi - cfg.eta_xi * grad_xi, new_c_values, grad_xi


def model_based_xi_update_momentum(
    env: Gridworld,
    xi: np.ndarray,
    prev_grad,
    policy: np.ndarray,
    c_values: np.ndarray,
    gamma: float,
    cfg: DiscreteConfig.safeRlConfig,
) -> np.ndarray:
    """Perform gradient update for Xi.

    Args:
        MDP (GridworldMDP): Markov Decision Process
        xi (np.ndarray): Dual variable of the constrained problem, shape (n_constraints,).
        policy (np.ndarray): Policy used to perform the update, shape (n,m).
        c_values (np.ndarray): Current constraints value, shape (n,).
        eta_xi (float): Step size.
        max_iter (Optional[int], optional): Maximum number of iteration. Defaults to 50.
        tol (Optional[float], optional): Stopping criterion. Defaults to 1e-9.
        threshold (Optional[float], optional): Threshold for projection. Defaults to float("inf").

    Returns:
        np.ndarray: Updated xi.
    """
    momentum = 0.8
    new_c_values = soft.approx_value_eval(
        env.P, c_values, env.Psi, policy, beta=0, gamma=gamma, max_iters=cfg.max_iter, tol=cfg.tol
    )
    grad_xi = env.b - np.einsum("i,ij->j", env.nu_0, new_c_values)
    grad = momentum * prev_grad + (1 - momentum) * cfg.eta_xi * grad_xi
    return np.clip(xi - grad, 0.0, cfg.xi_threshold), new_c_values, grad


def model_free_qpsi_xi_update(
    replay_buffer: Buffer,
    xi: np.ndarray,
    b: np.ndarray,
    q_psi: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    cfg: DiscreteConfig.safeRlConfig,
) -> np.ndarray:
    """Perform gradient update for Xi.

    Args:
        MDP (GridworldMDP): Markov Decision Process
        xi (np.ndarray): Dual variable of the constrained problem, shape (n_constraints,).
        policy (np.ndarray): Policy used to perform the update, shape (n,m).
        c_values (np.ndarray): Current constraints value, shape (n,).
        eta_xi (float): Step size.
        max_iter (Optional[int], optional): Maximum number of iteration. Defaults to 50.
        tol (Optional[float], optional): Stopping criterion. Defaults to 1e-9.
        xi_threshold (Optional[float], optional): Threshold for projection. Defaults to float("inf").

    Returns:
        np.ndarray: Updated xi.
    """
    grad_xi = np.zeros_like(xi)

    s, a, s_next, d, _, _ = replay_buffer.extract_datas()

    v_psi = pi_values(q_psi, policy, beta=0)
    targets = (1 - d) * gamma * v_psi[s_next]
    psi = q_psi[(s, a)] - targets

    # Compute b - E_D[Psi]
    grad_xi = b - np.sum(psi, axis=(0, 1))
    xi = xi - cfg.eta_xi * grad_xi
    return np.clip(xi, 0.0, cfg.xi_threshold), grad_xi


def model_free_xi_update(
    replay_buffer: Buffer,
    xi: np.ndarray,
    b: np.ndarray,
    Psi: np.ndarray,
    gamma: float,
    cfg: DiscreteConfig.safeRlConfig,
):
    s, a, _, _, _, _ = replay_buffer.extract_datas()
    gammas = replay_buffer.get_gamma_vec(gamma)

    grad_xi = b - np.sum(Psi[(s, a)] * gammas[:, None], axis=0) / replay_buffer.n_traj

    return np.clip(xi - cfg.eta_xi * grad_xi, 0, float("inf"))
