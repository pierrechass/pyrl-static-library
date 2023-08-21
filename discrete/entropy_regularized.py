"""
Defines all the function for value, q_value and policy evaluation and calculation for RL in the entropy regularized case.

"""
import numpy as np
from einops import rearrange
from scipy.special import logsumexp, softmax
from scipy.stats import entropy

# TODO: Add methods for entropy regularized rl which are classic, e.g. normal q_learning


# pylint:disable=W0621
def occ2policy(occ: np.ndarray) -> np.ndarray:
    z = np.where(np.sum(occ, axis=1, keepdims=True) > 1e-10, occ, 1.0)
    return z / np.sum(z, axis=1, keepdims=True)


def state_occupancy(policy: np.ndarray, nu_0: np.ndarray, P: np.ndarray, gamma: float) -> np.ndarray:
    """
    Evaluate state occupancy measure corresponding to policy.

    :param: Policy, shape (n,m).
    :param: nu_0, initial state distribution, shape (n,).
    :return: State occupancy measure, shape (n,).
    """

    P_policy_T = np.einsum("sak, ka->ks", P, policy)
    state_occ = np.linalg.solve(np.eye(P.shape[0]) - gamma * P_policy_T, nu_0)
    return state_occ / np.sum(state_occ)


def policy2occ(policy: np.ndarray, nu_0: np.ndarray, P: np.ndarray, gamma: float) -> np.ndarray:
    """Evaluate state-action occupancy measure corresponding to policy.

    Args:
        policy (np.ndarray): shape (n,m)

    Returns:
        occupancy(np.ndarray): shape (n,m)
    """

    state_occ = state_occupancy(policy, nu_0, P, gamma)
    return np.einsum("i, ij -> ij", state_occ / np.sum(state_occ), policy)


def value_it_step(P: np.ndarray, values: np.ndarray, reward: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Calculate single soft value iteration step.

    :param values: Current value function, shape (n,).
    :param beta: Regularization parameter > 0.
    :return: (T values): Bellman optimality operator applied to values, shape (n,).
    """

    r = reward
    return beta * logsumexp((r + gamma * np.einsum("jki,i->jk", P, values)) / beta, axis=1)


def q_it_step(P: np.ndarray, q_values: np.ndarray, reward: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Calculate single soft Q-value iteration step.

    :param q_values: Current value function, shape (n, m).
    :param beta: Regularization parameter > 0.
    :return: (T q_values): Bellman optimality operator applied to q_values, shape (n, m).
    """

    r = reward
    return r + gamma * np.einsum("jki,i->jk", P, opt_values(q_values, beta))


def opt_values(q_values: np.ndarray, beta: float) -> np.ndarray:
    """Compute optimal values corresponding to q.

    Args:
        q_values (np.ndarray): shape [n,m(, k)], Q values where the 3rd dim is optionnal (e.g. Qpsi for constraint violation.)
        beta (float): Temperature

    Returns:
        np.ndarray: values shape [n(, k)]
    """
    return beta * logsumexp(q_values / beta, axis=1)


def pi_values(q_values: np.ndarray, policy: np.ndarray, beta: float) -> np.ndarray:
    """Compute values corresponding to q and policy.

    Args:
        q_values (np.ndarray): shape [n,m(, k)], Q values where the 3rd dim is optionnal (e.g. Qpsi for constraint violation.)
        policy (np.ndarray): shape [n,m]
        beta (float): Temperature

    Returns:
        np.ndarray: values shape [n(, k)]
    """
    if q_values.ndim == 3:
        return np.einsum("nm...,nm->n...", q_values, policy) + beta * entropy(policy, axis=1)[:, None]
    return np.einsum("nm,nm->n", q_values, policy) + beta * entropy(policy, axis=1)


def npg_step(
    q_values: np.ndarray,
    policy: np.ndarray,
    beta: float,
    gamma: float,
    eta_pi: float,
):
    """Entropy regularized NPG for softmax policy parametrization. For eta = (1-gamma) / beta
    this reduces to soft-policy iteration.

    Args:
        P (np.ndarray): Transition dynamics, shape: (n,m,n).
        reward (np.ndarray): Rewards of the MDP, shape: (n,m).
        values (np.ndarray): Values corresponding to policy, shape (n,).
        policy (np.ndarray): Current policy, shape (n,m).
        beta (float): Entropy regularization parameter.
        gamma (float): Discounting factor.
        eta_p (float): Step-size for the update.

    Returns:
        np.ndarray: Updated policy, shape (n,m).
    """
    # For stability shift the qs
    q_shifted = q_values - np.amax(q_values, axis=1, keepdims=True)
    unnormalized_policy = policy ** (1 - eta_pi * beta / (1 - gamma)) * np.nan_to_num(
        np.exp(eta_pi * q_shifted / (1 - gamma))
    )
    return unnormalized_policy / np.sum(unnormalized_policy, axis=1)[:, None]


def q_learning_step(
    replay_buffer: Buffer,
    policy: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    gamma: float,
    beta: float,
    eta_q: float,
):
    """Perform soft Q learning by minimizing the Bellman error :

        J = (1/2)*(Q(s,a) - (r(s,a) + gamma*V(s')))^2

    Args:
        replay_buffer (Buffer): replay buffer of shape [batch_size, [s,a,s_next,d, r, Psi_1, Psi_2,...,Psi_n_constraints]]
        policy (ndarray): shape [n,m]
        q (ndarray): shape [n,m(,k)], Q values where third dimension is optionnal (for multiple Q learning).
        r (ndarray): shape [n,m(,k)], reward associated to the Q.
        eta_q (float): learning rate
        beta (float): regularization parameter
        gamma (float): discount factor

    Returns:
        np.ndarray: Updated Q functions
    """
    grad = np.zeros_like(q)
    s, a, s_next, d, _, _ = replay_buffer.extract_datas()
    next_values = pi_values(q, policy, beta=beta)[s_next]
    if r.ndim == 3:
        targets = r[(s, a)] + np.einsum("l,l...->l...", (1 - d), gamma * next_values)
    else:
        targets = r[(s, a)] + (1 - d) * gamma * next_values
    np.add.at(grad, (s, a), q[(s, a)] - targets)
    return q - eta_q * grad / replay_buffer.n_traj


def value_eval_step(
    P: np.ndarray,
    values: np.ndarray,
    reward: np.ndarray,
    policy: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    Calculate single soft value evaluation step. Handle multiple value iteration.

    :param values: Current value function, shape (n(,k)).
    :param policy: Policy to evaluate, shape (n,m).
    :param beta: Regularization parameter > 0.
    :return: (T^pi values): Soft Bellman expectation operator applied to values, shape (n(,k)).
    """
    r = reward
    H = entropy(policy, axis=1)
    V_next = np.einsum("nmk,k... -> nm...", P, values)
    if V_next.ndim == 3:
        return np.einsum("nm,nm...->n...", policy, (r + gamma * V_next)) + beta * H[:, None]

    return np.einsum("nm,nm->n", policy, (r + gamma * V_next)) + beta * H


def q_eval_step(
    P: np.ndarray,
    q_values: np.ndarray,
    reward: np.ndarray,
    policy: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    Calculate single soft Q-value evaluation step. Handle mutliple Q.

    :param q_values: Current Q-value function, shape (n, m(,k)).
    :param policy: Policy to evaluate, shape (n,m).
    :param beta: Regularization parameter > 0.
    :return: (T^pi q_values): Soft Bellman expectation operator applied to q_values, shape (n, m(,k)).
    """
    r = reward

    values = pi_values(q_values, policy, beta)

    if q_values.ndim == 3:
        return r + gamma * np.einsum("nmk,k...-> nm...", P, values)
    return r + gamma * np.einsum("nmk,k -> nm", P, values)


def approx_value_eval(
    P: np.ndarray,
    values: np.ndarray,
    reward: np.ndarray,
    policy: np.ndarray,
    beta: float,
    gamma: float,
    max_iters: float = 50,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Calculate optimal soft value function via soft value iteration.

    :param values: Initial values.
    :param beta: Regularization parameter > 0.
    :param max_iters: Max number of iterations.
    :param tol: Error tolerance for stopping.
    :param log_steps: Pause time for logging.
    :return: values: Optimal values.
    """

    it = 0
    error = float("inf")
    while it < max_iters and error > tol:
        it += 1
        new_values = value_eval_step(P, values, reward, policy, beta, gamma)
        error = np.max(abs(new_values - values))
        values = new_values
    return values


def approx_q_eval(
    P: np.ndarray,
    q_values: np.ndarray,
    reward: np.ndarray,
    policy: np.ndarray,
    beta: float,
    gamma: float,
    max_iters: float = 50,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Calculate optimal soft Q-value function via soft-Q iteration.

    :param q_values: Initial Q values.
    :param beta: Regularization parameter > 0.
    :param max_iters: Max number of iterations.
    :param tol: Error tolerance for stopping.
    :param log_steps: Pause time for logging.
    :return: q_values: Optimal Q values.
    """

    it = 0
    error = float("inf")
    while it < max_iters and error > tol:
        it += 1
        new_q_values = q_eval_step(P, q_values, reward, policy, beta, gamma)
        error = np.max(abs(new_q_values - q_values))
        q_values = new_q_values
    return q_values


def value_eval(P: np.ndarray, policy: np.ndarray, reward: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Evaluate soft value function of policy by solving the linear equations.

    :policy: Policy, shape (n,m).
    :param beta: Regularization parameter > 0.
    :return: values: values, shape (n,).
    """

    r = reward
    P_policy = np.einsum("jki, jk->ji", P, policy)
    r_policy = np.sum(policy * r, axis=1) + beta * entropy(policy, axis=1)
    return np.linalg.solve(np.eye(policy.shape[0]) - gamma * P_policy, r_policy)


def q_eval(P: np.ndarray, policy: np.ndarray, reward: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Evaluate soft Q-value function of policy by solving the linear equations.

    :policy: Policy, shape (n,m).
    :param beta: Regularization parameter > 0.
    :return: q_values: Q-values, shape (n,m).
    """

    r = reward
    P_policy = rearrange(np.einsum("jki, il-> jkil", P, policy), "j k i l -> (j k) (i l)")
    # P_policy = rearrange(np.einsum("jki, il-> jkil", P, policy), "j k i l -> (j k) (i l)")
    r_regularized = rearrange(r + beta * gamma * np.einsum("nmk, k -> nm", P, entropy(policy, axis=1)), "s a -> (s a)")
    return rearrange(
        np.linalg.solve(np.eye(P.shape[0] * P.shape[1]) - gamma * P_policy, r_regularized),
        "(s a) -> s a",
        s=P.shape[0],
        a=P.shape[1],
    )


def greedy_pi_v(P: np.ndarray, values: np.ndarray, reward: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Get soft greedy policy from soft values.

    :param values: values, shape (n,).
    :param beta: Regularization parameter.
    :return: Greedy policy, shape (n,m).
    """

    r = reward
    q_values = r + gamma * np.einsum("jki,i -> jk", P, values)
    return softmax(q_values / beta, axis=1)


def greedy_pi_q(q_values: np.ndarray, beta: float) -> np.ndarray:
    """
    Get soft greedy policy from soft q_values.

    :param q_values: Q-values, shape (n,m).
    :param beta: Regularization parameter.
    :return: Greedy policy, shape (n,m).
    """

    return softmax(q_values / beta, axis=1)


def value_it(
    P: np.ndarray,
    values: np.ndarray,
    reward: np.ndarray,
    beta: float,
    gamma: float,
    max_iters: float = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    Calculate optimal soft value function via soft value iteration.

    :param values: Initial values, shape (n,).
    :param beta: Regularization parameter > 0.
    :param max_iters: Max number of iterations.
    :param tol: Error tolerance for stopping.
    :return: values: Optimal values, shape (n,).
    """

    it = 0
    error = float("inf")
    while it < max_iters and error > tol:
        it += 1
        new_values = value_it_step(P, values, reward, beta, gamma)
        error = np.max(abs(new_values - values))
        values = new_values
    return values


def q_it(
    P: np.ndarray,
    q_values: np.ndarray,
    reward: np.ndarray,
    beta: float,
    gamma: float,
    max_iters: float = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    Calculate optimal soft Q-value function via soft-Q iteration.

    :param q_values: Initial Q values, shape (n,m).
    :param beta: Regularization parameter > 0.
    :param max_iters: Max number of iterations.
    :param tol: Error tolerance for stopping.
    :return: q_values: Optimal Q values, shape (n,m).
    """

    it = 0
    error = float("inf")
    while it < max_iters and error > tol:
        it += 1
        new_q_values = q_it_step(P, q_values, reward, beta, gamma)
        error = np.max(abs(new_q_values - q_values))
        q_values = new_q_values
    return q_values


def policy_it(
    P: np.ndarray,
    values: np.ndarray,
    reward: np.ndarray,
    beta: float,
    gamma: float,
    mode: str = "exact",
    n_eval_steps: int = 10,
    eval_tol: float = 1e-9,
    max_iters: float = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    Calculate optimal soft value function via soft policy iteration.

    :param values: Initial soft values, shape (n,).
    :param beta: Regularization parameter > 0.
    :param mode: String in {'exact', 'approx'} indicating policy evaluation mode.
    :param n_eval_steps: Number of Bellman updates for policy evaluation.
    :param eval_tol: Tolerance for stopping Bellman updates.
    :param max_iters: Max number of iterations.
    :param tol: Error tolerance for stopping.
    :param log_steps: Logging interval.
    :param logging: Whether to print outputs or not.
    :param reward: Optional reward different from r.
    :return: values: Optimal soft values, shape (n,).
    """

    it = 0
    error = float("inf")
    while it < max_iters and error > tol:
        policy = greedy_pi_v(P, values, reward, beta, gamma)
        if mode == "exact":
            new_values = value_eval(P, policy, reward, beta, gamma)
        else:
            new_values = approx_value_eval(
                P, values, reward, policy, beta, gamma, max_iters=n_eval_steps, tol=eval_tol
            )
        error = np.max(abs(new_values - values))
        values = new_values
        it += 1
    return values


def q_policy_it(
    P: np.ndarray,
    q_values: np.ndarray,
    reward: np.ndarray,
    beta: float,
    gamma: float,
    mode: str = "exact",
    n_eval_steps: int = 10,
    eval_tol: float = 1e-9,
    max_iters: float = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    Calculate optimal soft q value function via soft q policy iteration.

    :param q_values: Initial soft q values, shape (n, m).
    :param beta: Regularization parameter > 0.
    :param mode: String in {'exact', 'approx'} indicating policy evaluation mode.
    :param n_eval_steps: Number of Bellman updates for policy evaluation.
    :param eval_tol: Tolerance for stopping Bellman updates.
    :param max_iters: Max number of iterations.
    :param tol: Error tolerance for stopping.
    :param log_steps: Logging interval.
    :param logging: Whether to print outputs or not.
    :param reward: Optional reward different from r.
    :return: q_values: Optimal soft q values, shape (n, m).
    """

    it = 0
    error = float("inf")
    while it < max_iters and error > tol:
        policy = greedy_pi_q(q_values, beta)
        if mode == "exact":
            new_q_values = q_eval(P, policy, reward, beta, gamma)
        else:
            new_q_values = approx_q_eval(
                P, q_values, reward, policy, beta, gamma, max_iters=n_eval_steps, tol=eval_tol
            )
        error = np.max(abs(new_q_values - q_values))
        q_values = new_q_values
        it += 1
    return q_values
