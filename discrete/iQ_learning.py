from enum import Enum

import numpy as np
import torch
from scipy.stats import entropy

from . import entropy_regularized as soft

S, A, S_next, D, T, R, Psi = 0, 1, 2, 3, 4, 5, 6


class iQDivMethod(Enum):
    JENSEN_SHANNON = "js"
    HELLINGER = "h"
    CHI2 = "chi2"
    TOTALVAR = "tv"


class iQLossType(Enum):
    VALUE = "value"
    VALUE_EXP = "value_expert"
    V0 = "v0"


def reward(env: Gridworld, q_r: np.ndarray, gamma: float, beta: float):  # Checked OK
    r = np.zeros_like(q_r)
    Vopt = np.einsum("nmk,k->nm", env.P, soft.opt_values(q_r, beta))
    r = q_r - gamma * Vopt
    return r


def q_update_torch_grad(
    expert_buffer: Buffer,
    q_r: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    beta: float,
    cfg: DiscreteConfig.inverseRlConfig,
    policy_buffer: np.ndarray = None,
):
    q_values = torch.tensor(q_r, requires_grad=True)
    pi = torch.tensor(policy, requires_grad=False)
    loss = torch.tensor(0.0, requires_grad=True)
    buffer = Buffer(np.append(expert_buffer, policy_buffer, axis=0))
    is_expert = np.append(
        np.ones(np.array(expert_buffer).shape[0], dtype=bool),
        np.zeros(np.array(policy_buffer).shape[0], dtype=bool),
    )
    # if iQLossType(cfg.loss_type) == iQLossType.VALUE:
    #     buffer = Buffer(np.append(expert_buffer, policy_buffer, axis=0))
    #     is_expert = np.append(
    #         np.ones(np.array(expert_buffer).shape[0], dtype=bool),
    #         np.zeros(np.array(policy_buffer).shape[0], dtype=bool),
    #     )
    # else:
    #     buffer = expert_buffer
    #     is_expert = np.ones(np.array(expert_buffer).shape[0], dtype=bool)
    # print(buffer.n_traj)
    s, a, s_next, d, _, _ = buffer.extract_datas(as_torch=True)
    gammas = torch.tensor(buffer.get_gamma_vec(gamma), requires_grad=False)
    # values = beta * torch.logsumexp(q_values / beta, axis=1)
    values = beta * torch.einsum("nm,nm->n", pi, q_values)

    current_q = q_values[s, a]
    next_v = values[s_next]
    current_v = values[s]

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    y = (1 - d) * gamma * next_v
    if cfg.use_targets:
        with torch.no_grad():
            y = (1 - d) * gamma * next_v

    r = (current_q - y)[is_expert]
    with torch.no_grad():
        grad_phi = get_grad_phi(r, cfg.div_method, cfg.alpha_chi2)

    loss = (grad_phi * r * gammas[is_expert]).sum() / expert_buffer.n_traj
    loss_dict = {"softq_loss": loss.item()}

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if iQLossType(cfg.loss_type) == iQLossType.VALUE:
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = ((current_v - y) * gammas).sum() / buffer.n_traj
    elif iQLossType(cfg.loss_type) == iQLossType.VALUE_EXP:
        # sample using only expert states (works offline)
        # E_(ρE)[V(s) - γV(s')]
        value_loss = ((current_v - y) * gammas)[is_expert].sum() / expert_buffer.n_traj
    elif iQLossType(cfg.loss_type) == iQLossType.V0:
        # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
        # (1-γ)E_(ρ0)[V(s0)]
        value_loss = (1 - gamma) * current_v[is_expert].sum() / expert_buffer.n_traj

    if iQDivMethod(cfg.div_method) == iQDivMethod.CHI2:
        reward = current_q - y
        chi2_loss = 1 / (4 * cfg.alpha_chi2) * ((reward**2) * gammas).sum() / buffer.n_traj
        loss -= chi2_loss
        loss_dict["chi2_loss"] = chi2_loss

    loss -= value_loss
    loss_dict["value_loss"] = value_loss.item()
    loss_dict["total_loss"] = loss.item()
    loss.backward()
    q_r -= cfg.eta_qr * np.array(-q_values.grad)
    return q_r, loss_dict


def q_update_state_only_torch_grad(
    expert_buffer: Buffer,
    q_r: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    beta: float,
    cfg: DiscreteConfig.inverseRlConfig,
    policy_buffer: Buffer = None,
):
    # # TODO: doesn't work, still needs more investigation, but it might be that it just doesn't work theoritically or not as they say it should
    # q_values = torch.tensor(q_r, requires_grad=True)
    # loss = torch.tensor(0.0, requires_grad=True)

    # if iQLossType(cfg.loss_type) == iQLossType.VALUE:
    #     buffer = Buffer(np.append(expert_buffer, policy_buffer, axis=0))
    #     is_expert = np.append(
    #         np.ones(np.array(expert_buffer).shape[0], dtype=bool),
    #         np.zeros(np.array(policy_buffer).shape[0], dtype=bool),
    #     )
    # else:
    #     buffer = expert_buffer
    #     is_expert = np.ones(np.array(expert_buffer).shape[0], dtype=bool)

    # s, a, s_next, d, _, _ = buffer.extract_datas(as_torch=True)
    # gammas = torch.tensor(buffer.get_gamma_vec(gamma), requires_grad=False)
    # values = (
    #     torch.einsum("nm,nm->n", q_values, torch.tensor(policy, requires_grad=False))
    #     + torch.tensor(entropy(policy, axis=1), requires_grad=False) * beta
    # )

    # current_q = q_values[s, a]
    # next_v = values[s_next]
    # current_v = values[s]

    # #  calculate 1st term for IQ loss
    # #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    # y = (1 - d) * gamma * next_v
    # if cfg.use_targets:
    #     with torch.no_grad():
    #         y = (1 - d) * gamma * next_v

    # r = (current_q - y)[is_expert]
    # with torch.no_grad():
    #     grad_phi = get_grad_phi(r, cfg.div_method, cfg.alpha_chi2)

    # loss = (grad_phi * r * gammas[is_expert]).sum() / expert_buffer.n_traj
    # loss_dict = {"softq_loss": loss.item()}

    # # calculate 2nd term for IQ loss, we show different sampling strategies
    # if iQLossType(cfg.loss_type) == iQLossType.VALUE:
    #     # sample using expert and policy states (works online)
    #     # E_(ρ)[V(s) - γV(s')]
    #     value_loss = ((current_v - y) * gammas).sum() / buffer.n_traj
    # elif iQLossType(cfg.loss_type) == iQLossType.VALUE_EXP:
    #     # sample using only expert states (works offline)
    #     # E_(ρE)[V(s) - γV(s')]
    #     value_loss = ((current_v - y) * gammas[is_expert])[is_expert].sum() / expert_buffer.n_traj
    # elif iQLossType(cfg.loss_type) == iQLossType.V0:
    #     # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
    #     # (1-γ)E_(ρ0)[V(s0)]
    #     value_loss = (1 - gamma) * current_v[is_expert].sum() / expert_buffer.n_traj
    # loss -= value_loss

    # print(cfg.div_method)
    # if iQDivMethod(cfg.div_method) == iQDivMethod.CHI2:
    #     print("here1")
    #     reward = (current_q - y) * gammas
    #     chi2_loss = 1 / (4 * cfg.alpha_chi2) * (reward**2).sum() / buffer.n_traj
    #     loss -= chi2_loss

    # loss_dict["value_loss"] = value_loss.item()
    # loss_dict["total_loss"] = loss.item()
    # loss.backward()
    # # print(q_values.grad)
    # q_r -= cfg.eta_qr * np.array(-q_values.grad)

    # # q_values = torch.tensor(q_r, requires_grad=False)
    # # pi = torch.tensor(policy, requires_grad=True)
    # # piloss = (torch.einsum("nm,nm->n", q_values, pi) + _calc_entropy(pi, axis=1) * beta).sum()
    # # piloss.backward()
    # # policy -= 0.001 * np.array(pi.grad)
    # # policy[policy <= 0] = 0
    return None


def q_update(
    replay_buffer: Buffer,
    q_r: np.ndarray,
    gamma: float,
    beta: float,
    cfg: DiscreteConfig.inverseRlConfig,
    policy_buffer: np.ndarray = None,
):  # Might be DEPRECATED since use of torch gradient seems way more efficient, I let it here for now.
    """1-step udpate of Q update in iQ_learning.

    Args:
        replay_buffer (ndarray): replay buffer [batch_size, [s,a,s_next,d,r,psi1,...,psi_Ncons]]
        q_r (np.ndarray): shape [n,m]
        values (np.ndarray): shape [n]
        policy (np.ndarray): shape [n,m]
    """

    values = soft.opt_values(q_r, beta)

    grad_J = np.zeros_like(q_r)
    r = np.zeros_like(q_r)
    grad_V_Q = soft.opt_values(q_r, beta)

    s, a, s_next, d, _, _ = replay_buffer.extract_datas()

    if policy_buffer is not None:
        s_pi = policy_buffer[:, S].astype(int)
        a_pi = policy_buffer[:, A].astype(int)
        s_next_pi = policy_buffer[:, S_next].astype(int)
        d_pi = policy_buffer[:, D].astype(int)

    r[(s, a)] = q_r[(s, a)] - (1 - d) * gamma * values[s_next]

    grad_phi = get_grad_phi(r, cfg.div_method, cfg.alpha_chi2)

    np.add.at(grad_J, (s, a), grad_phi[(s, a)] - grad_V_Q[(s, a)])

    if not cfg.use_targets:
        grad_targets = (1 - d[:, None]) * gamma * grad_V_Q[s_next]
        np.add.at(grad_J, (s_next, slice(None)), grad_phi[s_next] * grad_targets - grad_targets)
        if policy_buffer is not None:
            grad_targets = (1 - d_pi[:, None]) * gamma * grad_V_Q[s_next_pi]
            np.add.at(grad_J, (s_pi, a_pi), grad_V_Q[(s_pi, a_pi)])
            np.add.at(grad_J, (s_next_pi, slice(None)), -grad_targets)

    grad_J /= replay_buffer.shape[0]
    q_r -= cfg.eta_qr * (-grad_J)
    return q_r


def get_grad_phi(r, div_method: iQDivMethod, alpha_chi2=1):
    if iQDivMethod(div_method) == iQDivMethod.JENSEN_SHANNON:
        grad = 1 / (2 * np.exp(r) - 1)
    elif iQDivMethod(div_method) == iQDivMethod.HELLINGER:
        grad = (1 + r) ** (-2)
    # elif iQDivMethod(div_method) == iQDivMethod.CHI2:
    #     grad = 1 - r / (2 * alpha_chi2)
    else:
        grad = np.ones_like(r)
        if isinstance(r, torch.Tensor):
            grad = torch.tensor(grad)
    return grad


def get_loss(
    replay_buffer: Buffer,
    q_r: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    beta: float,
    cfg: DiscreteConfig.inverseRlConfig,
):
    s, a, s_next, d, _, _ = replay_buffer.extract_datas()

    # keep track of value of initial states
    values = soft.opt_values(q_r, beta)
    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    if cfg.use_targets:
        values = soft.opt_values(q_r, beta)
    else:
        values = soft.pi_values(q_r, policy, beta)

    r = q_r[(s, a)] - (1 - d) * gamma * values[s_next]
    grad_phi = get_grad_phi(r, cfg.div_method, cfg.alpha_chi2)

    loss = -np.mean(grad_phi * r)
    loss_dict = {f"{cfg.div_method}_loss": loss}
    # calculate 2nd term for IQ loss
    value_loss = np.mean(values[s] - (1 - d) * gamma * values[s_next])

    loss += value_loss
    loss_dict["value_loss"] = value_loss
    loss_dict["total_loss"] = loss

    return loss_dict
