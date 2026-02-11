# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:47:29 2025

@author: kudva.7
"""

import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# True model functions
# -----------------------------
model_funcs = {
    'inh': lambda x, ai: x**2 / (ai**2 + x**2),
    'act': lambda x, bi: bi**2 / (bi**2 + x**2),
}

def base(t, x, u_func):
    """Underlying system dynamics."""
    x1, x2, x3, x4 = x
    u = u_func(t)
    a1, a2, a3, a4 = 0.6, 0.2, 0.2, 0.5
    b1, b2, b3, b4, b5 = 0.4, 0.7, 0.3, 0.5, 0.4
    inh, act = model_funcs['inh'], model_funcs['act']

    dx1 = -x1 + 0.5 * (act(x3, b4) * inh(u, a1) + inh(x2, a3))
    dx2 = -x2 + inh(x1, a2) * act(x3, b3)
    dx3 = -x3 + act(x2, b2) * act(x4, b5)
    dx4 = -x4 + 0.5 * (act(u, b1) + inh(x3, a4))
    return [dx1, dx2, dx3, dx4]

# -----------------------------
# Random Input Generator with Δu constraint
# -----------------------------
def generate_input_signal_observed(t_train, u_min=0.5, u_max=2.0, dt_obs=1.0, delta_u=0.1, seed=None):
    rng = np.random.default_rng(seed)
    t_obs = np.arange(0, t_train[-1] + dt_obs, dt_obs)
    n_obs = len(t_obs)

    u_values = np.zeros(n_obs)
    u_values[0] = rng.uniform(u_min, u_max)
    for i in range(1, n_obs):
        low = max(u_min, u_values[i-1] - delta_u)
        high = min(u_max, u_values[i-1] + delta_u)
        u_values[i] = rng.uniform(low, high)

    def u_func(tt):
        idx = np.searchsorted(t_obs, tt, side="right") - 1
        idx = np.clip(idx, 0, n_obs-1)
        return u_values[idx]

    return t_obs, u_values, u_func

# -----------------------------
# Trajectory Generator
# -----------------------------
def generate_trajectory_randomIC(t_span=[0, 15], dt=0.1,
                                 x0_lower=[0, 0, 0.2, 0.5],
                                 x0_upper=[0.2, 0.2, 0.4, 0.7],
                                 u_min=0.5, u_max=2.0,
                                 delta_u=0., seed=None):

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(x0_lower, x0_upper)
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    _, _, u_func = generate_input_signal_observed(t_eval, u_min, u_max, dt_obs=1.0, delta_u=delta_u, seed=seed)

    sol = solve_ivp(lambda t, x: base(t, x, u_func),
                    t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-8)
    return t_eval, sol.y.T, u_func

# -----------------------------
# Dataset creation
# -----------------------------
# def create_training_data_randomIC(n_traj=25, observed_state_idx=[2], t_span=[0,15], dt=0.1, seed_val = 1., delta_u = 0.):
#     """
#     Generate trajectories with random initial conditions.
#     Adds observation sampling and linear interpolation.
#     Returns:
#         all_data: list of (t_tensor, x_interp_tensor, u_tensor)
#         obs_idx: indices of observation times
#     """
#     all_data = []

#     fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
#     axs = axs.flatten()

#     for i in range(n_traj):
#         seed = i*seed_val
#         t_full, x_full, u_func = generate_trajectory_randomIC(
#             t_span=t_span, dt=dt, seed=seed, delta_u = delta_u
#         )

#         #pdb.set_trace()

#         # --- Generate input trajectory ---
#         u_traj = np.array([u_func(t) for t in t_full])

#         # --- Define observation indices ---
#         obs_idx = np.arange(0, len(t_full), int(1.0 / dt))
#         t_obs = t_full[obs_idx]
#         x_obs = x_full[obs_idx][:, observed_state_idx]

#         # --- Add small noise to observed data ---
#         # x_obs *= np.random.lognormal(mean=0.0, sigma=0.001, size=x_obs.shape)
#         # x_obs += np.random.normal(loc=0.0, scale=0.001, size=x_obs.shape)

#         # --- Linear interpolation for each observed state ---
#         x_interp = np.stack(
#             [np.interp(t_full, t_obs, x_obs[:, j]) for j in range(len(observed_state_idx))],
#             axis=-1
#         )

#         # --- Convert to tensors ---
#         t_tensor = torch.tensor(t_full, dtype=torch.float32)
#         x_tensor = torch.tensor(x_interp, dtype=torch.float32)
#         u_tensor = torch.tensor(u_traj, dtype=torch.float32).unsqueeze(-1)
#         x0_tensor = torch.tensor(x_full[0,:], dtype=torch.float32)

#         all_data.append((t_tensor, x_tensor, u_tensor, x0_tensor))

#         # --- Plot each state and input ---
#         axs[0].plot(t_full, x_full[:, 0], color='gray', alpha=0.4)
#         axs[1].plot(t_full, x_full[:, 1], color='gray', alpha=0.4)
#         axs[2].plot(t_full, x_full[:, 2], color='gray', alpha=0.4)
#         axs[3].plot(t_full, x_full[:, 3], color='gray', alpha=0.4)
#         axs[4].step(t_full, u_traj, where="post", color='gray', alpha=0.4)

#         # Plot observed points and interpolated lines for selected states
#         for j, idx in enumerate(observed_state_idx):
#             axs[idx].scatter(t_obs, x_obs[:, j], s=20, label=f"Obs x{idx+1}", alpha=0.7)
#             axs[idx].plot(t_full, x_interp[:, j], '--', label=f"Interp x{idx+1}", linewidth=1.2)

#     # --- Labels and formatting ---
#     labels = ["x1", "x2", "x3", "x4", "u(t)"]
#     for ax, label in zip(axs, labels):
#         ax.set_ylabel(label)
#         ax.grid(True)

#     axs[-1].set_xlabel("Time")
#     axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
#     plt.suptitle("Generated trajectories with random ICs and linear interpolation")
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.show()

#     return all_data, obs_idx

def create_training_data_randomIC(
    n_traj=25, 
    observed_state_idx=[2], 
    t_span=[0, 15], 
    dt=0.1, 
    seed_val=1.0, 
    delta_u=0.0
):
    """
    Generate trajectories with random initial conditions.
    Adds observation sampling and linear interpolation.

    Args:
        n_traj (int): Number of trajectories to generate
        observed_state_idx (list): indices of observed states
        t_span (list): start and end time
        dt (float): time step
        seed_val (float): seed multiplier for reproducibility
        delta_u (float): optional input perturbation

    Returns:
        all_data: list of tuples (t_tensor, x_interp_tensor, u_tensor, x0_tensor)
        obs_idx: indices of observation times
    """

    all_data = []

    # Figure setup for publication-quality
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), dpi=300, sharex=True)
    axs = axs.flatten()

    # Colors for clarity
    obs_color = "#0072B2"    # blue
    #interp_color = "#D55E00" # orange
    bg_color = "blue"
    test = True
    
    for i in range(n_traj):
        seed = i * seed_val
        t_full, x_full, u_func = generate_trajectory_randomIC(
            t_span=t_span, dt=dt, seed=seed, delta_u=delta_u
        )

        # --- Generate input trajectory ---
        u_traj = np.array([u_func(t) for t in t_full])

        # --- Observation indices ---
        obs_idx = np.arange(0, len(t_full), int(1.0 / dt))
        t_obs = t_full[obs_idx]
        x_obs = x_full[obs_idx][:, observed_state_idx]
        
        # --- Add small noise to observed data ---
        x_obs *= np.random.lognormal(mean=0.0, sigma=0.01, size=x_obs.shape)
        x_obs += np.random.normal(loc=0.0, scale=0.01, size=x_obs.shape)

        # --- Linear interpolation for each observed state ---
        x_interp = np.stack(
            [np.interp(t_full, t_obs, x_obs[:, j]) for j in range(len(observed_state_idx))],
            axis=-1
        )

        # --- Convert to tensors ---
        t_tensor = torch.tensor(t_full, dtype=torch.float32)
        x_tensor = torch.tensor(x_interp, dtype=torch.float32)
        u_tensor = torch.tensor(u_traj, dtype=torch.float32).unsqueeze(-1)
        x0_tensor = torch.tensor(x_full[0, :], dtype=torch.float32)

        all_data.append((t_tensor, x_tensor, u_tensor, x0_tensor))

        # --- Plot background trajectories ---
        for j in range(4):
            axs[j].plot(t_full, x_full[:, j], color=bg_color, alpha=0.4, linewidth=1.5, label="State Trajectory")
            # if test:
            #     axs[j].legend()
                
        axs[4].step(t_full, u_traj, where="post", color="black", alpha=0.4, linewidth=4.)
        
        
        
        # --- Plot observed points and interpolated lines ---
        for j, idx in enumerate(observed_state_idx):
            axs[idx].scatter(t_obs, x_obs[:, j], s=40, color="red", label=f"Experimental Observation $x_{idx+1}$", alpha=0.8)
            if test:
                axs[idx].legend(fontsize = 15)
                test = False
            #axs[idx].plot(t_full, x_interp[:, j], '--', color=interp_color, linewidth=2, label=f"Interp x{idx+1}")

    # --- Labels, ticks, and grid ---
    state_labels = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$u(t)$"]
    for ax, label in zip(axs, state_labels):
        ax.set_ylabel(label, fontsize=30)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    axs[-1].set_xlabel("Time (hours)", fontsize=30)

    # --- Legend ---
    #axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, frameon=True, edgecolor='black')

    # --- Title and layout ---
    #plt.suptitle("Generated trajectories with random ICs and linear interpolation", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("test_data_NBFK.pdf", format="pdf", bbox_inches="tight")
    plt.show()    

    return all_data, obs_idx


# -----------------------------
# Dataset and DataLoader
# -----------------------------
class TrajectoryDataset(data.Dataset):
    def __init__(self, train_tensors):
        self.train_tensors = train_tensors

    def __len__(self):
        return len(self.train_tensors)

    def __getitem__(self, idx):
        #pdb.set_trace()
        
        t_tensor, x_tensor, u_tensor, x0 = self.train_tensors[idx]
        fixed_params = torch.tensor([])
        
        return x0, x_tensor, t_tensor, u_tensor, fixed_params


if __name__ == "__main__":
    
    import pickle
    # -----------------------------
    # Example usage
    # -----------------------------
    observed_state_idx = [2]
    train_tensors, obs_idx = create_training_data_randomIC(n_traj=5, observed_state_idx=observed_state_idx, seed_val = 400, delta_u = 0.1)
    
    data = {}
    data['train_tensors'] = train_tensors
    data['obs_idx'] = obs_idx
    
    # with open("NBF_K_data", "wb") as f:
    #     pickle.dump(data, f)
        
        
    # dataset = TrajectoryDataset(train_tensors)
    # dataloader = data.DataLoader(dataset, batch_size=25, shuffle=True, num_workers=2)
    # all_idx = torch.tensor([i for i in range(151)]) # This is incase we want to use all the