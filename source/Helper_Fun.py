# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 11:09:50 2025

@author: kudva.7
"""

import matplotlib.pyplot as plt
from TNF_alpha_training_data import *



def visualize_using_plots_test(
    n,
    x_label,
    u_label,
    base,
    predictors=None,
    obs_state_idx=[2],
    T=30,
    seed=300
):
    """
    Visualization for TEST data (random IC) using the true system via solve_ivp
    and using multiple predictor models (same style as training visualize function).
    """

    device = "cpu"

    # ------------------------------------------------------------
    # STEP 1: Create synthetic test data using your random IC generator
    # ------------------------------------------------------------
    test_tensors, obs_idx = create_training_data_randomIC(
        n_traj=n,
        observed_state_idx=obs_state_idx,
        seed_val=seed,
        t_span=[0, T],
        delta_u=0.1
    )

    test_dataset = TrajectoryDataset(test_tensors)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Colors for models
    colors_predictor = ["green", "black", "orange", "maroon"]

    # Extract model names
    if predictors is not None:
        model_names = list(predictors.keys())

    all_predictions = []

    # ------------------------------------------------------------
    # STEP 2: Loop through each trajectory
    # ------------------------------------------------------------
    for x0_batch, x_obs_batch, t_obs_batch, u_obs_batch, fixed_params_batch in test_dataloader:

        # Get tensors
        x0 = x0_batch[0].to(device)
        x_obs = x_obs_batch[0].to(device)
        t_obs = t_obs_batch[0].to(device)
        u_obs = u_obs_batch[0].to(device)
        fixed_params = fixed_params_batch[0]

        t_np = t_obs.cpu().numpy()
        u_np = u_obs.cpu().numpy()

        # ------------------------------------------------------------
        # STEP 3: Construct true trajectory using SciPy solve_ivp
        # ------------------------------------------------------------
        from scipy.interpolate import interp1d

        u_interp = interp1d(
            t_np.flatten(),
            u_np.flatten(),
            kind="linear",
            fill_value="extrapolate"
        )
        u_func = lambda t: float(u_interp(t))

        x0_states = x0[:4].cpu().numpy()
        t_span = [float(t_np[0]), float(t_np[-1])]

        sol = solve_ivp(
            lambda t, x: base(t, x, u_func),
            t_span,
            x0_states,
            t_eval=t_np,
            rtol=1e-8,
            atol=1e-8
        )
        true_states = sol.y.T  # shape (T, 4)

        # ------------------------------------------------------------
        # STEP 4: Predictor model predictions
        # ------------------------------------------------------------
        pred_list = []

        if predictors is not None:
            for name in model_names:
                model = predictors[name]
                model.eval()

                # assign control and time to hybrid model
                model.hybrid_model.u_obs = u_obs.unsqueeze(0)
                model.hybrid_model.t_obs = t_obs
                model.hybrid_model.fixed_params = fixed_params

                with torch.no_grad():
                    x0_input = x0.unsqueeze(0)
                    t_batch = t_obs.unsqueeze(0)
                    pred_traj = model(x0_input, t_batch).squeeze(0).cpu().numpy()

                pred_list.append(pred_traj)

        all_predictions.append(pred_list)

        # ------------------------------------------------------------
        # STEP 5: Plotting (consistent with training visualization)
        # ------------------------------------------------------------
        num_states = len(x_label)
        num_inputs = len(u_label)

        fig, axes = plt.subplots(num_states + num_inputs, 1, figsize=(10, 14), sharex=True)
        fig.suptitle("Trajectory Visualization (Test Data)", fontsize=14, y=0.95)

        # --- Plot States ---
        for j in range(num_states):
            # True trajectory
            axes[j].scatter(t_np, true_states[:, j], color="red", label="True")

            # predictors
            if predictors is not None:
                for k, name in enumerate(model_names):
                    pred_traj = pred_list[k]
                    axes[j].plot(
                        t_np, pred_traj[:, :, j],
                        color=colors_predictor[k],
                        label=name
                    )

            axes[j].set_ylabel(x_label[j])
            axes[j].grid(True)
            axes[j].legend()

        # --- Plot Inputs ---
        for j in range(num_inputs):
            axes[num_states + j].plot(t_np, u_np[:, j], linewidth=2)
            axes[num_states + j].set_ylabel(u_label[j])
            axes[num_states + j].grid(True)

        axes[-1].set_xlabel("Time")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return all_predictions



if __name__ == "__main__":
    
    from torchdyn.models import NeuralODE    
    from TNF_K_FPM import NBFK_FPM
    from hybrid_ode import HybridODE
    from hybrid_model_learner_v1 import HybridLearner_v1
    import torch
    import sys
    import pytorch_lightning as pl
    import torch.nn as nn
    from TNF_alpha_training_data import *
    import pickle
    
    with open("NBF_K_data", "rb") as f:
        train_data = pickle.load(f)
        
        
    train_tensors = train_data['train_tensors']
    obs_idx = train_data['obs_idx']    
    observed_state_idx = [2]
    
    mech_param_dict = {'a1': 0.6,
     'a2': 0.2,
     'a3': 0.2,
     'b1': 0.4,
     'b2': 0.7,
     'b3': 0.3, 
     'b4': 0.5,
     'b5': 0.4}
    
    
    # Fermentor FPM:
    nbf_ode = NBFK_FPM(mech_params = mech_param_dict, mode = "inference")
    
    hybrid_model = NeuralODE(nbf_ode, solver='rk4', sensitivity='interpolated_adjoint',   #'adjoint' 'tsit5
                          rtol=1e-3, atol=1e-3).to(device)
    
    learner_hybrid = HybridLearner_v1(node_model = hybrid_model,
                    hybrid_model = nbf_ode,
                    observed_idx = obs_idx,
                    BIMT_model = None,
                    observed_state = observed_state_idx,
                    params_to_train = nbf_ode.parameters(),   # only physical params
                    optimizer_type = "adamw",
                    lr = 0.01)
    
    predictor = {'FPM_optimal': learner_hybrid}
    
    visualize_using_plots_test(n = 5 ,base = base, x_label = ['x1','x2','x3','x4'] ,u_label = ['u'], predictors = predictor, obs_state_idx = observed_state_idx, T = 30, seed = 300)
    
    