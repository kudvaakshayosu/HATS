# HATS – Preliminary TNF-α Case Study

This repository contains the implementation used for the **first case study** of the HATS framework, focused on hybrid modeling of the TNF-α system.

> ⚠️ **Important:**  
> This repository is intended only as a preliminary implementation for the first case study and associated experimental conditions. It does **not** represent the complete HATS framework.  
> The SHAP-based feature attribution analysis and symbolic regression components are **not included** in this repository and must be implemented separately.

---

## Repository Structure

```
source/
├── Helper_Fun.py
├── hybrid_model_learner_v1.py
├── hybrid_ode.py
├── TNF_alpha_training_data.py
├── TNF_K_FPM.py
└── TNF_K_Hybrid.py
```

---

## File Descriptions

### Main Scripts

**TNF_alpha_training_data.py**  
This script must be executed **first**.  
It generates the TNF-α training trajectories and stores them in a `.pkl` file.  
This file is required for training the hybrid model.

**TNF_K_FPM.py**  
Implements the first-principles (mechanistic) TNF-α model.

**TNF_K_Hybrid.py**  
Trains the hybrid model using the generated dataset.

---

### Helper Modules

The following files contain utility functions and model classes built around the `torchdyn` library:

- **Helper_Fun.py** – Helper utilities for data processing and plotting
- **hybrid_model_learner_v1.py** – Hybrid model training wrapper implemented with PyTorch Lightning.
- **hybrid_ode.py** – Defines the hybrid neural ODE architecture - defines custom HybridODE class.

These modules provide abstractions for defining and training hybrid neural ODE models.

---

## Installation

Create a virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 – Generate Training Data

From the `source/` directory:

```bash
python TNF_alpha_training_data.py
```

This generates a pickle file containing the training trajectories.

### Step 2 – Train the Hybrid Model

```bash
python TNF_K_Hybrid.py
```

---

## Notes

- This repository serves as a **minimal working implementation** for the TNF-α case study.
- It does **not** include:
  - SHAP-based attribution analysis
  - Feature pruning logic
  - Symbolic regression
  - Full HATS pipeline automation
- These components must be implemented separately to reproduce the complete HATS framework.

---

## Dependencies

Core libraries used:

- PyTorch
- PyTorch Lightning
- NumPy
- SciPy
- Matplotlib
- torchdyn

See `requirements.txt` for details.

---

For any questions regarding the code, simulation setup, or implementation details, please feel free to reach out via email: kudva.7@osu.edu
