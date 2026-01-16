import os
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Create output directory
# ======================================================
output_dir = "nep_training_plots"
os.makedirs(output_dir, exist_ok=True)

# ======================================================
# Load TRAINING data only
# ======================================================
energy_train = np.loadtxt("energy_train.out")
force_train  = np.loadtxt("force_train.out")

# ======================================================
# ENERGY RMSE (per atom)
# ======================================================
energy_rmse = np.sqrt(
    np.mean((energy_train[:, 0] - energy_train[:, 1])**2)
)

# ======================================================
# Energy parity plot (TRAIN)
# ======================================================
plt.figure()

plt.plot(
    energy_train[:, 1],  # DFT
    energy_train[:, 0],  # NEP
    '.', markersize=10
)

ref = np.linspace(
    min(energy_train[:, 1]),
    max(energy_train[:, 1]),
    200
)
plt.plot(ref, ref, linewidth=2)

plt.xlabel(r'DFT energy (eV/atom)', fontsize=15)
plt.ylabel(r'NEP energy (eV/atom)', fontsize=15)
plt.tick_params(axis='both', labelsize=15, length=10)

# ---- RMSE annotation ----
plt.text(
    0.05, 0.95,
    f"RMSE = {energy_rmse:.4f} eV/atom",
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "energy_parity_train.png"),
    dpi=300
)
plt.close()

# ======================================================
# FORCE RMSE (component-wise)
# ======================================================
nep_forces = force_train[:, 0:3].flatten()
dft_forces = force_train[:, 3:6].flatten()

force_rmse = np.sqrt(
    np.mean((nep_forces - dft_forces)**2)
)

# ======================================================
# Force parity plot (TRAIN)
# ======================================================
plt.figure()

plt.plot(
    dft_forces,
    nep_forces,
    '.', markersize=10
)

ref = np.linspace(
    dft_forces.min(),
    dft_forces.max(),
    400
)
plt.plot(ref, ref, linewidth=2)

plt.xlabel(r'DFT force (eV/$\AA$)', fontsize=15)
plt.ylabel(r'NEP force (eV/$\AA$)', fontsize=15)
plt.tick_params(axis='both', labelsize=15, length=10)

# ---- RMSE annotation ----
plt.text(
    0.05, 0.95,
    f"RMSE = {force_rmse:.4f} eV/$\\AA$",
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "force_parity_train.png"),
    dpi=300
)
plt.close()

print(f"Training plots saved in: {os.path.abspath(output_dir)}")
