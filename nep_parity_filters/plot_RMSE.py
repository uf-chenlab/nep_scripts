import os
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# TRAINING PLOTS
# ======================================================
output_dir = "nep_training_plots"
os.makedirs(output_dir, exist_ok=True)

energy_train = np.loadtxt("energy_train.out")
virial_train = np.loadtxt("virial_train.out")
force_train  = np.loadtxt("force_train.out")
loss         = np.loadtxt("loss.out")

# ----------------------
# ENERGY RMSE (TRAIN)
# ----------------------
energy_rmse_train = np.sqrt(
    np.mean((energy_train[:, 0] - energy_train[:, 1])**2)
)

# ----------------------
# Energy parity plot
# ----------------------
plt.figure()
plt.plot(energy_train[:, 1], energy_train[:, 0], '.', markersize=10)

ref = np.arange(-7.5, -6.0, 0.01)
plt.plot(ref, ref, linewidth=2)

plt.xlabel(r'DFT energy (eV/atom)', fontsize=15)
plt.ylabel(r'NEP energy (eV/atom)', fontsize=15)
plt.tick_params(axis='both', labelsize=15, length=10)

plt.text(
    0.05, 0.95,
    f"RMSE = {energy_rmse_train:.4f} eV/atom",
    transform=plt.gca().transAxes,
    fontsize=14,
    va='top'
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "energy_parity.png"), dpi=300)
plt.close()

# ----------------------
# FORCE RMSE (TRAIN)
# ----------------------
nep_f_train = force_train[:, 0:3].flatten()
dft_f_train = force_train[:, 3:6].flatten()

force_rmse_train = np.sqrt(
    np.mean((nep_f_train - dft_f_train)**2)
)

# ----------------------
# Force parity plot
# ----------------------
plt.figure()
plt.plot(dft_f_train, nep_f_train, '.', markersize=10)

ref = np.arange(-4.0, 4.0, 0.01)
plt.plot(ref, ref, linewidth=2)

plt.xlabel(r'DFT force (eV/$\AA$)', fontsize=15)
plt.ylabel(r'NEP force (eV/$\AA$)', fontsize=15)
plt.tick_params(axis='both', labelsize=15, length=10)

plt.text(
    0.05, 0.95,
    f"RMSE = {force_rmse_train:.4f} eV/$\\AA$",
    transform=plt.gca().transAxes,
    fontsize=14,
    va='top'
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "force_parity.png"), dpi=300)
plt.close()

# ----------------------
# Loss evolution (log-log)
# ----------------------
plt.figure()
plt.loglog(loss[:, 1:6], linewidth=2)

plt.xlabel(r'Generation / 100', fontsize=15)
plt.ylabel(r'Loss functions', fontsize=15)
plt.tick_params(axis='both', labelsize=15, length=10)
plt.legend(
    ['Total', 'L1-Reg', 'L2-Reg', 'Energy-train', 'Force-train'],
    fontsize=12
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=300)
plt.close()

print(f"Training plots saved in: {os.path.abspath(output_dir)}")

# ======================================================
# TEST PLOTS
# ======================================================
output_dir = "nep_parity_plots"
os.makedirs(output_dir, exist_ok=True)

energy_test = np.loadtxt("energy_test.out")
force_test  = np.loadtxt("force_test.out")

# ----------------------
# ENERGY RMSE (TEST)
# ----------------------
energy_rmse_test = np.sqrt(
    np.mean((energy_test[:, 0] - energy_test[:, 1])**2)
)

# ----------------------
# Energy parity plot
# ----------------------
plt.figure()
plt.plot(energy_test[:, 1], energy_test[:, 0], '.', markersize=10)

ref = np.arange(-7.5, -6.0, 0.01)
plt.plot(ref, ref, linewidth=2)

plt.xlabel(r'DFT energy (eV/atom)', fontsize=15)
plt.ylabel(r'NEP energy (eV/atom)', fontsize=15)
plt.tick_params(axis='both', labelsize=15, length=10)

plt.text(
    0.05, 0.95,
    f"RMSE = {energy_rmse_test:.4f} eV/atom",
    transform=plt.gca().transAxes,
    fontsize=14,
    va='top'
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "energy_parity.png"), dpi=300)
plt.close()

# ----------------------
# FORCE RMSE (TEST)
# ----------------------
nep_f_test = force_test[:, 0:3].flatten()
dft_f_test = force_test[:, 3:6].flatten()

force_rmse_test = np.sqrt(
    np.mean((nep_f_test - dft_f_test)**2)
)

# ----------------------
# Force parity plot
# ----------------------
plt.figure()
plt.plot(dft_f_test, nep_f_test, '.', markersize=10)

ref = np.arange(-4.0, 4.0, 0.01)
plt.plot(ref, ref, linewidth=2)

plt.xlabel(r'DFT force (eV/$\AA$)', fontsize=15)
plt.ylabel(r'NEP force (eV/$\AA$)', fontsize=15)
plt.tick_params(axis='both', labelsize=15, length=10)

plt.text(
    0.05, 0.95,
    f"RMSE = {force_rmse_test:.4f} eV/$\\AA$",
    transform=plt.gca().transAxes,
    fontsize=14,
    va='top'
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "force_parity.png"), dpi=300)
plt.close()

print(f"Test plots saved in: {os.path.abspath(output_dir)}")
