#!/usr/bin/env python3
"""
Identify force parity outliers using signed perpendicular distance
from the parity diagonal (F_NEP = F_DFT).

Separates:
- OVERPREDICTED forces (NEP > DFT, above diagonal)
- UNDERPREDICTED forces (NEP < DFT, below diagonal)

Outputs OVITO-safe XYZ files and signed summaries.
"""

import numpy as np
import os
import re
from collections import defaultdict

# ============================================================
# USER SETTINGS
# ============================================================

PERP_THRESHOLD = 2.2  # eV/Å (distance from parity diagonal)

xyz_file   = "train.xyz"
force_file = "force_train.out"

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
OUTPUT_DIR = "force_parity_outliers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

over_xyz     = os.path.join(OUTPUT_DIR, "force_overpredicted.xyz")
under_xyz    = os.path.join(OUTPUT_DIR, "force_underpredicted.xyz")
over_summary = os.path.join(OUTPUT_DIR, "force_overpredicted_summary.txt")
under_summary= os.path.join(OUTPUT_DIR, "force_underpredicted_summary.txt")

# ============================================================
# Load force data
# ============================================================
forces = np.loadtxt(force_file)

f_nep = forces[:, 0:3]
f_dft = forces[:, 3:6]

# Signed perpendicular distance per component
d_perp = (f_nep - f_dft) / np.sqrt(2)

# Atom-level worst component (keep sign!)
abs_d = np.abs(d_perp)
worst_comp = np.argmax(abs_d, axis=1)

signed_d_atom = d_perp[np.arange(len(d_perp)), worst_comp]
abs_d_atom    = abs_d[np.arange(len(d_perp)), worst_comp]

# Select atoms
over_atoms  = set(np.where(signed_d_atom >  PERP_THRESHOLD)[0])
under_atoms = set(np.where(signed_d_atom < -PERP_THRESHOLD)[0])

print(f"Atoms OVER predicted : {len(over_atoms)}")
print(f"Atoms UNDER predicted: {len(under_atoms)}")

# ============================================================
# Parse XYZ (track atom indices)
# ============================================================
frames = []
frame_atom_indices = []

with open(xyz_file, "r") as f:
    atom_idx = 0
    while True:
        line = f.readline()
        if not line:
            break

        natoms = int(line.strip())
        comment = f.readline()

        frame_lines = [line, comment]
        indices = []

        for _ in range(natoms):
            atom_line = f.readline()
            frame_lines.append(atom_line)
            indices.append(atom_idx)
            atom_idx += 1

        frames.append(frame_lines)
        frame_atom_indices.append(indices)

print(f"Read {len(frames)} frames, {atom_idx} atoms")

# ============================================================
# Structure-level aggregation
# ============================================================
def build_stats(target_atoms, sign_label):
    stats = []

    for frame_id, atom_indices in enumerate(frame_atom_indices):
        bad_atoms = []
        species_errs = defaultdict(list)

        for atom_idx, atom_line in zip(atom_indices, frames[frame_id][2:]):
            if atom_idx in target_atoms:
                elem = atom_line.split()[0]
                bad_atoms.append(atom_idx)
                species_errs[elem].append(signed_d_atom[atom_idx])

        if bad_atoms:
            errs = np.array([signed_d_atom[i] for i in bad_atoms])

            stats.append({
                "frame": frame_id,
                "natoms": len(atom_indices),
                "n_bad": len(bad_atoms),
                "max_signed": errs[np.argmax(np.abs(errs))],
                "mean_signed": errs.mean(),
                "species_errs": dict(species_errs)
            })

    # Rank by |signed error|
    stats.sort(key=lambda x: abs(x["max_signed"]), reverse=True)
    return stats

over_stats  = build_stats(over_atoms,  "OVER")
under_stats = build_stats(under_atoms, "UNDER")

# ============================================================
# Write summaries
# ============================================================
def write_summary(fname, stats, title):
    with open(fname, "w") as f:
        f.write(f"# {title}\n")
        f.write(
            "# Rank Frame NumAtoms NumBadAtoms "
            "MaxSigned_dPerp(eV/A) MeanSigned_dPerp(eV/A) SpeciesStats\n"
        )

        for rank, s in enumerate(stats, start=1):
            species_stats = ",".join(
                f"{sp}:{len(v)}(μ={np.mean(v):+.2f},max={np.max(np.abs(v)):.2f})"
                for sp, v in s["species_errs"].items()
            )

            f.write(
                f"{rank:4d} {s['frame']:6d} {s['natoms']:8d} "
                f"{s['n_bad']:12d} "
                f"{s['max_signed']:+14.6f} {s['mean_signed']:+14.6f} "
                f"{species_stats}\n"
            )

write_summary(
    over_summary,
    over_stats,
    "FORCE OVERPREDICTION (NEP > DFT, ABOVE DIAGONAL)"
)

write_summary(
    under_summary,
    under_stats,
    "FORCE UNDERPREDICTION (NEP < DFT, BELOW DIAGONAL)"
)

print("Wrote summaries")

# ============================================================
# Write OVITO-safe XYZ
# ============================================================
def write_xyz(fname, stats, target_atoms):
    with open(fname, "w") as f:
        for s in stats:
            frame = s["frame"]
            atom_indices = frame_atom_indices[frame]

            f.write(f"{len(atom_indices)}\n")

            orig_comment = frames[frame][1].strip()
            orig_comment = re.sub(r'Properties=[^\s]+', '', orig_comment).strip()

            # Add per-atom force vectors for OVITO hover/inspect
            props = (
                "Properties=species:S:1:pos:R:3:"
                "F_dft:R:3:F_nep:R:3:"
                "d_perp:R:1:IsOutlier:I:1"
            )

            f.write(
                f"{orig_comment} {props} | "
                f"Frame={frame} | "
                f"NumOutliers={s['n_bad']} | "
                f"MaxSigned_dPerp={s['max_signed']:+.6f} eV/A\n"
            )

            for atom_line, atom_idx in zip(frames[frame][2:], atom_indices):
                parts = atom_line.split()
                elem = parts[0]
                x, y, z = parts[1:4]

                # Real + predicted forces (from your force_train.out)
                fx_nep, fy_nep, fz_nep = f_nep[atom_idx]
                fx_dft, fy_dft, fz_dft = f_dft[atom_idx]

                dval = signed_d_atom[atom_idx]
                is_bad = 1 if atom_idx in target_atoms else 0

                f.write(
                    f"{elem} {x} {y} {z} "
                    f"{fx_dft:+.6f} {fy_dft:+.6f} {fz_dft:+.6f} "
                    f"{fx_nep:+.6f} {fy_nep:+.6f} {fz_nep:+.6f} "
                    f"{dval:+.6f} {is_bad}\n"
                )

write_xyz(over_xyz,  over_stats,  over_atoms)
write_xyz(under_xyz, under_stats, under_atoms)

print(f"Wrote XYZ files to {OUTPUT_DIR}")
print("\n✔ Signed force parity outlier analysis complete.")


# ======================================================
# FORCE PARITY PLOT WITH SIGNED OUTLIERS HIGHLIGHTED
# ======================================================


# ======================================================
# Parity plot filename (criteria-aware)
# ======================================================

plot_path = os.path.join(
    OUTPUT_DIR,
    f"force_parity_thresh_{PERP_THRESHOLD:.2f}eVA.png"
)



import matplotlib.pyplot as plt

# Flatten component-wise forces
nep_forces = f_nep.flatten()
dft_forces = f_dft.flatten()

# Signed perpendicular distance (component-wise)
d_perp_components = (nep_forces - dft_forces) / np.sqrt(2)

# Masks
over_mask  = d_perp_components >  PERP_THRESHOLD
under_mask = d_perp_components < -PERP_THRESHOLD
normal_mask = ~(over_mask | under_mask)

# RMSE
force_rmse = np.sqrt(
    np.mean((nep_forces - dft_forces) ** 2)
)

plt.figure(figsize=(6,6))

# ---- Normal points ----
plt.plot(
    dft_forces[normal_mask],
    nep_forces[normal_mask],
    '.',
    markersize=8,
    alpha=0.5,
    label="Normal"
)

# ---- Overpredicted (above diagonal) ----
plt.plot(
    dft_forces[over_mask],
    nep_forces[over_mask],
    'r.',
    markersize=10,
    label="Overpredicted (NEP > DFT)"
)

# ---- Underpredicted (below diagonal) ----
plt.plot(
    dft_forces[under_mask],
    nep_forces[under_mask],
    'b.',
    markersize=10,
    label="Underpredicted (NEP < DFT)"
)

# ---- Parity line ----
ref = np.linspace(dft_forces.min(), dft_forces.max(), 400)
plt.plot(ref, ref, 'k-', linewidth=2)

plt.xlabel(r'DFT force (eV/$\AA$)', fontsize=14)
plt.ylabel(r'NEP force (eV/$\AA$)', fontsize=14)
plt.tick_params(axis='both', labelsize=12)

# ---- Annotation ----
plt.text(
    0.05, 0.95,
    f"RMSE = {force_rmse:.4f} eV/$\\AA$\n"
    f"Selection: |d⊥| > {PERP_THRESHOLD:.2f} eV/$\\AA$\n"
    f"Over: {np.sum(over_mask)}  Under: {np.sum(under_mask)}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)


plt.legend(frameon=False)
plt.tight_layout()

# plot_path = os.path.join(
#     OUTPUT_DIR, "force_parity_train_signed_outliers.png"
# )
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved parity plot with signed outliers → {plot_path}")


# ============================================================
# FILTER XYZ: REMOVE BAD STRUCTURES
# ============================================================

filtered_xyz = os.path.join(
    OUTPUT_DIR, "test_filtered.xyz"
)

# Collect bad frame IDs
bad_frames = set(
    s["frame"] for s in over_stats
).union(
    s["frame"] for s in under_stats
)

print(f"Total bad structures to remove: {len(bad_frames)}")

with open(filtered_xyz, "w") as f:
    kept = 0
    removed = 0

    for frame_id, frame_lines in enumerate(frames):
        if frame_id in bad_frames:
            removed += 1
            continue

        for line in frame_lines:
            f.write(line)
        kept += 1

print(f"Wrote filtered XYZ → {filtered_xyz}")
print(f"Kept structures   : {kept}")
print(f"Removed structures: {removed}")

