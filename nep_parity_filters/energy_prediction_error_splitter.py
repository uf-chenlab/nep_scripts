#!/usr/bin/env python3
"""
Signed energy parity outlier detection for NEP.

Capabilities:
- Split under / over predicted structures
- OVITO-safe XYZ outputs
- Signed summaries
- Energy parity plot with highlighted outliers
- Filtered XYZ with bad structures removed
"""

import numpy as np
import re
import os
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

ENERGY_MODE = "abs_thresh"
# Options:
#   "topN"
#   "abs_thresh"

N_WORST_STRUCTS = 100
ENERGY_THRESHOLD = 0.175 # eV per structure (used if abs_thresh)

ERROR_SIDE = "both"   ### NEW
# Options:
#   "over"   → NEP > DFT only
#   "under"  → NEP < DFT only  ## if selected this means we remove this from the set
#   "both"   → both sides

xyz_file    = "train.xyz"
energy_file = "energy_train.out"

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "energy_parity_outliers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

under_xyz     = os.path.join(OUTPUT_DIR, "energy_underpredicted.xyz")
over_xyz      = os.path.join(OUTPUT_DIR, "energy_overpredicted.xyz")
under_summary = os.path.join(OUTPUT_DIR, "energy_underpredicted_summary.txt")
over_summary  = os.path.join(OUTPUT_DIR, "energy_overpredicted_summary.txt")
filtered_xyz  = os.path.join(OUTPUT_DIR, "test_filtered.xyz")

# ============================================================
# Load energy data
# ============================================================

energies = np.loadtxt(energy_file)

E_nep = energies[:, 0]
E_dft = energies[:, 1]
dE = E_nep - E_dft

print(f"Read {len(dE)} energy entries")

# ============================================================
# Parse XYZ
# ============================================================

frames = []
frame_atom_counts = []

with open(xyz_file, "r") as f:
    while True:
        line = f.readline()
        if not line:
            break

        natoms = int(line.strip())
        comment = f.readline()

        frame_lines = [line, comment]
        for _ in range(natoms):
            frame_lines.append(f.readline())

        frames.append(frame_lines)
        frame_atom_counts.append(natoms)

assert len(frames) == len(dE), "XYZ / energy mismatch"
print(f"Read {len(frames)} XYZ frames")

# ============================================================
# Select bad frames
# ============================================================

under_frames = np.where(dE < 0.0)[0]
over_frames  = np.where(dE > 0.0)[0]

def filter_frames(frame_ids):
    if ENERGY_MODE == "topN":
        idx = np.argsort(np.abs(dE[frame_ids]))[-N_WORST_STRUCTS:]
        return frame_ids[idx]
    elif ENERGY_MODE == "abs_thresh":
        return frame_ids[np.abs(dE[frame_ids]) > ENERGY_THRESHOLD]
    else:
        raise ValueError("Unknown ENERGY_MODE")

### NEW: side-aware filtering
if ERROR_SIDE in ("under", "both"):
    under_frames = filter_frames(under_frames)
else:
    under_frames = np.array([], dtype=int)

if ERROR_SIDE in ("over", "both"):
    over_frames = filter_frames(over_frames)
else:
    over_frames = np.array([], dtype=int)

print(f"Underpredicted structures: {len(under_frames)}")
print(f"Overpredicted structures : {len(over_frames)}")

# ============================================================
# Build stats
# ============================================================

def build_stats(frame_ids):
    stats = []
    for f in frame_ids:
        n = frame_atom_counts[f]
        stats.append({
            "frame": f,
            "natoms": n,
            "dE": dE[f],
            "dE_pa": dE[f] / n
        })
    return stats

under_stats = build_stats(under_frames)
over_stats  = build_stats(over_frames)

under_stats.sort(key=lambda x: x["dE"])
over_stats.sort(key=lambda x: x["dE"], reverse=True)

# ============================================================
# Write summaries
# ============================================================

def write_summary(fname, stats, title):
    with open(fname, "w") as f:
        f.write(f"# {title}\n")
        f.write("# Rank Frame NumAtoms dE(eV) dE_per_atom(eV)\n")
        for i, s in enumerate(stats, 1):
            f.write(
                f"{i:4d} {s['frame']:6d} {s['natoms']:8d} "
                f"{s['dE']:+14.6f} {s['dE_pa']:18.6e}\n"
            )

if under_stats:
    write_summary(
        under_summary, under_stats,
        "ENERGY UNDERPREDICTION (NEP < DFT)"
    )

if over_stats:
    write_summary(
        over_summary, over_stats,
        "ENERGY OVERPREDICTION (NEP > DFT)"
    )

# ============================================================
# Write XYZ files
# ============================================================

def write_xyz(fname, stats):
    with open(fname, "w") as f:
        for s in stats:
            frame = s["frame"]
            f.write(f"{s['natoms']}\n")

            comment = re.sub(
                r'Properties=[^\s]+', '',
                frames[frame][1].strip()
            )

            f.write(
                f"{comment} Properties=species:S:1:pos:R:3 | "
                f"Frame={frame} | dE={s['dE']:+.6f} eV\n"
            )

            for line in frames[frame][2:]:
                f.write(line)

if under_stats:
    write_xyz(under_xyz, under_stats)

if over_stats:
    write_xyz(over_xyz, over_stats)

# ============================================================
# Energy parity plot
# ============================================================

if ENERGY_MODE == "abs_thresh":
    parity_plot = os.path.join(
        OUTPUT_DIR,
        f"energy_parity_thresh_{ENERGY_THRESHOLD:.2f}eV.png"
    )
elif ENERGY_MODE == "topN":
    parity_plot = os.path.join(
        OUTPUT_DIR,
        f"energy_parity_topN_{N_WORST_STRUCTS}.png"
    )
else:
    parity_plot = os.path.join(OUTPUT_DIR, "energy_parity.png")

if ENERGY_MODE == "abs_thresh":
    criteria_str = f"Selection: |ΔE| > {ENERGY_THRESHOLD:.3f} eV"
else:
    criteria_str = f"Selection: Top {N_WORST_STRUCTS} by |ΔE|"

rmse = np.sqrt(np.mean(dE ** 2))

bad_frames = set(
    s["frame"] for s in under_stats
).union(
    s["frame"] for s in over_stats
)

mask_under = np.zeros(len(dE), dtype=bool)
mask_over  = np.zeros(len(dE), dtype=bool)

mask_under[list(s["frame"] for s in under_stats)] = True
mask_over[list(s["frame"] for s in over_stats)] = True
mask_normal = ~(mask_under | mask_over)

plt.figure(figsize=(6,6))

plt.plot(E_dft[mask_normal], E_nep[mask_normal], '.', alpha=0.4, label="Normal")
plt.plot(E_dft[mask_over],   E_nep[mask_over],   'r.', label="Overpredicted")
plt.plot(E_dft[mask_under],  E_nep[mask_under],  'b.', label="Underpredicted")

ref = np.linspace(E_dft.min(), E_dft.max(), 200)
plt.plot(ref, ref, 'k-', linewidth=2)

plt.xlabel("DFT energy (eV/atom)")
plt.ylabel("NEP energy (eV/atom)")

plt.text(
    0.05, 0.95,
    f"RMSE = {rmse:.4f} eV/atom\n"
    f"{criteria_str}\n"
    f"Side: {ERROR_SIDE}\n"
    f"Over: {mask_over.sum()}  Under: {mask_under.sum()}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top"
)

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(parity_plot, dpi=300)
plt.close()

# ============================================================
# Filter XYZ
# ============================================================

with open(filtered_xyz, "w") as f:
    kept = removed = 0
    for i, frame in enumerate(frames):
        if i in bad_frames:
            removed += 1
            continue
        for line in frame:
            f.write(line)
        kept += 1

print(f"Wrote filtered XYZ → {filtered_xyz}")
print(f"Kept: {kept}, Removed: {removed}")
print("\n✔ Energy parity analysis complete.")
