import numpy as np
import re

# ============================================================
# USER SETTINGS
# ============================================================

ENERGY_MODE = "abs_thresh"
# Options:
#   "topN"        → worst N structures by |ΔE|
#   "abs_thresh"  → |ΔE| > threshold

# ---- Parameters ----
N_WORST_STRUCTS = 50
ENERGY_THRESHOLD = 0.05   # eV per structure (adjust as needed)

# ---- Files ----
xyz_file = "test.xyz"
energy_file = "energy_train.out"
output_xyz = "bad_energy.xyz"
summary_file = "bad_energy_summary.txt"

# ============================================================
# Load energy data
# ============================================================
energies = np.loadtxt(energy_file)

E_nep = energies[:, 0]
E_dft = energies[:, 1]

energy_error = np.abs(E_nep - E_dft)

print(f"Read {len(energy_error)} energy entries")

# ============================================================
# Parse XYZ (preserve comment lines)
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

print(f"Read {len(frames)} XYZ frames")

# ============================================================
# Sanity check
# ============================================================
assert len(frames) == len(energy_error), (
    "Mismatch between XYZ frames and energy entries!"
)

# ============================================================
# Select bad structures
# ============================================================
if ENERGY_MODE == "topN":
    bad_frames = np.argsort(energy_error)[-N_WORST_STRUCTS:]

elif ENERGY_MODE == "abs_thresh":
    bad_frames = np.where(energy_error > ENERGY_THRESHOLD)[0]

else:
    raise ValueError(f"Unknown ENERGY_MODE: {ENERGY_MODE}")

bad_frames = sorted(bad_frames)

print(f"Selected {len(bad_frames)} bad-energy structures")

# ============================================================
# Build structure statistics
# ============================================================
structure_stats = []

for frame in bad_frames:
    natoms = frame_atom_counts[frame]
    Eerr = energy_error[frame]

    structure_stats.append({
        "frame": frame,
        "natoms": natoms,
        "Eerr": Eerr,
        "Eerr_pa": Eerr / natoms
    })

# Rank by worst energy error
structure_stats.sort(key=lambda x: x["Eerr"], reverse=True)

# ============================================================
# Write summary file
# ============================================================
with open(summary_file, "w") as f:
    f.write(
        "# Rank Frame_ID NumAtoms EnergyErr(eV) EnergyErrPerAtom(eV)\n"
    )

    for rank, s in enumerate(structure_stats, start=1):
        f.write(
            f"{rank:4d} {s['frame']:8d} {s['natoms']:8d} "
            f"{s['Eerr']:14.6f} {s['Eerr_pa']:18.6e}\n"
        )

print(f"Wrote summary → {summary_file}")

# ============================================================
# Write energy-annotated XYZ (OVITO-safe)
# ============================================================
with open(output_xyz, "w") as f:
    for s in structure_stats:
        frame = s["frame"]
        natoms = s["natoms"]

        f.write(f"{natoms}\n")

        # Original comment line
        orig_comment = frames[frame][1].strip()

        # Remove any existing Properties= field
        orig_comment = re.sub(r'Properties=[^\s]+', '', orig_comment)

        # Use standard XYZ properties (no per-atom energy)
        new_properties = "Properties=species:S:1:pos:R:3"

        f.write(
            f"{orig_comment} {new_properties} | "
            f"Frame={frame} | "
            f"EnergyErr={s['Eerr']:.6f} eV | "
            f"EnergyErrPerAtom={s['Eerr_pa']:.6e} eV\n"
        )

        # Write original atom lines unchanged
        for atom_line in frames[frame][2:]:
            f.write(atom_line)

print(f"Wrote annotated XYZ → {output_xyz}")
