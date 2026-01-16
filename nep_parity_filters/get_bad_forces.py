import numpy as np
import re
from collections import defaultdict

# ============================================================
# USER SETTINGS
# ============================================================

BAD_MODE = "norm_thresh"
# Options:
#   "topN"             → worst N atoms by |ΔF|
#   "norm_thresh"      → |ΔF| > threshold
#   "component_thresh" → max(|ΔFx|,|ΔFy|,|ΔFz|) > threshold

# ---- Parameters for each mode ----
N_WORST_ATOMS = 50
NORM_THRESHOLD = 1.0      # eV/Å
COMP_THRESHOLD = 1.0     # eV/Å

# ---- Files ----
xyz_file = "test.xyz"
force_file = "force_train.out"
output_xyz = "bad_forces.xyz"
summary_file = "bad_forces_summary.txt"

# ============================================================
# Load force data
# ============================================================
forces = np.loadtxt(force_file)

f_nep = forces[:, 0:3]
f_dft = forces[:, 3:6]

df = f_nep - f_dft
abs_df = np.abs(df)

# ============================================================
# Select bad atoms
# ============================================================
if BAD_MODE == "topN":
    force_error = np.linalg.norm(df, axis=1)
    bad_atom_set = set(np.argsort(force_error)[-N_WORST_ATOMS:])

elif BAD_MODE == "norm_thresh":
    force_error = np.linalg.norm(df, axis=1)
    bad_atom_set = set(np.where(force_error > NORM_THRESHOLD)[0])

elif BAD_MODE == "component_thresh":
    force_error = np.linalg.norm(df, axis=1)
    bad_atom_set = set(
        np.where(np.max(abs_df, axis=1) > COMP_THRESHOLD)[0]
    )

else:
    raise ValueError(f"Unknown BAD_MODE: {BAD_MODE}")

print(f"Bad atom selection mode: {BAD_MODE}")
print(f"Total bad atoms: {len(bad_atom_set)}")

# ============================================================
# Parse XYZ (preserve comment lines)
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

print(f"Read {len(frames)} frames")
print(f"Read {atom_idx} atoms")

# ============================================================
# Compute structure-level statistics
# ============================================================
structure_stats = []

for frame_id, atom_indices in enumerate(frame_atom_indices):

    bad_species = set()
    bad_species_counts = defaultdict(int)
    species_force_errors = defaultdict(list)

    bad_atom_count = 0

    for atom_idx, atom_line in zip(atom_indices, frames[frame_id][2:]):
        if atom_idx in bad_atom_set:
            elem = atom_line.split()[0]
            ferr = force_error[atom_idx]

            bad_atom_count += 1
            bad_species.add(elem)
            bad_species_counts[elem] += 1
            species_force_errors[elem].append(ferr)

    if bad_atom_count > 0:
        errs = force_error[atom_indices]

        structure_stats.append({
            "frame": frame_id,
            "natoms": len(atom_indices),
            "bad_atom_count": bad_atom_count,
            "max_err": errs.max(),
            "mean_err": errs.mean(),
            "bad_species": sorted(bad_species),
            "bad_species_counts": dict(bad_species_counts),
            "species_force_errors": dict(species_force_errors)
        })

# Rank by worst atom in structure
structure_stats.sort(key=lambda x: x["max_err"], reverse=True)

# ============================================================
# Write summary file
# ============================================================
with open(summary_file, "w") as f:
    f.write(
        "# Rank Frame_ID NumAtoms NumBadAtoms "
        "MaxForceErr(eV/A) MeanForceErr(eV/A) "
        "BadSpecies BadSpeciesCounts SpeciesForceStats\n"
    )

    for rank, s in enumerate(structure_stats, start=1):

        species_list = ",".join(s["bad_species"])
        species_counts = ",".join(
            f"{k}:{v}" for k, v in s["bad_species_counts"].items()
        )

        species_stats = []
        for sp, ferrs in s["species_force_errors"].items():
            species_stats.append(
                f"{sp}(μ={np.mean(ferrs):.2f},max={np.max(ferrs):.2f})"
            )
        species_stats_str = ";".join(species_stats)

        f.write(
            f"{rank:4d} {s['frame']:8d} {s['natoms']:8d} "
            f"{s['bad_atom_count']:12d} "
            f"{s['max_err']:14.6f} {s['mean_err']:14.6f} "
            f"{species_list} {species_counts} {species_stats_str}\n"
        )

print(f"Wrote summary → {summary_file}")

# ============================================================
# Write corrected XYZ (OVITO-safe)
# ============================================================
with open(output_xyz, "w") as f:
    for s in structure_stats:
        frame = s["frame"]
        atom_indices = frame_atom_indices[frame]

        f.write(f"{len(atom_indices)}\n")

        # Original comment line
        orig_comment = frames[frame][1].strip()

        # Remove any existing Properties= field
        orig_comment = re.sub(r'Properties=[^\s]+', '', orig_comment)

        # Correct Properties definition
        new_properties = (
            "Properties=species:S:1:pos:R:3:ForceError:R:1:IsBad:I:1"
        )

        f.write(
            f"{orig_comment} {new_properties} | "
            f"Frame={frame} | "
            f"NumBadAtoms={s['bad_atom_count']} | "
            f"MaxForceErr={s['max_err']:.6f} eV/A | "
            f"MeanForceErr={s['mean_err']:.6f} eV/A\n"
        )

        for line, atom_idx in zip(frames[frame][2:], atom_indices):
            parts = line.split()
            elem = parts[0]
            x, y, z = parts[1:4]

            ferr = force_error[atom_idx]
            is_bad = 1 if atom_idx in bad_atom_set else 0

            f.write(
                f"{elem} {x} {y} {z} {ferr:.6f} {is_bad}\n"
            )

print(f"Wrote annotated XYZ → {output_xyz}")
