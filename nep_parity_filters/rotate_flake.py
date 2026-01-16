#!/usr/bin/env python3
"""
Post-process a POSCAR containing:
  - periodic sapphire substrate
  - finite MoS2 flake on top

This script:
  - separates substrate vs flake using a USER-DEFINED Z cut
  - rigidly translates and rotates ONLY the flake
  - keeps the substrate fixed
  - writes POSCARs into folders
  - writes an OVITO-friendly all_structures.xyz with metadata
"""

import os
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# ============================================================
# ===================== USER PARAMETERS ======================
# ============================================================

INPUT_POSCAR = "POSCAR"      # reference flake+substrate structure
OUTPUT_BASE_DIR = "post_processed"

# ---------- Flake identification ----------
# Atoms with z > Z_CUT are treated as the flake
Z_CUT = 26.0  # Å  <-- USER CONTROL

# ---------- Displacement sampling ----------
N_DISP = 4
DISPLACEMENTS_FRAC = [
    (i / N_DISP, j / N_DISP)
    for i in range(N_DISP)
    for j in range(N_DISP)
]

# ---------- Rotation sampling ----------
ROTATION_DEGREES = [0, 60, 120, 180, 240, 300]

# ---------- XYZ output ----------
WRITE_EXTXYZ = True
EXTXYZ_FILENAME = "all_structures.xyz"

# ============================================================


# ============================================================
# ===================== HELPER FUNCTIONS =====================
# ============================================================

def rotate_about_z(coords, theta_deg, center_xy):
    """Rotate Cartesian coordinates about Z."""
    theta = np.deg2rad(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    out = coords.copy()
    out[:, :2] = (out[:, :2] - center_xy) @ R.T + center_xy
    return out


def make_shift_cart_from_frac(dx_frac, dy_frac, lattice):
    """Fractional → Cartesian shift."""
    return (
        dx_frac * lattice.matrix[0] +
        dy_frac * lattice.matrix[1]
    )


def write_extended_xyz(structures, metadata, filename):
    """Write OVITO-friendly extended XYZ with per-frame metadata."""
    with open(filename, "w") as f:
        for i, (s, meta) in enumerate(zip(structures, metadata)):
            lat = s.lattice.matrix.flatten()

            f.write(f"{len(s)}\n")

            header = (
                'Lattice="{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}" '
                'Properties=species:S:1:pos:R:3 '
                f'frame={i} '
            ).format(*lat)

            for k, v in meta.items():
                header += f"{k}={v} "

            f.write(header.strip() + "\n")

            for site in s:
                x, y, z = site.coords
                f.write(f"{site.species_string} {x:.6f} {y:.6f} {z:.6f}\n")


# ============================================================
# ============================ MAIN ==========================
# ============================================================

def main():

    struct = Structure.from_file(INPUT_POSCAR)
    lattice = struct.lattice
    coords = struct.cart_coords
    species = list(struct.species)

    # --------------------------------------------------------
    # Separate substrate and flake using Z_CUT
    # --------------------------------------------------------
    flake_mask = coords[:, 2] > Z_CUT
    sub_mask   = ~flake_mask

    if not np.any(flake_mask):
        raise RuntimeError(
            "No flake atoms found above Z_CUT. "
            "Check your Z_CUT value."
        )

    sub_coords   = coords[sub_mask]
    sub_species  = [species[i] for i in np.where(sub_mask)[0]]

    flake_coords0 = coords[flake_mask]
    flake_species = [species[i] for i in np.where(flake_mask)[0]]

    # Flake center for rotation
    flake_center_xy = flake_coords0[:, :2].mean(axis=0)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    all_structures = []
    all_metadata   = []

    # --------------------------------------------------------
    # Loop over displacements and rotations
    # --------------------------------------------------------
    for dx_frac, dy_frac in DISPLACEMENTS_FRAC:

        shift_cart = make_shift_cart_from_frac(dx_frac, dy_frac, lattice)

        disp_dir = os.path.join(
            OUTPUT_BASE_DIR,
            f"dx_{dx_frac:.2f}_dy_{dy_frac:.2f}"
        )
        os.makedirs(disp_dir, exist_ok=True)

        for theta in ROTATION_DEGREES:

            flake_coords = flake_coords0.copy()

            # Rotate flake
            flake_coords = rotate_about_z(
                flake_coords, theta, flake_center_xy
            )

            # Translate flake
            flake_coords[:, :2] += shift_cart[:2]

            # Reassemble structure
            new_coords  = np.vstack([sub_coords, flake_coords])
            new_species = sub_species + flake_species

            new_struct = Structure(
                lattice=lattice,
                species=new_species,
                coords=new_coords,
                coords_are_cartesian=True
            )

            # Write POSCAR
            rot_dir = os.path.join(disp_dir, f"rot_{theta:03d}")
            os.makedirs(rot_dir, exist_ok=True)
            Poscar(new_struct).write_file(
                os.path.join(rot_dir, "POSCAR")
            )

            # Save for XYZ
            all_structures.append(new_struct)
            all_metadata.append({
                "dx_frac": dx_frac,
                "dy_frac": dy_frac,
                "theta": theta,
                "z_cut": Z_CUT,
                "flake_atoms": len(flake_species),
                "uid": f"dx{dx_frac}_dy{dy_frac}_th{theta}"
            })

            print(
                f"dx={dx_frac:.2f} dy={dy_frac:.2f} "
                f"θ={theta:3d}° → {rot_dir}"
            )

    # --------------------------------------------------------
    # Write OVITO XYZ
    # --------------------------------------------------------
    if WRITE_EXTXYZ:
        write_extended_xyz(
            all_structures,
            all_metadata,
            os.path.join(OUTPUT_BASE_DIR, EXTXYZ_FILENAME)
        )
        print(f"\n🧩 Wrote OVITO XYZ → {EXTXYZ_FILENAME}")

    print("\n✅ Post-processing move + rotate complete.")


if __name__ == "__main__":
    main()
