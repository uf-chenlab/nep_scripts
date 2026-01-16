#!/usr/bin/env python3
"""
Finite MoS2 triangular ZZ-edge flakes on periodic sapphire substrate with:
  - lateral displacements (registry sampling)
  - rotations
  - selectable ZZ termination (S or Mo)
  - OVITO-friendly extended XYZ metadata

Defaults:
  - triangle flake
  - ZZ-aligned edges
  - S-terminated ZZ
  - ZZ_ONLY rotation mode (0,60,120,180,240,300)

Author: Emir Bilgili workflow
"""

import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar

# ============================================================
# ===================== USER PARAMETERS ======================
# ============================================================

SUBSTRATE_POSCAR = "POSCAR_4X4X1"
EPILAYER_POSCAR  = "POSCAR_epilayer"

OUTPUT_BASE_DIR = "finite_triZZ_disp_rot"

# vdW gap: top substrate atom -> bottom S layer (Å)
Z_MEAN_LIST = [3.1]

# Build a larger MoS2 patch, then cut a finite flake
EPILAYER_SUPERCELL = (9, 9, 1)

# ---------------- Flake shape/size ----------------
FLAKE_SHAPE   = "triangle"     # currently only "triangle"
TRIANGLE_SIDE = 16.0           # Å (geometric target)

# ---------------- Edge + termination ----------------
EDGE_TYPE       = "ZZ"         # only ZZ implemented
ZZ_TERMINATION  = "S"          # "S" or "Mo"

# ---------------- Displacement sampling ----------------
N_DISP = 4
DISPLACEMENTS_FRAC = [
    (i / N_DISP, j / N_DISP)
    for i in range(N_DISP)
    for j in range(N_DISP)
]

# ---------------- Rotation sampling ----------------
ROTATION_MODE = "ZZ_ONLY"      # "ZZ_ONLY" or "GENERAL"

ZZ_ONLY_ROTATIONS = [0, 60, 120, 180, 240, 300]
GENERAL_ROTATIONS = list(range(0, 180, 10))

# ---------------- Cell / vacuum ----------------
ADD_VACUUM_Z     = True
VACUUM_THICKNESS = 10.0

# ---------------- Outputs ----------------
WRITE_EXTXYZ    = True
EXTXYZ_FILENAME = "all_structures.xyz"

# ============================================================
# ===================== HELPER FUNCTIONS =====================
# ============================================================

def add_vacuum_to_lattice(lattice, vacuum):
    a, b, c = lattice.matrix
    return Lattice([a, b, c + np.array([0.0, 0.0, vacuum])])


def rotate_about_z(coords, theta_deg, center_xy):
    theta = np.deg2rad(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    out = coords.copy()
    out[:, :2] = (out[:, :2] - center_xy) @ R.T + center_xy
    return out


def center_xy(coords, lattice):
    out = coords.copy()
    Lx = np.linalg.norm(lattice.matrix[0])
    Ly = np.linalg.norm(lattice.matrix[1])
    out[:, 0] += Lx / 2.0 - out[:, 0].mean()
    out[:, 1] += Ly / 2.0 - out[:, 1].mean()
    return out


def align_mos2_zigzag_to_x(epilayer_uc):
    a1 = epilayer_uc.lattice.matrix[0].copy()
    a1[2] = 0.0
    a1 /= np.linalg.norm(a1)
    return -np.degrees(np.arctan2(a1[1], a1[0]))


def make_shift_cart_from_frac(dx_frac, dy_frac, lattice):
    return dx_frac * lattice.matrix[0] + dy_frac * lattice.matrix[1]


def inside_triangle_mask(xy, v1, v2, v3):
    def sign(p, a, b):
        return (p[:, 0]-b[0])*(a[1]-b[1]) - (a[0]-b[0])*(p[:, 1]-b[1])
    b1 = sign(xy, v1, v2) < 0
    b2 = sign(xy, v2, v3) < 0
    b3 = sign(xy, v3, v1) < 0
    return (b1 == b2) & (b2 == b3)


def cut_equilateral_triangle(coords, species, side, center):
    h = np.sqrt(3) * side / 2
    cx, cy = center
    v1 = np.array([cx, cy + 2*h/3])
    v2 = np.array([cx - side/2, cy - h/3])
    v3 = np.array([cx + side/2, cy - h/3])
    mask = inside_triangle_mask(coords[:, :2], v1, v2, v3)
    return [species[i] for i in np.where(mask)[0]], coords[mask]


def write_extended_xyz(structures, metadata, filename):
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

    substrate   = Structure.from_file(SUBSTRATE_POSCAR)
    epilayer_uc = Structure.from_file(EPILAYER_POSCAR)

    lattice = substrate.lattice
    if ADD_VACUUM_Z:
        lattice = add_vacuum_to_lattice(lattice, VACUUM_THICKNESS)

    substrate_species = list(substrate.species)
    substrate_coords  = substrate.cart_coords
    z_sub_top = substrate_coords[:, 2].max()

    epilayer_big = epilayer_uc * EPILAYER_SUPERCELL
    epi_species  = list(epilayer_big.species)
    epi_coords_base = epilayer_big.cart_coords.copy()
    epi_coords_base[:, 2] -= epi_coords_base[:, 2].min()
    epi_coords_base = center_xy(epi_coords_base, lattice)

    phi_align = align_mos2_zigzag_to_x(epilayer_uc)

    rotations = (
        ZZ_ONLY_ROTATIONS if ROTATION_MODE.upper() == "ZZ_ONLY"
        else GENERAL_ROTATIONS
    )

    Lx = np.linalg.norm(lattice.matrix[0])
    Ly = np.linalg.norm(lattice.matrix[1])
    center = np.array([Lx/2, Ly/2])

    a1 = epilayer_uc.lattice.matrix[0].copy()
    a1[2] = 0.0
    half_a1_xy = 0.5 * a1[:2]

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    all_structures = []
    all_metadata   = []

    for z_mean in Z_MEAN_LIST:
        for dx_frac, dy_frac in DISPLACEMENTS_FRAC:
            shift = make_shift_cart_from_frac(dx_frac, dy_frac, lattice)
            for theta in rotations:

                coords = epi_coords_base.copy()
                coords[:, :2] += shift[:2]
                coords = rotate_about_z(coords, phi_align, center)
                coords = rotate_about_z(coords, theta, center)

                if ZZ_TERMINATION.upper() == "MO":
                    coords[:, :2] += half_a1_xy

                flake_species, flake_coords = cut_equilateral_triangle(
                    coords, epi_species, TRIANGLE_SIDE, center
                )

                flake_coords[:, 2] += z_sub_top + z_mean

                struct = Structure(
                    lattice=lattice,
                    species=substrate_species + flake_species,
                    coords=np.vstack([substrate_coords, flake_coords]),
                    coords_are_cartesian=True
                )

                all_structures.append(struct)
                all_metadata.append({
                    "z_mean": z_mean,
                    "dx_frac": dx_frac,
                    "dy_frac": dy_frac,
                    "theta": theta,
                    "termination": ZZ_TERMINATION,
                    "shape": FLAKE_SHAPE,
                    "flake_atoms": len(flake_species),
                    "uid": f"Z{z_mean}_dx{dx_frac}_dy{dy_frac}_th{theta}_term{ZZ_TERMINATION}"
                })

    if WRITE_EXTXYZ:
        write_extended_xyz(
            all_structures,
            all_metadata,
            os.path.join(OUTPUT_BASE_DIR, EXTXYZ_FILENAME)
        )

    print("\n✅ Finite triangular ZZ flake dataset generated successfully.")


if __name__ == "__main__":
    main()
