#!/usr/bin/env python3

import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar

# ============================================================
# ===================== USER PARAMETERS ======================
# ============================================================

SUBSTRATE_POSCAR = "SUPERCELL_331.vasp"
EPILAYER_POSCAR  = "POSCAR_epilayer"

OUTPUT_BASE_DIR = "production_run_unit_cell_epilayers"

Z_MEAN_LIST = [3.05]
N_STRUCTURES_PER_Z = 200
Z_NOISE = 0.125

# --- Epilayer jiggle ---
JIGGLE_AMPLITUDE = 0.01  # Å

# --- Substrate jiggle ---
JIGGLE_SUBSTRATE = False
SUBSTRATE_JIGGLE_AMPLITUDE = 0.00  # Å

ROTATE_EPILAYER = True
ROTATION_RANGE  = (0.0, 360.0)

ADD_VACUUM_Z     = True
VACUUM_THICKNESS = 2.0

WRITE_EXTXYZ = True
EXTXYZ_FILENAME = "all_structures.xyz"

RANDOM_SEED = 42

# ============================================================

np.random.seed(RANDOM_SEED)

# ============================================================
# ===================== HELPER FUNCTIONS =====================
# ============================================================

def rotate_z(coords, theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    return coords @ R.T


def jiggle_coords(coords, amplitude):
    return coords + np.random.uniform(
        -amplitude, amplitude, size=coords.shape
    )


def prepare_epilayer_reference(epi_struct):
    coords = epi_struct.cart_coords.copy()
    coords[:, :2] -= coords[:, :2].mean(axis=0)
    coords[:, 2]  -= coords[:, 2].min()
    return coords


def get_xy_extent(coords):
    dx = coords[:, 0].max() - coords[:, 0].min()
    dy = coords[:, 1].max() - coords[:, 1].min()
    return dx, dy


def random_place_xy_that_fits(coords, lattice):
    a, b = lattice.matrix[0], lattice.matrix[1]
    Lx = np.linalg.norm(a)
    Ly = np.linalg.norm(b)

    dx, dy = get_xy_extent(coords)
    if dx > Lx or dy > Ly:
        raise ValueError("Epilayer footprint larger than substrate cell.")

    x0 = np.random.uniform(0.0, Lx - dx)
    y0 = np.random.uniform(0.0, Ly - dy)

    coords[:, 0] += x0 - coords[:, 0].min()
    coords[:, 1] += y0 - coords[:, 1].min()
    return coords


def add_vacuum_to_lattice(lattice, vacuum):
    a, b, c = lattice.matrix
    c_new = c + np.array([0.0, 0.0, vacuum])
    return Lattice([a, b, c_new])


def randomize_epilayer(
    epi_coords,
    lattice,
    z_sub_top,
    z_mean,
    z_noise,
    jiggle
):
    coords = epi_coords.copy()

    if ROTATE_EPILAYER:
        theta = np.deg2rad(np.random.uniform(*ROTATION_RANGE))
        coords = rotate_z(coords, theta)

    coords = random_place_xy_that_fits(coords, lattice)

    coords[:, 2] += z_sub_top + z_mean + np.random.uniform(-z_noise, z_noise)
    coords = jiggle_coords(coords, jiggle)

    return coords


def write_extended_xyz(structures, filename):
    with open(filename, "w") as f:
        for i, s in enumerate(structures):
            lat = s.lattice.matrix.flatten()
            f.write(f"{len(s)}\n")
            f.write(
                'Lattice="{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}" '
                'Properties=species:S:1:pos:R:3 '
                f'frame={i}\n'.format(*lat)
            )
            for site in s:
                x, y, z = site.coords
                f.write(f"{site.species_string} {x:.6f} {y:.6f} {z:.6f}\n")


# ============================================================
# ============================ MAIN ==========================
# ============================================================

def main():

    substrate = Structure.from_file(SUBSTRATE_POSCAR)
    epilayer  = Structure.from_file(EPILAYER_POSCAR)

    lattice = substrate.lattice
    if ADD_VACUUM_Z:
        lattice = add_vacuum_to_lattice(lattice, VACUUM_THICKNESS)

    substrate_species = substrate.species
    substrate_coords_ref = substrate.cart_coords
    z_sub_top = substrate_coords_ref[:, 2].max()

    epi_species    = epilayer.species
    epi_ref_coords = prepare_epilayer_reference(epilayer)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    all_structures = []

    for z_mean in Z_MEAN_LIST:
        z_dir = os.path.join(OUTPUT_BASE_DIR, f"Z_{z_mean:.2f}")
        os.makedirs(z_dir, exist_ok=True)

        for i in range(N_STRUCTURES_PER_Z):

            epi_coords_rand = randomize_epilayer(
                epi_coords=epi_ref_coords,
                lattice=lattice,
                z_sub_top=z_sub_top,
                z_mean=z_mean,
                z_noise=Z_NOISE,
                jiggle=JIGGLE_AMPLITUDE
            )

            if JIGGLE_SUBSTRATE:
                substrate_coords = jiggle_coords(
                    substrate_coords_ref,
                    SUBSTRATE_JIGGLE_AMPLITUDE
                )
            else:
                substrate_coords = substrate_coords_ref

            all_species = list(substrate_species) + list(epi_species)
            all_coords  = np.vstack([substrate_coords, epi_coords_rand])

            struct = Structure(
                lattice=lattice,
                species=all_species,
                coords=all_coords,
                coords_are_cartesian=True
            )

            frac = struct.frac_coords % 1.0
            struct = Structure(
                lattice=struct.lattice,
                species=struct.species,
                coords=frac,
                coords_are_cartesian=False
            )

            out_dir = os.path.join(z_dir, f"{i:05d}")
            os.makedirs(out_dir, exist_ok=True)

            Poscar(struct).write_file(os.path.join(out_dir, "POSCAR"))
            all_structures.append(struct)

            print(f"[Z={z_mean:.2f} | {i+1}/{N_STRUCTURES_PER_Z}] {out_dir}")

    if WRITE_EXTXYZ:
        write_extended_xyz(
            all_structures,
            os.path.join(OUTPUT_BASE_DIR, EXTXYZ_FILENAME)
        )
        print(f"\n🧩 Wrote extended XYZ with native cells: {EXTXYZ_FILENAME}")

    print("\n✅ All randomized interface structures generated successfully.")


if __name__ == "__main__":
    main()
