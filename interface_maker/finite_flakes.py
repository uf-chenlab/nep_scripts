import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar

# ============================================================
# ===================== USER PARAMETERS ======================
# ============================================================

SUBSTRATE_POSCAR = "POSCAR_4X4X1"
EPILAYER_POSCAR  = "POSCAR_epilayer"

OUTPUT_BASE_DIR = "finite_domains"

Z_MEAN_LIST = [3.1]

# Build a reasonably large MoS2 sheet first
EPILAYER_SUPERCELL = (6, 6, 1)

# Final finite flake size (Å)
FLAKE_SIZE_X = 6.0
FLAKE_SIZE_Y = 6.0

ADD_VACUUM_Z     = True
VACUUM_THICKNESS = 2.0   # give lateral islands some breathing room

WRITE_EXTXYZ = True
EXTXYZ_FILENAME = "all_structures.xyz"

# ============================================================


def add_vacuum_to_lattice(lattice, vacuum):
    a, b, c = lattice.matrix
    return Lattice([a, b, c + np.array([0.0, 0.0, vacuum])])


def center_xy(coords, lattice):
    Lx = np.linalg.norm(lattice.matrix[0])
    Ly = np.linalg.norm(lattice.matrix[1])

    coords[:, 0] += Lx / 2 - coords[:, 0].mean()
    coords[:, 1] += Ly / 2 - coords[:, 1].mean()
    return coords


def cut_rectangular_flake(coords, species, size_x, size_y):
    """
    Keep only atoms inside a centered rectangle of size_x x size_y.
    """
    x0, y0 = coords[:, 0].mean(), coords[:, 1].mean()

    mask = (
        (np.abs(coords[:, 0] - x0) <= size_x / 2) &
        (np.abs(coords[:, 1] - y0) <= size_y / 2)
    )

    new_coords = coords[mask]
    new_species = [species[i] for i in np.where(mask)[0]]

    return new_species, new_coords


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


def main():

    substrate = Structure.from_file(SUBSTRATE_POSCAR)
    epilayer_uc = Structure.from_file(EPILAYER_POSCAR)

    lattice = substrate.lattice
    if ADD_VACUUM_Z:
        lattice = add_vacuum_to_lattice(lattice, VACUUM_THICKNESS)

    substrate_species = list(substrate.species)
    substrate_coords = substrate.cart_coords
    z_sub_top = substrate_coords[:, 2].max()

    # Build large MoS2 sheet
    epilayer_big = epilayer_uc * EPILAYER_SUPERCELL
    epi_species = list(epilayer_big.species)
    epi_coords = epilayer_big.cart_coords.copy()

    # Normalize and center
    epi_coords[:, 2] -= epi_coords[:, 2].min()
    epi_coords = center_xy(epi_coords, lattice)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    all_structures = []

    for z_mean in Z_MEAN_LIST:

        z_dir = os.path.join(OUTPUT_BASE_DIR, f"Z_{z_mean:.2f}")
        os.makedirs(z_dir, exist_ok=True)

        # Cut finite flake
        flake_species, flake_coords = cut_rectangular_flake(
            epi_coords, epi_species,
            FLAKE_SIZE_X, FLAKE_SIZE_Y
        )

        flake_coords[:, 2] += z_sub_top + z_mean

        all_species = substrate_species + flake_species
        all_coords  = np.vstack([substrate_coords, flake_coords])

        struct = Structure(
            lattice=lattice,
            species=all_species,
            coords=all_coords,
            coords_are_cartesian=True
        )

        # IMPORTANT:
        # Only wrap substrate atoms effectively; epilayer is finite
        struct = Structure(
            lattice=struct.lattice,
            species=struct.species,
            coords=struct.frac_coords,
            coords_are_cartesian=False
        )

        out_dir = os.path.join(z_dir, "flake")
        os.makedirs(out_dir, exist_ok=True)
        Poscar(struct).write_file(os.path.join(out_dir, "POSCAR"))

        all_structures.append(struct)
        print(f"[Z={z_mean:.2f}] finite flake written → {out_dir}")

    if WRITE_EXTXYZ:
        write_extended_xyz(
            all_structures,
            os.path.join(OUTPUT_BASE_DIR, EXTXYZ_FILENAME)
        )

    print("\n✅ Finite epilayer domain generation complete.")


if __name__ == "__main__":
    main()
