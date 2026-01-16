import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar

# ============================================================
# ===================== USER PARAMETERS ======================
# ============================================================

SUBSTRATE_POSCAR = "SUPERCELL_221.vasp"
EPILAYER_POSCAR  = "POSCAR_epilayer"

OUTPUT_BASE_DIR = "displaced_interfaces12"

Z_MEAN_LIST = [3.0]

# ------------------------------------------------------------
# In-plane displacements in SUBSTRATE FRACTIONAL coordinates
# (cleanest and fully PBC-consistent)
#
# Example: 4x4 grid over the substrate cell
# ------------------------------------------------------------
N_DISP = 12
DISPLACEMENTS_FRAC = [
    (i / N_DISP, j / N_DISP)
    for i in range(N_DISP)
    for j in range(N_DISP)
]

ADD_VACUUM_Z     = True
VACUUM_THICKNESS = 1.0

WRITE_EXTXYZ = True
EXTXYZ_FILENAME = "all_structures.xyz"

# ============================================================


# ============================================================
# ===================== HELPER FUNCTIONS =====================
# ============================================================

def add_vacuum_to_lattice(lattice, vacuum):
    a, b, c = lattice.matrix
    c_new = c + np.array([0.0, 0.0, vacuum])
    return Lattice([a, b, c_new])


def make_epilayer_supercell(epi_struct, substrate_lattice):
    """
    Build a periodic epilayer supercell that fits inside substrate.
    Orientation preserved, no strain.
    """
    a_epi, b_epi = epi_struct.lattice.matrix[:2]
    a_sub, b_sub = substrate_lattice.matrix[:2]

    Lx_sub = np.linalg.norm(a_sub)
    Ly_sub = np.linalg.norm(b_sub)

    Lx_epi = np.linalg.norm(a_epi)
    Ly_epi = np.linalg.norm(b_epi)

    Nx = int(np.floor(Lx_sub / Lx_epi))
    Ny = int(np.floor(Ly_sub / Ly_epi))

    if Nx < 1 or Ny < 1:
        raise ValueError("Substrate too small for epilayer.")

    print(f"▶ Epilayer supercell size: {Nx} × {Ny} × 1")

    return epi_struct * (Nx, Ny, 1)


def center_epilayer_xy(coords, substrate_lattice):
    """
    Center epilayer supercell inside substrate cell.
    """
    a_sub, b_sub = substrate_lattice.matrix[:2]
    Lx = np.linalg.norm(a_sub)
    Ly = np.linalg.norm(b_sub)

    coords[:, 0] += Lx / 2 - coords[:, 0].mean()
    coords[:, 1] += Ly / 2 - coords[:, 1].mean()

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
    epilayer_uc = Structure.from_file(EPILAYER_POSCAR)

    lattice = substrate.lattice
    if ADD_VACUUM_Z:
        lattice = add_vacuum_to_lattice(lattice, VACUUM_THICKNESS)

    substrate_species = list(substrate.species)
    substrate_coords = substrate.cart_coords
    z_sub_top = substrate_coords[:, 2].max()

    # --------------------------------------------------------
    # Build periodic epilayer
    # --------------------------------------------------------
    epilayer_sc = make_epilayer_supercell(epilayer_uc, lattice)
    epi_species = list(epilayer_sc.species)
    epi_coords_base = epilayer_sc.cart_coords.copy()

    # Normalize Z and center
    epi_coords_base[:, 2] -= epi_coords_base[:, 2].min()
    epi_coords_base = center_epilayer_xy(epi_coords_base, lattice)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    all_structures = []

    for z_mean in Z_MEAN_LIST:
        z_dir = os.path.join(OUTPUT_BASE_DIR, f"Z_{z_mean:.2f}")
        os.makedirs(z_dir, exist_ok=True)

        for (dx_frac, dy_frac) in DISPLACEMENTS_FRAC:

            disp_dir = os.path.join(
                z_dir, f"dx_{dx_frac:.2f}_dy_{dy_frac:.2f}"
            )
            os.makedirs(disp_dir, exist_ok=True)

            coords = epi_coords_base.copy()

            # ------------------------------------------------
            # Apply in-plane displacement (fractional → cart)
            # ------------------------------------------------
            shift_cart = (
                dx_frac * lattice.matrix[0] +
                dy_frac * lattice.matrix[1]
            )
            coords[:, 0] += shift_cart[0]
            coords[:, 1] += shift_cart[1]

            # Place epilayer above substrate
            coords[:, 2] += z_sub_top + z_mean

            # Combine system
            all_species = substrate_species + epi_species
            all_coords  = np.vstack([substrate_coords, coords])

            struct = Structure(
                lattice=lattice,
                species=all_species,
                coords=all_coords,
                coords_are_cartesian=True
            )

            # Wrap once
            struct = Structure(
                lattice=struct.lattice,
                species=struct.species,
                coords=struct.frac_coords % 1.0,
                coords_are_cartesian=False
            )

            Poscar(struct).write_file(os.path.join(disp_dir, "POSCAR"))
            all_structures.append(struct)

            print(
                f"[Z={z_mean:.2f} | dx={dx_frac:.2f}, dy={dy_frac:.2f}] → {disp_dir}"
            )

    if WRITE_EXTXYZ:
        write_extended_xyz(
            all_structures,
            os.path.join(OUTPUT_BASE_DIR, EXTXYZ_FILENAME)
        )
        print(f"\n🧩 Wrote extended XYZ: {EXTXYZ_FILENAME}")

    print("\n✅ Displaced MoS₂/substrate interfaces generated successfully.")


if __name__ == "__main__":
    main()
