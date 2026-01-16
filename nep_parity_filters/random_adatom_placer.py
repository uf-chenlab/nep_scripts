#!/usr/bin/env python3
"""
Generate slab + epilayer POSCARs for MLIP / DFT data generation.

All user-editable settings are in the CONFIG section below.
"""

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic
import os
from datetime import datetime


# ============================================================
# ====================== USER CONFIG =========================
# ============================================================

SUBSTRATE_POSCAR = "POSCAR_4X4X1"
BASE_DIR         = "all_atoms_jiggle_sub_0005"

N_SAMPLES = 50          # per Z offset
VACUUM_Z  = 2.0          # Å

EPILAYER_SPECIES = {
    "Mo": 1,
    "S":  2,
}

Z_OFFSETS = [3, 3.5, 3.5, 4.5, 5.0, 5.5, 6.0]
Z_NOISE   = 0.2           # ± Å

MIN_ADATOM_DISTANCE = 2.2
MAX_ATTEMPTS        = 500000

BASE_RANDOM_SEED = 0     # set to None for non-reproducible runs

# --- Substrate jiggle (NEW) ---
JIGGLE_SUBSTRATE = True
SUBSTRATE_JIGGLE_AMPLITUDE = 0.005  # Å

# --- XYZ trajectory output ---
WRITE_XYZ    = True
XYZ_FILENAME = "all_structures.xyz"

# ============================================================
# ==================== END USER CONFIG =======================
# ============================================================


def add_vacuum_z_only(atoms, vacuum_z):
    atoms = atoms.copy()
    cell = atoms.cell.array
    cell[2, 2] += vacuum_z
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc((True, True, False))
    return atoms


def jiggle_atoms(atoms, amplitude, rng):
    """
    Apply uniform random Cartesian jiggle to atoms.
    """
    atoms = atoms.copy()
    disp = rng.uniform(-amplitude, amplitude, size=atoms.positions.shape)
    atoms.positions += disp
    return atoms


def topmost_z(atoms):
    return np.max(atoms.positions[:, 2])


def random_xy_position(cell, rng):
    u, v = rng.random(), rng.random()
    return u * cell[0] + v * cell[1]


def mic_distance_xy(cell, dr):
    dmic, _ = find_mic(dr, cell=cell, pbc=(True, True, False))
    return np.linalg.norm(dmic)


def place_epilayer_atoms(substrate, species_list, z_offset, rng):
    cell = substrate.cell.array
    z_top = topmost_z(substrate)

    positions = []
    symbols   = []

    for symbol in species_list:
        for _ in range(MAX_ATTEMPTS):
            xy = random_xy_position(cell, rng)
            z  = z_top + z_offset + rng.uniform(-Z_NOISE, Z_NOISE)
            pos = np.array([xy[0], xy[1], z])

            if all(mic_distance_xy(cell, pos - p) >= MIN_ADATOM_DISTANCE for p in positions):
                positions.append(pos)
                symbols.append(symbol)
                break
        else:
            raise RuntimeError(
                f"Failed to place adatom {symbol} at z_offset={z_offset} Å"
            )

    return Atoms(
        symbols=symbols,
        positions=np.array(positions),
        cell=substrate.cell,
        pbc=substrate.pbc,
    )


def expand_species_dict(species_dict):
    expanded = []
    for s, n in species_dict.items():
        expanded.extend([s] * n)
    return expanded


def write_summary(base_dir, folder_list):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = os.path.join(base_dir, f"summary_{timestamp}.txt")

    with open(summary_path, "w") as f:
        f.write("Epilayer Dataset Generation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Substrate POSCAR: {SUBSTRATE_POSCAR}\n")
        f.write(f"Vacuum Z: {VACUUM_Z} Å\n\n")

        f.write("Epilayer species:\n")
        for s, n in EPILAYER_SPECIES.items():
            f.write(f"  {s}: {n}\n")

        f.write("\nZ offsets (Å):\n")
        for z in Z_OFFSETS:
            f.write(f"  {z}\n")

        f.write(f"\nSamples per Z offset: {N_SAMPLES}\n")
        f.write(f"Z noise: ±{Z_NOISE} Å\n")
        f.write(f"Min adatom distance: {MIN_ADATOM_DISTANCE} Å\n")
        f.write(f"Base random seed: {BASE_RANDOM_SEED}\n")
        f.write(f"Substrate jiggle: {JIGGLE_SUBSTRATE}\n")
        if JIGGLE_SUBSTRATE:
            f.write(f"Substrate jiggle amplitude: {SUBSTRATE_JIGGLE_AMPLITUDE} Å\n")

        f.write("\nGenerated folders:\n")
        for folder in folder_list:
            f.write(f"  {folder}\n")

    print(f"\nSummary written to: {summary_path}\n")


def main():
    substrate_ref = read(SUBSTRATE_POSCAR, format="vasp")
    substrate_ref.set_pbc((True, True, False))
    substrate_ref = add_vacuum_z_only(substrate_ref, VACUUM_Z)

    species_list = expand_species_dict(EPILAYER_SPECIES)

    os.makedirs(BASE_DIR, exist_ok=True)
    created_folders = []
    all_structures = []

    for z_index, z_offset in enumerate(Z_OFFSETS):
        z_dir = os.path.join(BASE_DIR, f"z_{z_offset:.1f}")
        os.makedirs(z_dir, exist_ok=True)

        for i in range(N_SAMPLES):
            seed = None if BASE_RANDOM_SEED is None else (
                BASE_RANDOM_SEED + z_index * N_SAMPLES + i
            )
            rng = np.random.default_rng(seed)

            # --- optionally jiggle substrate ---
            if JIGGLE_SUBSTRATE:
                substrate = jiggle_atoms(
                    substrate_ref,
                    SUBSTRATE_JIGGLE_AMPLITUDE,
                    rng
                )
            else:
                substrate = substrate_ref.copy()

            epilayer = place_epilayer_atoms(
                substrate, species_list, z_offset, rng
            )
            combined = substrate + epilayer

            all_structures.append(combined.copy())

            sample_dir = os.path.join(z_dir, f"{i:05d}")
            os.makedirs(sample_dir, exist_ok=True)
            created_folders.append(sample_dir)

            write(
                os.path.join(sample_dir, "POSCAR"),
                combined,
                format="vasp",
                vasp5=True,
                direct=False,
                sort=False,
            )

            print(
                f"[z={z_offset:.1f} Å | {i+1}/{N_SAMPLES}] "
                f"seed={seed} → {sample_dir}/POSCAR"
            )

    if WRITE_XYZ:
        xyz_path = os.path.join(BASE_DIR, XYZ_FILENAME)
        write(xyz_path, all_structures, format="extxyz")
        print(f"\n🧩 Extended XYZ written to: {xyz_path}")

    write_summary(BASE_DIR, created_folders)


if __name__ == "__main__":
    main()
