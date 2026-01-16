#!/usr/bin/env python3
import os
import shutil
import sys

BASE_INPUT_DIR = "/blue/ypchen/emir.bilgili/Mo-S-Al-O/epilayer_on_substrate_random/much_larger_cell/vasp_inputs"
INCAR_SRC      = os.path.join(BASE_INPUT_DIR, "INCAR")
KPOINTS_SRC    = os.path.join(BASE_INPUT_DIR, "KPOINTS")

def read_species_from_poscar(poscar_path):
    with open(poscar_path, "r") as f:
        lines = f.readlines()
    # VASP 5+ format: line 6 has species
    species = lines[5].split()
    return species

def build_potcar(species, target_dir):
    potcar_path = os.path.join(target_dir, "POTCAR")
    with open(potcar_path, "wb") as potcar_out:
        for elem in species:
            potcar_elem = os.path.join(BASE_INPUT_DIR, f"{elem}_POTCAR")
            if not os.path.isfile(potcar_elem):
                raise FileNotFoundError(f"Missing POTCAR for element: {elem}")
            with open(potcar_elem, "rb") as f:
                potcar_out.write(f.read())

def process_directory(dirpath):
    poscar = os.path.join(dirpath, "POSCAR")
    if not os.path.isfile(poscar):
        return

    print(f"▶ Processing {dirpath}")

    # Copy INCAR and KPOINTS
    shutil.copy(INCAR_SRC, os.path.join(dirpath, "INCAR"))
    shutil.copy(KPOINTS_SRC, os.path.join(dirpath, "KPOINTS"))

    # Build POTCAR
    species = read_species_from_poscar(poscar)
    build_potcar(species, dirpath)

def main(root="."):
    for dirpath, dirnames, filenames in os.walk(root):
        if "POSCAR" in filenames:
            try:
                process_directory(dirpath)
            except Exception as e:
                print(f"❌ Failed in {dirpath}: {e}")

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root)
