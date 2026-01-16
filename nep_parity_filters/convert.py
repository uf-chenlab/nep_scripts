#!/usr/bin/env python3
"""
Remap LAMMPS atom types and rewrite Masses block
to enforce a fixed NEP atom-type ordering.

Final enforced order:
1 = Mo
2 = O
3 = Al
4 = S
"""

# ============================================================
# ====================== USER SETTINGS =======================
# ============================================================

INPUT_DATA  = "Sapphire_sub_og.data"
OUTPUT_DATA = "Sapphire_sub_remapped.data"

# Old type → New type mapping
TYPE_MAP = {
    1: 3,  # Al → Al
    2: 2,  # O  → O
}

# Enforced NEP type ordering
NEW_MASSES = {
    1: ("Mo", 95.95),
    2: ("O", 15.9994),
    3: ("Al", 26.981538),
    4: ("S", 32.06),
}

# ============================================================


def remap_lammps_data(infile, outfile):
    with open(infile, "r") as f:
        lines = f.readlines()

    out_lines = []
    i = 0
    n = len(lines)

    allowed_types = set(NEW_MASSES.keys())

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # ---------- FORCE atom types count ----------
        if stripped.endswith("atom types"):
            out_lines.append(f"{len(NEW_MASSES)} atom types\n")
            i += 1
            continue

        # ---------- Masses (DELETE EVERYTHING UNTIL Atoms) ----------
        if stripped == "Masses":
            out_lines.append("Masses\n\n")

            # Write enforced Masses block
            for t in sorted(NEW_MASSES):
                name, mass = NEW_MASSES[t]
                out_lines.append(f"{t} {mass}  # {name}\n")

            out_lines.append("\n")

            # Skip EVERYTHING until "Atoms"
            i += 1
            while i < n and not lines[i].strip().startswith("Atoms"):
                i += 1

            continue

        # ---------- Atoms ----------
        if stripped.startswith("Atoms"):
            out_lines.append(line)
            out_lines.append("\n")
            i += 2  # skip blank line

            while i < n and lines[i].strip():
                parts = lines[i].split()

                atom_id = parts[0]
                old_type = int(parts[1])
                new_type = TYPE_MAP.get(old_type, old_type)

                if new_type not in allowed_types:
                    raise ValueError(
                        f"❌ Atom {atom_id} mapped to invalid type {new_type}. "
                        f"Allowed types: {sorted(allowed_types)}"
                    )

                parts[1] = str(new_type)
                out_lines.append(" ".join(parts) + "\n")
                i += 1

            out_lines.append("\n")
            continue

        # ---------- Default ----------
        out_lines.append(line)
        i += 1

    with open(outfile, "w") as f:
        f.writelines(out_lines)

    print(f"✅ Successfully wrote remapped file: {outfile}")


if __name__ == "__main__":
    remap_lammps_data(INPUT_DATA, OUTPUT_DATA)
