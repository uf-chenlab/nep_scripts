import random
from pathlib import Path

# ================= USER SETTINGS =================
xyz_files  = [
    "test_filtered.xyz",
   # "file2.xyz",
]

test_ratio = 0.075  # 10% test
seed       = 42
# =================================================

random.seed(seed)

def read_xyz_frames(path):
    frames = []
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        n = int(lines[i])
        frames.append("".join(lines[i:i+n+2]))
        i += n + 2

    return frames

# Load all frames
all_frames = []
for xyz in xyz_files:
    all_frames.extend(read_xyz_frames(xyz))

# Shuffle
random.shuffle(all_frames)

# Split
n_test = int(len(all_frames) * test_ratio)
test_frames  = all_frames[:n_test]
train_frames = all_frames[n_test:]

# Write output
with open("train.xyz", "w") as f:
    f.writelines(train_frames)

with open("test.xyz", "w") as f:
    f.writelines(test_frames)

print(f"Total frames : {len(all_frames)}")
print(f"Train frames : {len(train_frames)}")
print(f"Test frames  : {len(test_frames)}")
