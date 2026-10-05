import os

dataset_path = r"D:\PLADICS\merged_dataset_v2\labels"
splits = ["train", "val", "test"]

def process_file(file_path):
    fixed_lines = []
    changed = False
    removed = 0

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # must be exactly 5 items: class x y w h
        if len(parts) != 5:
            changed = True
            removed += 1
            continue

        try:
            cls = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            changed = True
            removed += 1
            continue

        # reject negative class
        if cls < 0:
            changed = True
            removed += 1
            continue

        # clamp coords
        new_x = max(0.0, min(1.0, x))
        new_y = max(0.0, min(1.0, y))
        new_w = max(0.0, min(1.0, w))
        new_h = max(0.0, min(1.0, h))

        # reject zero-size boxes
        if new_w <= 0.0 or new_h <= 0.0:
            changed = True
            removed += 1
            continue

        if (cls, x, y, w, h) != (cls, new_x, new_y, new_w, new_h):
            changed = True

        fixed_lines.append(f"{cls} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)

    return changed, removed

total_changed = 0
total_removed = 0

for split in splits:
    label_dir = os.path.join(dataset_path, split)
    if not os.path.exists(label_dir):
        continue

    print(f"\nChecking {split} labels...")
    for name in os.listdir(label_dir):
        if not name.endswith(".txt"):
            continue
        fp = os.path.join(label_dir, name)
        changed, removed = process_file(fp)
        if changed:
            total_changed += 1
        total_removed += removed

print(f"\nDone. Files changed: {total_changed}")
print(f"Invalid label lines removed: {total_removed}")