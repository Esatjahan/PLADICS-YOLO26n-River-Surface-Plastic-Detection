import os

# 🔁 CHANGE THIS ROOT PATH
dataset_path = r"D:\PLADICS\merged_dataset_v2\labels"

splits = ["train", "val", "test"]

def fix_labels(file_path):
    fixed_lines = []
    changed = False
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        
        if len(parts) != 5:
            continue
        
        cls, x, y, w, h = parts
        x, y, w, h = float(x), float(y), float(w), float(h)
        
        original = (x, y, w, h)
        
        # clamp values to [0,1]
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        if original != (x, y, w, h):
            changed = True
        
        fixed_lines.append(f"{cls} {x} {y} {w} {h}\n")
    
    if changed:
        with open(file_path, "w") as f:
            f.writelines(fixed_lines)
        return True
    
    return False


total_fixed = 0

for split in splits:
    label_dir = os.path.join(dataset_path, split)
    
    if not os.path.exists(label_dir):
        continue
    
    print(f"\nChecking {split} labels...")
    
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(label_dir, file)
            
            if fix_labels(file_path):
                total_fixed += 1

print(f"\nDone ✅ Total files fixed: {total_fixed}")