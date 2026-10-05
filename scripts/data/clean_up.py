import os
import glob

label_dirs = [
    'D:/PLADICS/merged_dataset_crossval/labels/train',
    'D:/PLADICS/merged_dataset_crossval/labels/val'
]

fixed_count = 0
removed_count = 0

for label_dir in label_dirs:
    for txt_file in glob.glob(os.path.join(label_dir, '*.txt')):
        cleaned_lines = []
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            
            
            if len(parts) < 5:
                removed_count += 1
                continue
            
            try:
                cls = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                
            
                coords_fixed = [max(0.0, min(1.0, c)) for c in coords]
                
                
                if cls < 0:
                    removed_count += 1
                    continue
                
                
                if coords_fixed[2] <= 0 or coords_fixed[3] <= 0:
                    removed_count += 1
                    continue
                
                cleaned_lines.append(
                    f"{cls} " + " ".join(f"{c:.6f}" for c in coords_fixed) + "\n"
                )
                fixed_count += 1
                
            except ValueError:
                removed_count += 1
                continue
        
        with open(txt_file, 'w') as f:
            f.writelines(cleaned_lines)

print(f" Fixed labels: {fixed_count}")
print(f" Removed corrupt lines: {removed_count}")