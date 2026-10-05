import os
import shutil

# =========================
# PATH CONFIG
# =========================
RISID = r"D:\PLADICS\datasets_clean\RiSID_single_clean"
RIVERINE = r"D:\PLADICS\datasets_clean\Riverine_single_clean"

OUTPUT = r"D:\PLADICS\datasets_clean\merged_clean_dataset"

# =========================
# CREATE OUTPUT FOLDERS
# =========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, "labels", split), exist_ok=True)

# =========================
# SAFE COPY FUNCTION
# =========================
def copy_dataset(src_root, prefix):
    for split in ["train", "val", "test"]:

        img_src = os.path.join(src_root, "images", split)
        lbl_src = os.path.join(src_root, "labels", split)

        img_dst = os.path.join(OUTPUT, "images", split)
        lbl_dst = os.path.join(OUTPUT, "labels", split)

        if not os.path.exists(img_src):
            continue

        for file in os.listdir(img_src):

            src_img_path = os.path.join(img_src, file)

            # 🔥 IMPORTANT: add prefix to avoid name collision
            new_name = f"{prefix}_{file}"

            dst_img_path = os.path.join(img_dst, new_name)

            shutil.copy2(src_img_path, dst_img_path)

            # label file
            label_name = os.path.splitext(file)[0] + ".txt"
            src_lbl_path = os.path.join(lbl_src, label_name)

            if os.path.exists(src_lbl_path):
                dst_lbl_path = os.path.join(
                    lbl_dst,
                    f"{prefix}_{label_name}"
                )
                shutil.copy2(src_lbl_path, dst_lbl_path)


# =========================
# RUN MERGE
# =========================
print("🔄 Merging RiSID...")
copy_dataset(RISID, "risid")

print("🔄 Merging Riverine...")
copy_dataset(RIVERINE, "riverine")

print("\n✅ MERGE COMPLETE")
print("Output folder:", OUTPUT)