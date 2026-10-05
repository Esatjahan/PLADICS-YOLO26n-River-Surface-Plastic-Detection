import os
import shutil

# Dataset paths
train_dir = 'D:/PLADICS/merged_dataset_crossval/images/train'
val_dir = 'D:/PLADICS/merged_dataset_crossval/images/val'
label_dir = 'D:/PLADICS/merged_dataset_crossval/labels'  # Assuming labels are stored here

# Corrupted images list (Add all corrupted image names here)
corrupted_images_list = [
    '20221123193750_001152_7_0.png', 
    '20221123193750_001170_1_0.png',
    '20221123193750_001176_7_0.png',
    '20221123193750_001188_1_0.png',
    '20221123193750_001200_1_0.png',
    '20101031080847_001224_1_0.png',
    '20101101162602_001284_1_0.png',
    '20101101165626_000384_7_0.png',
    '20221007122108_c.flv20221007122108_c_000180_3_0.png',
    '20221007122108_c.flv20221007122108_c_000185_3_0.png',
    '20221007122108_c.flv20221007122108_c_000190_3_0.png',
    '20221007122108_c.flv20221007122108_c_000220_3_0.png',
    '20221007123228_c.flv20221007123228_c_002550_6_0.png',
    '20221007123228_c.flv20221007123228_c_002565_1_0.png',
    '20221007123228_c.flv20221007123228_c_002570_6_0.png',
    '20221007123228_c.flv20221007123228_c_002580_6_0.png',
    '20221007123228_c.flv20221007123228_c_002585_4_0.png',
    '20221007123228_c.flv20221007123228_c_002610_6_0.png',
    '20221007123228_c.flv20221007123228_c_002625_6_0.png',
    '20221007124344_d.flv20221007124344_d_000710_6_0.png',
    '20221007124344_d.flv20221007124344_d_001170_6_0.png',
    '20221007124344_d.flv20221007124344_d_001200_6_0.png',
    '20221007124344_d.flv20221007124344_d_002400_6_0.png',
    '20221007124344_d.flv20221007124344_d_002750_6_0.png',
    '20221007124344_d.flv20221007124344_d_002755_6_0.png',
    '20221007124344_d.flv20221007124344_d_002760_6_0.png',
    '20221007124344_d.flv20221007124344_d_002770_6_0.png',
    '20221007124922_d.flv20221007124922_d_000070_6_0.png',
    '20221007125500_d.flv20221007125500_d_000045_6_0.png',
    '20221007125500_d.flv20221007125500_d_000055_6_0.png',
    '20221007125500_d.flv20221007125500_d_000060_6_0.png',
    '20221007125500_d.flv20221007125500_d_000805_6_0.png',
    '20221007125500_d.flv20221007125500_d_002035_6_0.png',
    '20221007125731_d.flv20221007125731_d_000480_6_0.png',
    '20221007130020.flv20221007130020_002355_5_0.png',
    '20221007131426_d.flv20221007131426_d_002745_6_0.png',
    '20221007134543_c.flv20221007134543_c_001685_5_0.png',
    '20221010073923_c.flv20221010073923_c_001615_7_0.png',
    '20221010113025_c.flv20221010113025_c_002765_6_0.png',
    '20221010135654.flv20221010135654_001030_5_0.png',
    '20221123104249.flv20221123104249_001390_3_0.png',
    '20221123104815.flv20221123104815_001795_6_0.png',
    '20221123104815.flv20221123104815_001810_6_0.png',
    '20221123104815.flv20221123104815_001815_6_0.png',
    '20221123104815.flv20221123104815_001820_6_0.png',
    '20221123104815.flv20221123104815_001825_6_0.png',
    '20221123104815.flv20221123104815_001830_6_0.png',
    '20221123104815.flv20221123104815_001850_1_0.png',
    '20221123104815.flv20221123104815_001860_1_0.png',
    '20221123104815.flv20221123104815_001880_1_0.png',
    '20221123104815.flv20221123104815_001885_4_0.png',
    '20221123104815.flv20221123104815_001890_1_0.png',
    '20221123105353.flv20221123105353_001645_6_0.png',
    '20221123105405.flv20221123105405_002290_6_0.png',
    '20221123105931.flv20221123105931_000820_6_0.png',
    '20221123105931.flv20221123105931_000835_5_0.png',
    '20221123110510.flv20221123110510_000770_1_0.png',
    '20221123112742.flv20221123112742_000865_6_0.png',
    '20221123112742.flv20221123112742_000870_6_0.png',
    '20221123112742.flv20221123112742_000880_6_0.png',
    '20221123113858.flv20221123113858_001610_6_0.png',
    '20221123134835.flv20221123134835_000220_6_0.png',
    '20221123134835.flv20221123134835_000255_6_0.png',
    '20221123134835.flv20221123134835_000275_6_0.png',
    '20221123143153.flv20221123143153_000245_6_0.png',
    '20221123143340.flv20221123143340_002205_5_0.png',
    '2110260600-01_001446_3_1.png',
    '2110260900_000048_6_0.png',
    '2110260900_000060_6_0.png',
    '2110260900_000072_6_0.png',
    '2110260900_000096_6_0.png',
    '2110260900_000144_6_0.png',
    '2110260900_000150_6_0.png',
    '2110260900_000366_5_0.png',
    '2110260930_000186_6_0.png',
    'MAH00261_11CentreTrim_000600_6_0.png',
    'MAH00261_11CentreTrim_000606_6_0.png',
    'MAH00261_11CentreTrim_000612_6_0.png',
    'MAH00261_11CentreTrim_000618_6_0.png',
    'MAH00261_11CentreTrim_000636_6_0.png',
    'MAH00261_11CentreTrim_000642_6_0.png',
    'MAH00261_11CentreTrim_000648_6_0.png',
    'MAH00261_11CentreTrim_000654_6_0.png',
    'MAH00261_11CentreTrim_000660_6_0.png',
    'MAH00261_11CentreTrim_000666_6_0.png',
    'MAH00261_11CentreTrim_000672_6_0.png',
    'MAH00261_11CentreTrim_000678_6_0.png',
    'MAH00261_11CentreTrim_000684_6_0.png',
    'MAH00261_11CentreTrim_000690_6_0.png',
    'MAH00261_11CentreTrim_000696_6_0.png',
    'MAH00261_11CentreTrim_000702_6_0.png',
    'MAH00261_11CentreTrim_000714_6_0.png',
    'MAH00261_11CentreTrim_000720_6_0.png',
    'MAH00261_11CentreTrim_000726_6_0.png',
    'MAH00261_11CentreTrim_000738_6_0.png',
    'wlgcam2_20200216070000_video_000150_6_0.png',
    'wlgcam2_20200216070000_video_000900_1_0.png',
    'wlgcam2_20200216073001_video_000426_1_0.png',
    'wlgcam2_20200216073001_video_000462_1_0.png',
    'wlgcam2_20200216073001_video_000474_1_0.png',
    'wlgcam2_20200216073001_video_000486_1_0.png',
    'wlgcam2_20200901071001_video_000342_7_0.png', 
]

# Remove corrupted images from the train directory
for image in corrupted_images_list:
    try:
        image_path = os.path.join(train_dir, image)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Removed corrupted image: {image}")
    except Exception as e:
        print(f"Error removing image {image}: {e}")

# You can also apply the same logic to val_dir for validation images
for image in corrupted_images_list:
    try:
        image_path = os.path.join(val_dir, image)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Removed corrupted image: {image}")
    except Exception as e:
        print(f"Error removing image {image}: {e}")

# Dataset check
# Function to count the number of images in a directory
def count_images_in_directory(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# Count training and validation images
train_image_count = count_images_in_directory(train_dir)
val_image_count = count_images_in_directory(val_dir)

print(f"Number of training images: {train_image_count}")
print(f"Number of validation images: {val_image_count}")

# Dataset Analysis (label files check)
# Check if label files for each image exist and are valid
def check_labels(directory):
    invalid_labels = []
    for image in os.listdir(directory):
        label_file = os.path.splitext(image)[0] + '.txt'  # Assuming label files have the same name as the image
        if not os.path.exists(os.path.join(directory, label_file)):
            invalid_labels.append(image)
    
    return invalid_labels

# Check invalid labels for training and validation images
invalid_train_labels = check_labels(label_dir + '/train')
invalid_val_labels = check_labels(label_dir + '/val')

print(f"Invalid labels in training set: {invalid_train_labels}")
print(f"Invalid labels in validation set: {invalid_val_labels}")

# Training Process Recheck
# Ensure the training and validation sets are ready for training
if train_image_count == 0 or val_image_count == 0:
    raise ValueError("Not enough images in train or validation directory after cleanup.")
else:
    print("Dataset is ready for training.")