import os

# Dataset paths
train_dir = 'D:/PLADICS/merged_dataset_crossval/images/train'
val_dir = 'D:/PLADICS/merged_dataset_crossval/images/val'
label_dir = 'D:/PLADICS/merged_dataset_crossval/labels'  # Assuming labels are stored here

# Step 1: Check the number of images in training and validation directories
def count_images_in_directory(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# Count training and validation images
train_image_count = count_images_in_directory(train_dir)
val_image_count = count_images_in_directory(val_dir)

print(f"Number of training images after cleanup: {train_image_count}")
print(f"Number of validation images after cleanup: {val_image_count}")

# Step 2: Check if each image has a corresponding valid label
def check_labels(directory):
    invalid_labels = []
    for image in os.listdir(directory):
        label_file = os.path.splitext(image)[0] + '.txt'  # Assuming label files have the same name as the image
        if not os.path.exists(os.path.join(directory, label_file)):
            invalid_labels.append(image)
    
    return invalid_labels

# Check for invalid labels for training and validation images
invalid_train_labels = check_labels(label_dir + '/train')
invalid_val_labels = check_labels(label_dir + '/val')

print(f"Invalid labels in training set: {invalid_train_labels}")
print(f"Invalid labels in validation set: {invalid_val_labels}")

# Step 3: Final Check
if train_image_count == 0 or val_image_count == 0:
    raise ValueError("Not enough images in train or validation directory after cleanup.")
else:
    print("Dataset is ready for training.")