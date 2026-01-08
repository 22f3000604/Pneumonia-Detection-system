import os
import shutil
import argparse

def organize_kaggle_data(source_dir, target_dir="data"):
    """
    Moves data from Kaggle structure to project structure.
    Kaggle: source_dir/train/{NORMAL, PNEUMONIA}
    Project: target_dir/{NORMAL, PNEUMONIA}
    """
    classes = ['NORMAL', 'PNEUMONIA']
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # We prioritize the 'train' folder from Kaggle as it has the most data
    kaggle_train_path = os.path.join(source_dir, 'train')
    
    if not os.path.exists(kaggle_train_path):
        print(f"Error: Could not find 'train' folder in {source_dir}")
        print("Make sure you point to the unzipped 'chest_xray' folder.")
        return

    print(f"Moving files from {kaggle_train_path} to {target_dir}...")

    for class_name in classes:
        src_class_path = os.path.join(kaggle_train_path, class_name)
        dst_class_path = os.path.join(target_dir, class_name)
        
        if not os.path.exists(src_class_path):
            print(f"Warning: Class folder {class_name} not found in source.")
            continue
            
        os.makedirs(dst_class_path, exist_ok=True)
        
        files = os.listdir(src_class_path)
        count = 0
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(os.path.join(src_class_path, f), os.path.join(dst_class_path, f))
                count += 1
        
        print(f"Copied {count} images for class: {class_name}")

    print("\nOrganization complete! You can now run 'python train/train_model.py'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize Kaggle X-ray data.")
    parser.add_argument("source", help="Path to the unzipped Kaggle 'chest_xray' folder")
    args = parser.parse_args()
    
    organize_kaggle_data(args.source)
