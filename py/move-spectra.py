import os
import shutil

def move_files_to_master(src_dir, master_dir):
    """
    Move all files from subdirectories of src_dir to master_dir.
    
    If a file with the same name exists in master_dir, it appends a number to avoid overwriting.
    """
    if not os.path.exists(master_dir):
        os.makedirs(master_dir)

    for root, _, files in os.walk(src_dir):
        if root == master_dir:
            continue  # Skip the master directory if it's inside src_dir

        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(master_dir, file)

            # Rename if file already exists in master_dir
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(master_dir, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(src_path, dest_path)
            print(f"Moved: {src_path} -> {dest_path}")

# Example usage
src_directory = "/Users/dhortadarrington/Documents/Projects/Lux-BOSS/spec/sdsswork"  # Change this to your source directory
master_directory = "/Users/dhortadarrington/Documents/Projects/Lux-BOSS/spectra"  # Change this to your master folder

move_files_to_master(src_directory, master_directory)