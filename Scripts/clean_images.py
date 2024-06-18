import os
from PIL import Image

def delete_small_images(base_dir, min_width, min_height):
    train_dir = os.path.join(base_dir, 'train')
    deleted_images = []

    for root, _, files in os.walk(train_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width < min_width or height < min_height:
                        img.close()
                        os.remove(file_path)
                        deleted_images.append(file_path)
                        print(f"Deleted {file_path} (size: {width}x{height})")
            except PermissionError:
                print(f"Permission denied: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return deleted_images

if __name__ == "__main__":
    base_dir = 'C:/Users/Vartotojas/Desktop/project/dataset'
    min_width, min_height = 100, 100
    delete_small_images(base_dir, min_width, min_height)