import os
import random
from PIL import Image, ImageDraw

def create_dummy_data(base_dir, num_samples=20):
    classes = ['NORMAL', 'PNEUMONIA']
    
    for class_name in classes:
        dir_path = os.path.join(base_dir, class_name)
        os.makedirs(dir_path, exist_ok=True)
        
        print(f"Generating {num_samples} samples for {class_name}...")
        for i in range(num_samples):
            # Create a simplified 'X-ray' like image
            # Dark background
            img = Image.new('RGB', (224, 224), color=(30, 30, 30))
            draw = ImageDraw.Draw(img)
            
            # Draw 'ribs'
            for y in range(40, 200, 30):
                draw.arc([40, y, 184, y+40], start=180, end=0, fill=(200, 200, 200), width=5)
            
            # If Pneumonia, add some 'hazy' artifacts
            if class_name == 'PNEUMONIA':
                # Add random cloudy patches
                for _ in range(5):
                    x = random.randint(50, 150)
                    y = random.randint(50, 150)
                    r = random.randint(10, 30)
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=(100, 100, 100), outline=None)
            
            img.save(os.path.join(dir_path, f"sample_{i}.jpg"))
            
    print(f"Dummy data generated in {base_dir}")

if __name__ == "__main__":
    # Generate data in the 'data' folder
    create_dummy_data("data")
