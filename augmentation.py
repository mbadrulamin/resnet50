import torchvision.transforms as transforms
from PIL import Image
import os

# Directory containing your original 100 images
original_data_dir = 'data/dentistry_mix/val/contact'

# Directory to save augmented images
output_data_dir = 'data/augmented_data/val/contact'

# Define transformations for augmentation
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation((0, 360)),
    # Add more transformations as needed
    transforms.ToTensor(),
])

# Function to generate augmented images
def augment_data(input_image_path, output_directory, num_images):
    image = Image.open(input_image_path)
    image_name = os.path.splitext(os.path.basename(input_image_path))[0]

    for i in range(num_images):
        transformed_image = data_transforms(image)
        new_image_path = os.path.join(output_directory, f'{image_name}_aug_{i}.png')
        transformed_image = transforms.functional.to_pil_image(transformed_image)
        transformed_image.save(new_image_path)

# Augment each image in the original dataset
for root, dirs, files in os.walk(original_data_dir):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):  # Modify extensions if needed
            image_path = os.path.join(root, file)
            augment_data(image_path, output_data_dir, 1)  # Create 9 augmented images per original image

print("Augmentation process finished")
