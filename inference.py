import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# Define the transformations for the test data
data_transforms_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Path to the testing data
test_data_path = '/home/user2/Downloads/testing_data_mix'

# Load the testing data
test_dataset = ImageFolder(test_data_path, transform=data_transforms_test)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load the trained model
model = torch.load('/home/user2/Documents/model/resnet18_epoch_100_mix.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Initialize lists to store predictions and true labels
all_predictions = []
true_labels = []

# Perform inference on the testing data
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Save the results
results_file_path = '/home/user2/Documents/model_evaluation'
os.makedirs(results_file_path, exist_ok=True)

with open(os.path.join(results_file_path, 'predictions.txt'), 'w') as file:
    file.write("Predictions\n")
    file.write(str(all_predictions))

with open(os.path.join(results_file_path, 'true_labels.txt'), 'w') as file:
    file.write("True Labels\n")
    file.write(str(true_labels))

print("Finished execution")
