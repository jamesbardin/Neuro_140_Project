import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pickle

class CarDataset(Dataset):
    def __init__(self, data_dir, att_dict):
        self.data_dir = data_dir
        self.att_dict = att_dict
        self.labels = [label for label in att_dict.keys()]
        self.image_paths, self.image_labels = self.get_image_paths_and_labels()

    def get_image_paths_and_labels(self):
        image_paths = []
        image_labels = []
        for label in self.labels:
            label_str = str(label).zfill(2)
            subfolders = os.listdir(os.path.join(self.data_dir, r'biased_cars_1\biased_cars_1'))
            for subfolder in subfolders:
                if not os.path.isdir(os.path.join(self.data_dir, r'biased_cars_1\biased_cars_1', subfolder)):
                    continue
                image_folder = os.path.join(self.data_dir, r'biased_cars_1\biased_cars_1', subfolder, r'train\images')
                for filename in os.listdir(image_folder):
                    # if filename.endswith('.png') and label in self.att_dict[filename]:
                    if filename in att_dict:
                        if filename.endswith('.png') and label in self.att_dict[filename]:
                            image_paths.append(os.path.join(image_folder, filename))
                            image_labels.append(label)
                image_folder = os.path.join(self.data_dir, r'biased_cars_1\biased_cars_1', subfolder, r'test\images')
                for filename in os.listdir(image_folder):
                    # if filename.endswith('.png') and label in self.att_dict[filename]:
                    if filename in att_dict:
                        if filename.endswith('.png') and label in self.att_dict[filename]:
                            image_paths.append(os.path.join(image_folder, filename))
                            image_labels.append(label)
        if not image_paths:
            raise ValueError("No images found")
        return image_paths, image_labels

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        label = self.image_labels[index]
        return image, label

    def __len__(self):
        return len(self.image_paths)


# load data
data_dir = r'C:\NEURO140\assignment2'
att_dict_path = r'C:\NEURO140\assignment2\att_dict_simplified.p'
with open(att_dict_path, 'rb') as f:
    att_dict = pickle.load(f)
train_dataset = CarDataset(data_dir, att_dict)
test_dataset = CarDataset(data_dir, att_dict)

# data loaders for the train and test datasets
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# ResNet18 model
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(in_features=512, out_features=5)

# loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train for 5 epochs
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print stats
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print(running_loss)