import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import pickle
import os
from PIL import Image

torch.randn(5).cuda()


# Define a simple model
model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 10)
)


# Fanicer Model

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits
model = ResNet().cuda()


# Optimizer
optimiser = optim.SGD(model.parameters(), lr = 1e-2)

loss = nn.CrossEntropyLoss()


# Train, Val split



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

# data loaders for the train and test datasets
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# Training and validation loops
nb_epochs = 5
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    model.train()
    for batch in train_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1).cuda()

        #1 forward training
        l = model(x)  # l: logits

        #2 obective function
        J = loss(l, y.cuda())

        #3 cleaning the gradients
        model.zero_grad()

        #4 accumulate the partial derivatives of J w.r.t. the parameters
        J.backward()

        #5 step in opposite direction of the gradient
        optimiser.step()
            #with torch.no_grad(): params = params - eta * params.grad (manual version of step above) (params is dict so use for every item in...)

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1).cpu()).float().mean())

    print(f'Epoch {epoch + 1}, train loss: {torch.tensor(losses).mean():.2f}, train acc: {torch.tensor(accuracies).mean():.2f}')

    losses = list()
    accuracies = list()
    model.eval()

    for batch in test_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1).cuda()

        #1 forward training
        with torch.no_grad():
            l = model(x) # logits

        #2 computing the obective function
        J = loss(l, y.cuda())

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1).cpu()).float().mean())

    print(f'Epoch {epoch + 1}, validation loss: {torch.tensor(losses).mean():.3f}, train acc: {torch.tensor(accuracies).mean():.3f}')



