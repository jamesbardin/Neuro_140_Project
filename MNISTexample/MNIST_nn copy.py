import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Define a simple model
model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Optimizer
optimiser = optim.SGD(model.parameters(), lr = 1e-2)

loss = nn.CrossEntropyLoss()

# Train, Val split
train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size =32)
val_loader = DataLoader(val, batch_size=32)

# Training and validation loops
nb_epochs = 5
for epoch in range(nb_epochs):
    losses = list()
    for batch in train_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        #1 forward training
        l = model(x)  # l: logits

        #2 computing the obective function
        J = loss(l, y)

        #3 cleaning the gradients
        model.zero_grad()

        #4 accumulate the partial derivatives of J w.r.t. the parameters
        J.backward()

        #5 step in opposite direction of the gradient
        optimiser.step()
            #with torch.no_grad(): params = params - eta * params.grad (manual version of step above) (params is dict so use for every item in...)

        losses.append(J.item())

    print(f'Epoch {epoch + 1}, train loss: {torch.tensor(losses).mean():.2f}')


    losses = list()
    for batch in val_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        #1 forward training
        with torch.no_grad():

            l = model(x)  # l: logits
        #2 computing the obective function
        J = loss(l, y)

        losses.append(J.item())

    print(f'Epoch {epoch + 1}, validation loss: {torch.tensor(losses).mean():.2f}')