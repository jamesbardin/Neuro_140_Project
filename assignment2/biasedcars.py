import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pickle
import imageio as iio
from skimage import io
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import copy

data_directory = 'C:/NEURO140/assignment2/biased_cars_1/biased_cars_1'

label_path = "C:/NEURO140/assignment2/att_dict_simplified.p"
file_label = open(label_path,'rb')
set_label = pickle.load(file_label)

os.listdir(data_directory)

class CarDataset(Dataset):
  def __init__(self, data_directory, dictionary, train=True):
      self.dictionary = dictionary
      self.data_directory = data_directory
      self.images=[]
      self.path=[]

      folders = os.listdir(data_directory)
      if train:
        for directory in folders:
          sub_directory = os.listdir(data_directory + '/' + directory + '/train/images/')
          for img in sub_directory:
            if img in self.dictionary:
              self.images.append(img)
              self.path.append(data_directory + '/' + directory + '/train/images/' + img)
      else:
        for directory in folders:
          sub_directory = os.listdir(data_directory + '/' + directory + '/test/images/')
          for img in sub_directory:
            if img in self.dictionary:
              self.images.append(img)
              self.path.append(data_directory + '/' + directory + '/test/images/' + img)
      

  def __getitem__(self, index):
    image_name = self.images[index]
    image = io.imread(self.path[index]) / 255.0
    label = self.dictionary[image_name]
    return {'image': torch.transpose(torch.from_numpy(image), 0, 2).float(), 'label': label[2]}

  

  def __len__(self):
    return len(self.images)

train_dataset = CarDataset(dictionary = set_label, data_directory = data_directory)

test_dataset = CarDataset(dictionary = set_label, data_directory = data_directory, train=False)


def show_label(image, label):
    plt.imshow(image)

fig = plt.figure()

for i in range(len(train_dataset)):
    sample = train_dataset[i]
    sample['image'] = torch.transpose(sample['image'], 0, 2)

    ax = plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    
    ax.set_title('Sample {}'.format(i))
    ax.axis('off')
    show_label(**sample)

    if i == 5:
        plt.show()
        break

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)

dset_loaders = {'Train': train_loader, 'Val': test_loader}

dset_sizes = {'Train': len(train_dataset), 'Val': len(test_loader)}

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=30):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=5):
    since = time.time()

    best_model = model.cuda()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}, learning is {}'.format(epoch + 1, num_epochs, optimizer.param_groups[0]['lr']))
        print()
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                mode = 'Train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train() 
            else:
                model.eval()
                mode = 'Val'

            running_loss = 0.0
            running_corrects = 0

            counter = 0

            for data in dset_loaders[phase]:
                inputs = data['image']
                labels = data['label']
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # print stats every 500 batches so ik its still running
                if counter % 500 == 0:
                    print('Epoch: {} ({:.0f}%)\t{} Loss: {:.5f} Accuracy: {:.2f}'.format(
                        epoch + 1, 
                        100.0 * counter / len(dset_loaders[phase]),
                        phase, 
                        loss.item(), torch.sum(preds == labels.data).item() / len(inputs))
                    )

                counter += 1

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.double() / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            

            if phase == 'Val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print('Training finished in {:.0f} minutes and {:.0f} seconds'.format(
        (time.time() - since) // 60, (time.time() - since) % 60))
    print('Best Val Acc: {:4f}'.format(best_acc))

    return best_model

model = torchvision.models.resnet18(pretrained=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

torch.transpose(train_dataset[0]['image'], 0, 2).shape

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler)
