import torch
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

# Create data loader
class ImageDataset(datasets):
    def __init__(self, root, transform=None, target_transform=None, download=False):
        super(ImageDataset, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = self.load_image
        self.classes = ['dog', 'cat']

    def load_image(self, path):
        img = Image.open(path)
        img = img.resize((32, 32))
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32)
        img = img / 255.0
        return img
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        return len(self.data)


transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
image_dataset = ImageDataset(root='/home/chelvan/Documents/ANN-Pytorch/Convolution_NN/data', transform=transformer)
image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=2)

# LeNet
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# traininig
for epoch in range(2):
    for i, (images, labels) in enumerate(
        image_dataloader
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{2}], Step [{i + 1}/{len(image_dataloader)}], Loss: {loss.item():.4f}')


# Predict
def predict(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    img = img / 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()