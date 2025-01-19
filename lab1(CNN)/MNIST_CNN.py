
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

torch.manual_seed(1)

device = torch.device("mps" if torch.device('mps') else "cpu")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Neural Network Module
class Net(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net,self).__init__()

    #here the input of the image from MNIST is (128,1,28,28) => (3584,28)

    self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1,padding=1)
    self.bn2d = nn.BatchNorm2d(num_features=20)
    self.maxP2d = nn.MaxPool2d(2)

    #after conv1 layer the image becomes (128,20,26,26) => (66560, 26)
    #after the max pool layer it becomes (128,20,13,13) => (33280,13)

    #but the fc1 takes (128,784)

    #so we need to flat change the input size

    converted_input_size = 20* 13 *13


    self.fc1 = nn.Linear(converted_input_size, hidden_size)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.relu = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self,x):

    out = self.conv1(x)
    out = self.bn2d(out)
    out = self.maxP2d(out)
    out = self.relu(out)

    out = out.reshape(out.size(0), -1)


    out = self.fc1(out)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.dropout1(out)
    out = self.fc2(out)
    return out

# Hyperparameters
batch_size=128
epochs=50
lr=0.005
input_size = 784
hidden_size = 500
num_classes = 10

# Model initalization
model = Net(input_size, hidden_size, num_classes)
model = model.to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Datasets
train_data = datasets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = datasets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())
# Data loaders
train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                             batch_size = batch_size,
                                             shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                      batch_size = batch_size,
                                      shuffle = False)

# Training over all batches
def train(train_loader, model, loss, optimizer):
  losses=[]
  model.train()
  for images,labels in train_loader: # 1.data load
    # images = images.reshape(images.size(0),-1)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images) # 2.train
    l = loss(outputs, labels) # 3.loss

    optimizer.zero_grad()
    l.backward() # 4.optimize
    optimizer.step() # 5.update the gradiant
    losses.append(l)
  return torch.tensor(losses).mean()

torch.save(model.state_dict(), "model.pth")

# Test over all batches
def test(test_loader, model, loss):
  losses=[]
  err=0
  model.eval()
  for images,labels in test_loader:
    # images = images.reshape(images.size(0),-1)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    l = loss(outputs, labels)
    losses.append(l)

    predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
    err = err + (predictions!=labels).sum()
  return torch.tensor(losses).mean(), err/len(test_loader.dataset)

# Training loop
train_losses=[]
test_losses=[]
for epoch in range(epochs):
  train_loss = train(train_loader, model, loss, optimizer)
  test_loss, test_err = test(test_loader, model, loss)
  train_losses.append(train_loss)
  test_losses.append(test_loss)

  print('Epoch: {}  train_loss={:.4f}, test_loss={:.4f}, test_err={:.2f}%'.format(epoch+1, train_loss, test_loss, test_err*100))