import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(1)


class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.activation(x)

model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


losses = []
for epoch in range(100):

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad() #reset the gradiant
    loss.backward()  # Gradients are stored in .grad
    optimizer.step()


    losses.append(loss.item())


    if (epoch + 1) % 20 == 0: 
        print(f"Epoch {epoch+1}")
        for name, param in model.named_parameters():
            print(f"{name}.grad: {param.grad}")


plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss with Gradients Stored")
plt.show()


model.eval()


with torch.no_grad():  
    final_outputs = model(X)


print("Final outputs after training:")
print(final_outputs)


predicted_labels = (final_outputs >= 0.5).float()  # Convert probabilities to binary predictions
print("Predicted labels:")
print(predicted_labels)


print("Actual labels:")
print(y)

