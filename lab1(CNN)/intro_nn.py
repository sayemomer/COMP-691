import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(2,1)
optimizer = optim.SGD(model.parameters(),lr=0.01)

x = torch.tensor([[1.0,2.0],[3.0,4.0]],requires_grad=True)
y = torch.tensor([[1.0],[0.0]])

output = model(x)
print(output)
loss = nn.MSELoss()(output,y)

loss.backward()

for name,param in model.named_parameters():
    print(f"Gradient of {name}: {param.grad}")

optimizer.step()

optimizer.zero_grad()

print("Gradients after zero_grad:")
for name, param in model.named_parameters():
    print(f"Gradient of {name}: {param.grad}")