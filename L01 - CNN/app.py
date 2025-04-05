from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
import base64

# Define your model
class SimpleCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(20)
        self.maxP2d = nn.MaxPool2d(2)
        converted_input_size = 20 * 13 * 13
        self.fc1 = nn.Linear(converted_input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2d(x)
        x = self.maxP2d(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


# Flask App
app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(input_size=784, hidden_size=500, num_classes=10).to(device)
model.load_state_dict(torch.load("lab1(CNN)/model.pth", map_location=device))
model.eval()


# Preprocess the image
def preprocess_image(image_array):
    # Convert to grayscale
    image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    return image.unsqueeze(0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Receive the canvas data as a base64 string
    data = request.json["image"]

    # Decode the base64 string into image bytes
    image_bytes = base64.b64decode(data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Preprocess image and predict
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return jsonify({"prediction": predicted_class})


if __name__ == "__main__":
    app.run(debug=True)
