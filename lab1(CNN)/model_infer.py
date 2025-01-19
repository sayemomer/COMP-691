import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F

# Define the model class
class SimpleCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleCNN, self).__init__()
        # Input: MNIST images (1x28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(num_features=20)
        self.maxP2d = nn.MaxPool2d(2)  # Max pooling

        # After conv1: (20, 26, 26) -> After pooling: (20, 13, 13)
        converted_input_size = 20 * 13 * 13  # Flattened size

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

        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


# Function to create a scratchpad
def draw_scratchpad():
    canvas = np.ones((280, 280, 3), dtype="uint8") * 255  # White canvas
    drawing = False  # True if mouse is pressed

    # Mouse callback function
    def draw_circle(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(canvas, (x, y), 8, (0, 0, 0), -1)  # Draw black circle
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Scratchpad")
    cv2.setMouseCallback("Scratchpad", draw_circle)

    while True:
        cv2.imshow("Scratchpad", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # Clear the canvas
            canvas[:] = 255
        elif key == ord("q"):  # Quit the drawing loop
            break

    cv2.destroyAllWindows()
    return canvas


# Preprocess the drawn image
def preprocess_image(image):
    # Convert the OpenCV image (BGR) to grayscale and resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


# Predict the class of the drawn digit
def predict_image(model, device):
    print("Draw your digit. Press 'q' to quit and 'c' to clear.")
    canvas = draw_scratchpad()
    img_tensor = preprocess_image(canvas).to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"Predicted class: {predicted_class}")


# Main function to load the model and predict
def main():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters (match your training configuration)
    input_size = 784  # Flattened size (28x28)
    hidden_size = 500
    num_classes = 10  # Digits 0-9

    # Load the model
    model = SimpleCNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("mnist_model.pth", map_location=device))  # Load the saved model

    # Call the predict function
    predict_image(model, device)


if __name__ == "__main__":
    main()
