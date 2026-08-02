import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import numpy as np
from PIL import Image

# Class definitions
CLASSES = ["shoes", "bags", "electronics", "clothing", "groceries"]

def generate_synthetic_product(class_idx, variation):
    # Create a 224x224 RGB image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255 # start with white canvas
    
    # Random seed based on variation for reproducible augmentation
    np.random.seed(class_idx * 200 + variation)
    
    # Introduce random scale and position offsets
    scale = np.random.uniform(0.7, 1.3)
    pos_x = np.random.randint(-25, 25)
    pos_y = np.random.randint(-25, 25)
    
    # Draw custom colors and shapes representing different classes
    if class_idx == 0: # shoes: draw a dark grey/black horizontal oval
        center_y = int(160 + pos_y)
        center_x = int(112 + pos_x)
        axes_x = int(72 * scale)
        axes_y = int(20 * scale)
        cv2.ellipse(img, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, (50, 50, 50), -1)
    elif class_idx == 1: # bags: draw a brown rectangle in the middle
        start_y = int(70 + pos_y)
        end_y = int(170 + pos_y)
        start_x = int(60 + pos_x)
        end_x = int(160 + pos_x)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (139, 69, 19), -1)
        # handle
        cv2.circle(img, (int(112 + pos_x), start_y), int(25 * scale), (139, 69, 19), 3)
    elif class_idx == 2: # electronics: draw a blue/cyan screen rectangle
        start_y = int(50 + pos_y)
        end_y = int(170 + pos_y)
        start_x = int(80 + pos_x)
        end_x = int(144 + pos_x)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 200), -1)
    elif class_idx == 3: # clothing: draw a yellow shirt shape (t-shape)
        # body
        cv2.rectangle(img, (int(80 + pos_x), int(90 + pos_y)), (int(144 + pos_x), int(190 + pos_y)), (230, 230, 0), -1)
        # sleeves
        cv2.rectangle(img, (int(50 + pos_x), int(90 + pos_y)), (int(174 + pos_x), int(130 + pos_y)), (230, 230, 0), -1)
    else: # groceries: draw a green or red circle (like an apple/vegetable/tomato)
        # Alternate between green and red for groceries synthetic templates
        color = (34, 139, 34) if variation % 2 == 0 else (220, 20, 60)
        center_x = int(112 + pos_x)
        center_y = int(112 + pos_y)
        radius = int(45 * scale)
        cv2.circle(img, (center_x, center_y), radius, color, -1)
        # Draw a small green leaf for red apples to resemble the user's upload!
        if color == (220, 20, 60):
            cv2.ellipse(img, (center_x + 10, center_y - radius), (15, 8), -30, 0, 360, (34, 139, 34), -1)
            
    # Add random pixel noise to ensure the network generalizes and doesn't overfit perfectly on static pixels
    noise = np.random.randint(-20, 20, (224, 224, 3)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Convert to PyTorch channel-first format (3, 224, 224) and normalize to [0, 1]
    tensor = torch.from_numpy(img_noisy).permute(2, 0, 1).float() / 255.0
    return tensor

def train_classifier():
    print("Generating synthetic product image dataset...")
    X_train = []
    y_train = []
    
    # Generate 30 samples per class
    for idx, cname in enumerate(CLASSES):
        for var in range(30):
            tensor = generate_synthetic_product(idx, var)
            X_train.append(tensor)
            y_train.append(idx)
            
    X_train = torch.stack(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    print(f"Generated dataset shape: {X_train.shape}. Creating MobileNetV2 model...")
    
    # Initialize MobileNetV2 with pre-trained ImageNet weights (Actual Transfer Learning)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze all feature extraction layers to preserve ImageNet features
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier head for 5 classes
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training product classifier for 6 epochs on CPU...")
    model.train()
    
    dataset_size = len(X_train)
    batch_size = 15
    
    for epoch in range(6):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        correct = 0
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss /= dataset_size
        acc = correct / dataset_size
        print(f"Epoch {epoch+1}/6 - Loss: {epoch_loss:.4f} - Accuracy: {acc * 100:.2f}%")
        
    # Save the model
    os.makedirs("app/models", exist_ok=True)
    model_path = "app/models/product_classifier.pt"
    print(f"Saving trained model to {model_path}...")
    
    # Save the model state dictionary or full model. Let's save state_dict for clean loading
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': CLASSES
    }, model_path)
    
    print("Product classifier training complete!")

if __name__ == "__main__":
    train_classifier()
