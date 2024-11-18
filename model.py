import os
import torch
from torch import optim, nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F  # For softmax function
import numpy as np


class Model:
    def __init__(self, learning_rate, batch_size, patience_early_stopping, patience_reduce_learning_rate, train_dir,
                 test_dir, class_names, train_val_split_ratio):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.create_data_loaders(batch_size, test_dir, train_dir, train_val_split_ratio)
        # Load a pretrained resnet
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(class_names))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

        self.patience_early_stopping = patience_early_stopping
        # Define ReduceLROnPlateau scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                                              patience=patience_reduce_learning_rate)

    def train_model(self, num_epochs):
        best_val_loss = np.inf
        counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for input_tensor, label_tensor in self.train_loader:
                input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)

                output_tensor = self.model(input_tensor)
                loss = self.criterion(output_tensor, label_tensor)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for input_tensor, label_tensor in self.val_loader:
                    input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)
                    output_tensor = self.model(input_tensor)
                    loss = self.criterion(output_tensor, label_tensor)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Learning Rate: {self.optimizer.param_groups[0]['lr']}")

            # Reduce Learning Rate on Plateau
            self.scheduler.step(val_loss)

            # Check for Early Stopping und store best model
            # Improved mode
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model_state = self.model.state_dict()
            # No improvement
            else:
                counter += 1
                if counter >= self.patience_early_stopping:
                    print(f'Early stopping after {epoch + 1} Epochs')
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return train_losses, val_losses

    def evaluate(self):
        self.model.eval()  # Set model to evaluation mode
        true_labels = []
        predicted_labels = []
        confidence_values = []
        test_correct_preds = 0
        test_total_preds = 0

        # Assuming you have a test_loader for the test set
        with torch.no_grad():  # No need to compute gradients during evaluation
            for input_tensor, label_tensor in self.test_loader:
                input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)

                # Forward pass: get model outputs (logits)
                output_tensor = self.model(input_tensor)

                # Apply softmax to get probabilities (confidence scores)
                probabilities = F.softmax(output_tensor, dim=1)

                # Get the predicted class (class with the highest probability)
                value_tensor, predicted_tensor = torch.max(probabilities, 1)

                is_sample_correct_predicted = (predicted_tensor == label_tensor)
                for label, predicted_label, value in zip(label_tensor, predicted_tensor, value_tensor):
                    true_labels.append(label.item())
                    predicted_labels.append(predicted_label.item())
                    confidence_values.append(value.item())
                    if predicted_label == label:
                        test_correct_preds += 1

                    test_total_preds += 1

        test_acc = test_correct_preds / test_total_preds
        print(f"Test Accuracy: {test_acc:.4f}")
        return true_labels, predicted_labels, confidence_values

    def create_data_loaders(self, batch_size, test_dir, train_dir, train_val_split_ratio):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create custom ImageNet dataset loaders
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)

        # Split train set in train_set and val_set
        train_size = int(train_val_split_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        # Create data loaders for each subset
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
