import os
import torch
from torch import optim, nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from data_utils import get_class_names
from platt_scaling import PlattScaling
from temperature_scaling import TemperatureScaling


class Model:
    def __init__(self, learning_rate, batch_size, patience_early_stopping, patience_reduce_learning_rate, train_dir,
                 test_dir, train_val_split_ratio, weight_decay=1e-4, momentum=0.9):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.create_data_loaders(batch_size, test_dir, train_dir, train_val_split_ratio)

        # Load a pretrained resnet
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        num_features = self.model.fc.in_features

        class_names = get_class_names(train_dir)
        self.model.fc = nn.Linear(num_features, len(class_names))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                   weight_decay=weight_decay)

        self.patience_early_stopping = patience_early_stopping

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
        self.model.eval()
        true_labels = []
        predicted_labels = []
        confidence_predictions = []
        confidence_all_classes_tensor = []

        with torch.no_grad():
            for input_tensor, label_tensor in self.test_loader:
                input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)

                output_tensor = self.model(input_tensor)

                probabilities = torch.softmax(output_tensor, dim=1)

                for prob in probabilities: confidence_all_classes_tensor.append(prob)
                # Get the predicted class (class with the highest probability)
                confidence_tensor, predicted_tensor = torch.max(probabilities, dim=1)

                # Track true labels, predictions, and confidence scores
                true_labels.extend(label_tensor.cpu().tolist())
                predicted_labels.extend(predicted_tensor.cpu().tolist())
                confidence_predictions.extend(confidence_tensor.cpu().tolist())

        return true_labels, predicted_labels, confidence_predictions, np.array(confidence_all_classes_tensor)

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

    def optimize_temperature(self):
        temperature_model = TemperatureScaling().to(device)
        criterion = nn.CrossEntropyLoss()

        # alternativen (z. B. SGD oder Adam)
        # TODO
        optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimierungsfunktion (Closure für LBFGS)
        def closure():
            optimizer.zero_grad()
            scaled_logits = temperature_model(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimized Temperature: {temperature_model.temperature.item()}")
        return temperature_model.temperature.item()

    def optimize_platt_scaling(self):
        platt_scaling = PlattScaling().to(self.device)
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

        # Optimizer (z. B. SGD oder Adam)
        optimizer = optim.LBFGS([platt_scaling.w, platt_scaling.b], lr=0.01, max_iter=50)

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels.float())

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimierungsfunktion (Closure für LBFGS)
        def closure():
            optimizer.zero_grad()
            scaled_logits = platt_scaling(logits)  # Calibrated probs
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
        optimizer.step(closure)

        print(f"Optimized Parameters: w={platt_scaling.w.item()}, b={platt_scaling.b.item()}")
        return platt_scaling
