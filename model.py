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
from plot_utils import plot_multiclass_calibration_curve
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
        best_val_loss = float("inf")
        counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for input_tensor, label_tensor in self.train_loader:
                input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)

                self.optimizer.zero_grad()  # Reset gradients
                output_tensor = self.model(input_tensor)
                loss = self.criterion(output_tensor, label_tensor)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()  # Accumulate batch loss

            # Calculate average train loss
            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            true_labels, predicted_labels, confidence_all_classes, val_loss = self.evaluate(dataloader=self.val_loader)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']}"
            )

            plot_multiclass_calibration_curve(
                y_true=true_labels,
                y_pred_proba=np.array(confidence_all_classes),
                title="Baseline Training"
            )
            
            val_losses.append(val_loss)

            # Reduce Learning Rate on Plateau
            self.scheduler.step(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model_state = self.model.state_dict()
            else:
                counter += 1
                if counter >= self.patience_early_stopping:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return train_losses, val_losses

    def evaluate(self, calibration_model=None, dataloader=None):
        """
        Evaluate the model on the given dataloader.

        :param calibration_model: Optional calibration model to apply to logits.
        :param dataloader: DataLoader to evaluate on. Defaults to test_loader if None.
        :return: Tuple (true_labels, predicted_labels, confidence_all_classes, average_loss)
        """
        self.model.eval()
        true_labels = []
        predicted_labels = []
        confidence_all_classes = []
        total_loss = 0.0  # Use a separate variable for accumulating loss

        if dataloader is None:
            dataloader = self.test_loader

        with torch.no_grad():
            for input_tensor, label_tensor in dataloader:
                input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)

                output_tensor = self.model(input_tensor)

                if calibration_model is not None:
                    output_tensor = calibration_model(output_tensor)

                probabilities = torch.softmax(output_tensor, dim=1)
                confidence_all_classes.append(probabilities.cpu())
                _, predicted_tensor = torch.max(probabilities, dim=1)

                true_labels.extend(label_tensor.cpu().tolist())
                predicted_labels.extend(predicted_tensor.cpu().tolist())

                batch_loss = self.criterion(output_tensor, label_tensor)
                total_loss += batch_loss.item()  # Accumulate batch loss

        # Calculate average loss over all batches
        avg_loss = total_loss / len(dataloader)
        confidence_all_classes = torch.cat(confidence_all_classes).numpy()

        return true_labels, predicted_labels, confidence_all_classes, avg_loss

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
        """
        Optimizes the temperature scaling model for multi-class logits.
        :return: The optimized TemperatureScaling model.
        """
        temperature_model = TemperatureScaling().to(self.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

        logits_list = []
        labels_list = []

        # Collect logits and labels from the validation set
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)  # Shape: [num_samples, num_classes]
        labels = torch.cat(labels_list)  # Shape: [num_samples]

        # Optimization closure for LBFGS
        def closure():
            optimizer.zero_grad()
            scaled_logits = temperature_model(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        # Perform optimization
        optimizer.step(closure)
        return temperature_model

    def optimize_platt_scaling(self):
        num_classes = len(self.model.fc.weight)  # Number of output classes
        platt_scaling = PlattScaling(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()  # Multi-class Cross-Entropy Loss

        optimizer = optim.LBFGS([platt_scaling.w, platt_scaling.b], lr=0.01, max_iter=50)

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)  # Shape: [num_samples, num_classes]
        labels = torch.cat(labels_list)  # Shape: [num_samples]

        def closure():
            optimizer.zero_grad()
            scaled_logits = platt_scaling(logits)  # Calibrated probabilities
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return platt_scaling

    def load_existing_model(self, model_path):
        """
        Load an existing model's state from a file.

        :param model_path: Path to the saved model's state dictionary.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")
