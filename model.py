import os

import torch

from betacal import BetaCalibration
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import optim, nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
from data_utils import get_class_names, calculate_ece, calculate_mce
from histgram_binning import HistogramBinning
from isotonic_calibration import IsotonicCalibration

import ml_insights as mli
from temperature_scaling import TemperatureScaling


class Model:
    def __init__(self, train_dir,
                 test_dir, learning_rate=0.001, batch_size=32, patience_early_stopping=10,
                 patience_reduce_learning_rate=4,
                 factor_reduce_learning_rate=0.1, train_val_split_ratio=0.8, weight_decay=1e-4, momentum=0.9):
        self.factor_reduce_learning_rate = factor_reduce_learning_rate
        self.spline_calibration_model = None
        self.beta_calibration_model = None
        self.isotonic_calibration_model = None
        self.platt_scaling = None
        self.temperature_model = None
        self.histogram_binning_model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.create_data_loaders(batch_size, test_dir, train_dir, train_val_split_ratio)

        self.train_dir = train_dir

        # Load a pretrained VGG16 model
        base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        base_model.features = nn.Sequential(*list(base_model.features.children())[:30])  # Up to 'block5_conv3'

        # Add custom head
        self.model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),  # GlobalAveragePooling2D equivalent
            nn.Flatten(),
            nn.Linear(512, len(get_class_names(train_dir)))  # Replace 512 with num_features if different
        )

        self.model = self.model.to(self.device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                   weight_decay=weight_decay)

        self.patience_early_stopping = patience_early_stopping

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=self.factor_reduce_learning_rate,
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

            ece = calculate_ece(true_labels, confidence_all_classes)
            mce = calculate_mce(true_labels, confidence_all_classes)

            acc = accuracy_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']}, "
                f"ECE: {ece:.4f}, "
                f"MCE: {mce:.4f}, "
                f"Acc: {acc:.4f}, "
                f"F1: {f1:.4f}"
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

    def evaluate(self, dataloader=None):
        """
        Evaluate the model on the given dataloader.

        :param dataloader: DataLoader to evaluate on. Defaults to test_loader if None.
        :return: Tuple (true_labels, predicted_labels, confidence_all_classes, average_loss)
        """
        self.model.eval()
        true_labels = []
        predicted_labels = []
        confidence_all_classes = []
        logits = []
        total_loss = 0.0  # Use a separate variable for accumulating loss

        if dataloader is None:
            dataloader = self.test_loader

        with torch.no_grad():
            for input_tensor, label_tensor in dataloader:
                input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)

                output_tensor = self.model(input_tensor)
                logits.extend(output_tensor.cpu().tolist())
                probabilities = torch.softmax(output_tensor, dim=1).cpu().numpy()

                confidence_all_classes.append(probabilities)
                _, predicted_tensor = torch.max(torch.tensor(probabilities), dim=1)

                true_labels.extend(label_tensor.cpu().tolist())
                predicted_labels.extend(predicted_tensor.cpu().tolist())

                batch_loss = self.criterion(output_tensor, label_tensor)
                total_loss += batch_loss.item()  # Accumulate batch loss

        # Calculate average loss over all batches
        avg_loss = total_loss / len(dataloader)
        confidence_all_classes = np.vstack(confidence_all_classes)

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

        # Ensure split is all the Time the same! Important!
        torch.manual_seed(42)

        # Define train-validation split
        train_size = int(train_val_split_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size

        # Perform the split
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        # Create data loaders for each subset
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def get_class_mappings(self):
        train_dataset = ImageFolder(self.train_dir)
        return train_dataset.class_to_idx

    def optimize_temperature(self, lr=0.01):
        """
        Optimizes the temperature scaling model for multi-class logits.
        :return: The optimized TemperatureScaling model.
        """
        temperature_model = TemperatureScaling().to(self.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS([temperature_model.temperature], lr=lr, max_iter=50)

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
        self.temperature_model = temperature_model

    def optimize_platt_scaling(self, y_true, y_pred_confidence):
        y_true = np.array(y_true)
        y_pred_confidence = np.array(y_pred_confidence)
        self.platt_scaling = LogisticRegression(C=99999999999, solver='lbfgs')
        self.platt_scaling.fit(y_pred_confidence.reshape(-1, 1), y_true)

    def optimize_isotonic_calibration(self):
        logits_list = []
        labels_list = []

        # Collect logits and labels from validation data
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)  # Shape: [num_samples, num_classes]
        labels = torch.cat(labels_list)  # Shape: [num_samples]

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()

        # Fit isotonic regression
        isotonic_calibration = IsotonicCalibration()
        isotonic_calibration.fit(probabilities, true_labels)
        self.isotonic_calibration_model = isotonic_calibration

    def optimize_histogram_binning(self, n_bins=20):
        logits_list = []
        labels_list = []

        # Collect logits and labels from validation data
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)  # Shape: [num_samples, num_classes]
        labels = torch.cat(labels_list)  # Shape: [num_samples]

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        labels_one_hot = np.eye(probabilities.shape[1])[labels.cpu().numpy()]

        # Fit histogram binning
        histogram_binning = HistogramBinning(n_bins=n_bins)
        histogram_binning.fit(probabilities, labels_one_hot)
        self.histogram_binning_model = histogram_binning

    def optimize_beta_calibration(self, y_true, y_pred_confidence):
        y_true = np.array(y_true)
        y_pred_confidence = np.array(y_pred_confidence)

        self.beta_calibration_model = BetaCalibration()
        self.beta_calibration_model.fit(y_pred_confidence, y_true)

    def optimize_spline_calibration(self, y_true, y_pred_confidence):
        self.spline_calibration_model = mli.SplineCalib(
            knot_sample_size=40,
            cv_spline=5,
            unity_prior=False,
            unity_prior_weight=128)
        self.spline_calibration_model.fit(y_pred_confidence, y_true)

    def evaluate_with_platt_scaling(self, y_pred_confidence):
        y_pred_confidence = np.array(y_pred_confidence)
        calibrated_probabilities = self.platt_scaling.predict_proba(y_pred_confidence.reshape(-1, 1))[:, 1]
        # Apply a threshold of 0.5 to get predicted labels
        predicted_tensor = torch.tensor((calibrated_probabilities >= 0.5).astype(int))
        return predicted_tensor.tolist(), convert_1d_probs_to_2d_probs(calibrated_probabilities)

    def evaluate_with_temperature_scaling(self, logits):
        logits_tensor = torch.tensor(logits).to(self.device)

        with torch.no_grad():
            calibrated_logits = self.temperature_model(logits_tensor)
            calibrated_probabilities = torch.softmax(calibrated_logits, dim=1).cpu().numpy()
        _, predicted_tensor = torch.max(torch.tensor(calibrated_probabilities), dim=1)
        return predicted_tensor.tolist(), np.vstack(calibrated_probabilities)

    def evaluate_with_histogram_binning(self, probabilities):
        with torch.no_grad():
            calibrated_probabilities = self.histogram_binning_model.forward(probabilities)
        _, predicted_tensor = torch.max(torch.tensor(calibrated_probabilities), dim=1)
        return predicted_tensor.tolist(), np.vstack(calibrated_probabilities)

    def evaluate_with_isotonic_calibration(self, probabilities):
        with torch.no_grad():
            calibrated_probabilities = self.isotonic_calibration_model.forward(probabilities)
        _, predicted_tensor = torch.max(torch.tensor(calibrated_probabilities), dim=1)
        return predicted_tensor.tolist(), np.vstack(calibrated_probabilities)

    def evaluate_with_beta_calibration(self, y_pred_confidence):
        calibrated_probabilities = self.beta_calibration_model.predict(y_pred_confidence)
        # Apply a threshold of 0.5 to get predicted labels
        predicted_tensor = torch.tensor((calibrated_probabilities >= 0.5).astype(int))
        return predicted_tensor.tolist(), convert_1d_probs_to_2d_probs(calibrated_probabilities)

    def evaluate_with_spline_calibration(self, y_pred_confidence):
        calibrated_probabilities = self.spline_calibration_model.predict(y_pred_confidence)
        # Apply a threshold of 0.5 to get predicted labels
        predicted_tensor = torch.tensor((calibrated_probabilities >= 0.5).astype(int))
        return predicted_tensor.tolist(), convert_1d_probs_to_2d_probs(calibrated_probabilities)

    def load_existing_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")


def convert_1d_probs_to_2d_probs(probs):
    probs_2d = np.column_stack((
        1 - probs,  # Inverted probabilities
        probs  # Original probabilities
    ))
    return probs_2d
