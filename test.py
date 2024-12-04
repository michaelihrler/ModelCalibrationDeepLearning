from plot_utils import plot_binary_calibration_curve
from data_utils import get_class_names, calculate_ece
from data_utils import calculate_mce, calculate_cece, calculate_ace
from plot_utils import plot_loss, plot_histogram_balance_of_dataset, plot_confusion_matrix, plot_multiclass_roc_auc, \
    plot_metrics_table, plot_histogram_confidence
from model import Model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


train_val_split_ratio = 0.8
batch_size = 32
learning_rate = 1e-4
patience_early_stopping = 15
patience_reduce_learning_rate = 5
factor_reduce_learning_rate = 0.5
weight_decay = 1e-6
momentum = 0.9
epochs = 320
train_dir = "train_data_chest_xray_balanced"
test_dir = "data/chest_xray/test"

model = Model(learning_rate=learning_rate, batch_size=batch_size, patience_early_stopping=patience_early_stopping,
              patience_reduce_learning_rate=patience_reduce_learning_rate,
              factor_reduce_learning_rate=factor_reduce_learning_rate, train_dir=train_dir,
              weight_decay=weight_decay, momentum=momentum, test_dir=test_dir,
              train_val_split_ratio=train_val_split_ratio)


model.load_existing_model("baseline320.pth")

true_labels_val, predicted_labels_val, confidence_all_classes_val, _ = model.evaluate(model.val_loader)
model.optimize_platt_scaling(true_labels_val, confidence_all_classes_val[:,1])
true_labels_test, predicted_labels_test, confidence_all_classes_test, _ = model.evaluate()
model.evaluate_with_platt_scaling(confidence_all_classes_test)