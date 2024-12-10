# train_tc.py
import os
import argparse
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import logging
import scipy.ndimage
import sys
import seaborn as sns
import io

from model_def import build_modified_squeezenet
from plots import plot_learning_curves, plot_confusion_matrix
from utils import setup_logging, create_output_directory, restore_stdout, create_dataset, compute_dataset_statistics, load_dataset, parse_args_with_json, save_history_to_json

# Main script
if __name__ == "__main__":
    # Use the parse_args_with_json function to get parsed arguments
    #tf.config.set_visible_devices([], 'GPU')

    output_dir = create_output_directory()
    log_file_path = setup_logging(output_dir)

    args = parse_args_with_json(output_dir)

    # Parameters from parsed arguments
    dataset_dir = args.dataset_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    blur_pct = args.blur_pct
    median_pct = args.median_pct
    displacement_pct = args.displacement_pct
    data_aug = args.data_aug
    l_reg = args.l_reg
    verbosity = args.verbosity
    patience = args.patience

    # Load dataset
    all_file_paths, all_labels = load_dataset(dataset_dir)
    compute_dataset_statistics(all_labels, os.path.join(output_dir,"metrics"))

    # Split dataset
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_file_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=42)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

    # Create datasets
    train_dataset = create_dataset(train_files, train_labels, batch_size, shuffle=True, augment=data_aug,
                                   blur_pct=blur_pct, median_pct=median_pct, displacement_pct=displacement_pct,
                                   output_dir=output_dir)
    val_dataset = create_dataset(val_files, val_labels, batch_size, shuffle=False, augment=False)
    test_dataset = create_dataset(test_files, test_labels, batch_size, shuffle=False, augment=False)

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = dict(enumerate(class_weights))

    # Build model
    model = build_modified_squeezenet(input_shape=(64, 158, 1), num_classes=len(np.unique(all_labels)), l_reg=l_reg)
    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("\nTraining starts...\n")

    # Train model
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "models", "best_model.h5"), save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True, verbose=1)
    ]
    restore_stdout()
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, class_weight=class_weights, callbacks=callbacks, verbose=1)
    setup_logging(output_dir)
    save_history_to_json(history, os.path.join(output_dir,"metrics"))

    # Plot learning curves
    plot_learning_curves(history, os.path.join(output_dir,"plots"))

    # Evaluate on test set
    restore_stdout()
    test_loss, test_accuracy = model.evaluate(test_dataset)
    setup_logging(output_dir)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report and confusion matrix
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(len(np.unique(all_labels)))])
    cm = confusion_matrix(y_true, y_pred)
    print(report)
    print("Confusion Matrix:\n", cm)
    class_names = [f"Class {i}" for i in range(len(np.unique(all_labels)))]
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "plots", "confusion_matrix.png"))

    # Save report and model
    with open(os.path.join(output_dir,"metrics",  "classification_report.txt"), "w") as f:
        f.write(report)
    pd.DataFrame(cm).to_csv(os.path.join(output_dir,"metrics", "confusion_matrix.csv"))
    model.save(os.path.join(output_dir, "models", "final_model.h5"))

    print(f"Training complete. Outputs saved to {output_dir}")

