# xgboost_feature_importance.py
import os
import time
import json
import numpy as np
import pandas as pd
import threading
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import (
    load_images_from_folder,
    ensure_results_directory,
    save_training_plot,
    save_sensor_map,
    profile_resource_usage
)
import argparse
import sys
import matplotlib.pyplot as plt


def calculate_xgb_memory(xgb_model):
    """
    Estimate memory usage of an XGBoost model by checking serialized size.
    Args:
        xgb_model (XGBClassifier): Trained XGBoost model.
    Returns:
        int: Estimated memory usage in bytes.
    """
    booster_raw = xgb_model.get_booster().save_raw()
    return len(booster_raw)  # Length of serialized model data in bytes

def optimize_xgb(max_depth, n_estimators_max, target_memory, X_train, X_test, y_train, y_test):
    print("Optimizing XGBoost to fit within memory constraints...")
    best_accuracy = 0
    best_max_depth = 0
    best_n_estimators = 0
    best_xgb = None  # Initialize to None to prevent UnboundLocalError

    for n_estimators in range(n_estimators_max, 0, -1):
        max_depth_current = max_depth
        memory_in_bytes = float('inf')  # Start with a high invalid memory usage

        while memory_in_bytes > target_memory * 1024 * 1024:  # Target memory in bytes
            # Train XGBoost with current parameters
            xgb = XGBClassifier(
                verbosity=2,
                n_estimators=n_estimators,
                max_depth=max_depth_current,
                learning_rate=0.1,
                random_state=42,
                #scale_pos_weight=1,
                #use_label_encoder=False,
                tree_method="auto",
                n_jobs=-1
            )
            xgb.fit(X_train, y_train)

            # Estimate memory usage
            memory_in_bytes = calculate_xgb_memory(xgb)

            # Evaluate model
            y_pred = xgb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"n_estimators: {n_estimators}, max_depth: {max_depth_current}, "
                  f"Accuracy: {accuracy:.4f}, Memory: {memory_in_bytes / (1024**2):.2f} MB")

            if memory_in_bytes > target_memory * 1024 * 1024 and max_depth_current > 1:
                max_depth_current -= 1  # Reduce depth to save memory
            else:
                break

        if accuracy > best_accuracy:
            print(f"Old values: n_estimators: {best_n_estimators}, max_depth: {best_max_depth}, "
              f"Accuracy: {best_accuracy:.4f},\n New values:"
              f"n_estimators: {n_estimators}, max_depth: {max_depth_current}, "
              f"Accuracy: {accuracy:.4f}")
            best_accuracy = accuracy
            best_max_depth = max_depth_current
            best_n_estimators = n_estimators
            best_xgb = xgb

    print(f"Best combination: n_estimators={best_n_estimators}, max_depth={best_max_depth}, "
          f"accuracy={best_accuracy:.4f}")
    return best_xgb, best_max_depth, best_n_estimators

def xgb_feature_importance(data_dir, target_param, target_value):
    # Results directory
    result_dir = ensure_results_directory("xgboost")

    # Start resource profiling in the background
    stop_event = threading.Event()
    profiler_thread = threading.Thread(target=profile_resource_usage, args=(1, result_dir, stop_event))
    profiler_thread.start()

    try:
        # Load image data
        print("Loading dataset...")
        X, y, class_names = load_images_from_folder(data_dir)
        X = X / 255.0  # Normalize pixel values
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

        # Split data
        print("Splitting dataset into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Train XGBoost
        xgb, max_depth, n_estimators = optimize_xgb(
            max_depth=60, n_estimators_max=9, target_memory=3.2,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )
        print("Initial XGBoost training complete.")

        # Feature importance
        print("Calculating feature importance...")
        importances = xgb.feature_importances_
        ranked_features = np.argsort(importances)[::-1]

        # Iterative feature selection
        print("Starting iterative feature selection...")
        selected_features = []
        accuracies = []
        history = []  # To store [Number of features, accuracy, size]

        for i, idx in enumerate(ranked_features):
            selected_features.append(idx)
            X_train_reduced = X_train[:, selected_features]
            X_test_reduced = X_test[:, selected_features]

            # Evaluate
            xgb_reduced = XGBClassifier(
                verbosity=2,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                random_state=42,
                #scale_pos_weight=1,
                #use_label_encoder=False,
                tree_method="auto",
                n_jobs=-1
            )
            xgb_reduced.fit(X_train_reduced, y_train)
            y_pred = xgb_reduced.predict(X_test_reduced)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            memory_in_bytes = calculate_xgb_memory(xgb)

            # Log progress
            print(f"Iteration {i+1}: {len(selected_features)} features, Accuracy = {accuracy:.4f}, Memory: {memory_in_bytes / (1024**2):.2f} MB")
            history.append([len(selected_features), accuracy,memory_in_bytes])

            # Check stopping condition
            if target_param == "accuracy" and accuracy >= target_value:
                print("Target accuracy achieved.")
                break
            if target_param == "sensor_count" and len(selected_features) >= target_value:
                print("Target sensor count achieved.")
                break

        # Save history
        print("Saving history...")
        history_path = os.path.join(result_dir, "feature_selection_history.csv")
        pd.DataFrame(history, columns=["NumberOfFeatures", "Accuracy", "Memory"]).to_csv(history_path, index=False)

        # Visualize history
        print("Generating visualization for feature selection history...")
        history_df = pd.DataFrame(history, columns=["Number of Features", "Accuracy", "Memory"])
        plt.figure(figsize=(10, 6))
        plt.plot(history_df["Number of Features"], history_df["Accuracy"], marker='o', label="Accuracy")
        plt.title("Accuracy vs. Number of Features")
        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(result_dir, "feature_selection_accuracy_plot.png")
        plt.savefig(plot_path)
        plt.show()

        # Save sensor map
        print("Saving selected sensor map...")
        save_sensor_map(result_dir, selected_features)

        # Save metrics
        print("Saving metrics...")
        metrics = classification_report(y_test, y_pred, output_dict=True)
        with open(os.path.join(result_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Results saved in {result_dir}")

    finally:
        # Stop resource profiling
        stop_event.set()
        profiler_thread.join()
        print("Resource profiling completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Feature Importance")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--target_param', type=str, choices=["accuracy", "sensor_count"], required=True)
    parser.add_argument('--target_value', type=float, required=True)
    args = parser.parse_args()

    print(f"Starting XGBoost Feature Importance with target: {args.target_param}={args.target_value}")
    xgb_feature_importance(args.data_dir, args.target_param, args.target_value)
