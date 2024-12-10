# random_forest_feature_importance.py
import os
import time
import json
import numpy as np
import pandas as pd
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import (
    load_images_from_folder,
    ensure_results_directory,
    show_bar_chart,
    save_training_plot,
    save_sensor_map,
    profile_resource_usage
)
import argparse
import sys
import matplotlib.pyplot as plt

def optimize_rf(max_depth, n_estimators_max, target_memory, X_train, X_test, y_train, y_test):
    print("Optimizing Random Forest to fit within memory constraints...")
    max_depth_start=max_depth
    memory_in_bytes = float('inf')  # Start with an invalid high memory usage
    accuracies = []
    best_accuracy=0
    best_max_depth=0
    best_n_estimators=0

    for n_estimators in range(n_estimators_max, 0, -1):
        max_depth=max_depth_start
        memory_in_bytes = float('inf')  # Start with an invalid high memory usage
        accuracies = []
        while memory_in_bytes > target_memory * 1024 * 1024:  # Memory limit of 1 MB
            # Train Random Forest with current parameters
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced",     # Adjust weights to balance class frequencies
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            # Check memory usage
            memory_in_bytes = calculate_rf_memory(rf)
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, "
                  f"Accuracy: {accuracy:.4f}, Memory: {memory_in_bytes / (1024**2):.2f} MB")

            # Adjust parameters
            if memory_in_bytes > target_memory * 1024 * 1024:  # Still above memory limit
                if max_depth > 1:  # Reduce max_depth first
                    max_depth -= 1
                else:
                    print("Cannot reduce further without compromising model functionality.")
                    break
        
        if accuracy>best_accuracy:
            best_accuracy=accuracy
            best_max_depth=max_depth
            best_n_estimators=n_estimators
            best_rf=rf
            actual_memory=memory_in_bytes
    print(f"best combination: n_estimators: {best_n_estimators}, max_depth: {best_max_depth}, acc: {best_accuracy:.4f}, Memory: {actual_memory / (1024**2):.2f} MB")
    return best_rf, best_max_depth, best_n_estimators

def calculate_rf_memory(rf_model):
    """
    Calculate the total memory requirement of a trained Random Forest model.

    Args:
        rf_model (RandomForestClassifier): Trained Random Forest model.

    Returns:
        int: Total memory requirement in bytes.
    """
    total_memory = 0
    
    # Memory for each tree
    for tree in rf_model.estimators_:
        tree_memory = sys.getsizeof(tree.tree_)  # Access the underlying tree structure
        total_memory += tree_memory
        
        # Node and value memory
        total_memory += tree.tree_.capacity * (
            sys.getsizeof(tree.tree_.threshold) +
            sys.getsizeof(tree.tree_.children_left) +
            sys.getsizeof(tree.tree_.children_right) +
            sys.getsizeof(tree.tree_.impurity) +
            sys.getsizeof(tree.tree_.n_node_samples)
        )
        
        # Leaf values
        total_memory += sys.getsizeof(tree.tree_.value)


    return total_memory

def random_forest_feature_importance(data_dir, target_param, target_value):
    # Results directory
    result_dir = ensure_results_directory("random_forest")

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

        # Train Random Forest
        [rf, max_depth, n_estimators] = optimize_rf (max_depth=20, n_estimators_max=10, target_memory=5,X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        print("Initial Random Forest training complete.")

        # Feature importance
        print("Calculating feature importance...")
        importances = rf.feature_importances_
        ranked_features = np.argsort(importances)[::-1]

        print("Calculating Trained Random forrest memory requirement...")
        memory_in_bytes = calculate_rf_memory(rf)
        print(f"Total memory used by the model: {memory_in_bytes / (1024**2):.2f} MB")

        # Iterative feature selection
        print("Starting iterative feature selection...")
        selected_features = []
        accuracies = []
        history = []  # To store [Number of features, accuracy, size]
        start_time = time.time()

        for i, idx in enumerate(ranked_features):
            selected_features.append(idx)
            X_train_reduced = X_train[:, selected_features]
            X_test_reduced = X_test[:, selected_features]

            # Evaluate
            rf_reduced = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced",     # Adjust weights to balance class frequencies
                n_jobs=-1
            )
            rf_reduced.fit(X_train_reduced, y_train)
            y_pred = rf_reduced.predict(X_test_reduced)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            memory_in_bytes = calculate_rf_memory(rf_reduced)

            # Log progress
            print(f"Iteration {i+1}: {len(selected_features)} features, Accuracy = {accuracy:.4f}, memory used = {memory_in_bytes / (1024**2):.2f} MB")
            history.append([len(selected_features), accuracy, memory_in_bytes / (1024**2)])  # Append to history

            # Check stopping condition
            if target_param == "accuracy" and accuracy >= target_value:
                print("Target accuracy achieved.")
                break
            if target_param == "sensor_count" and len(selected_features) >= target_value:
                print("Target sensor count achieved.")
                break

        # Save the history
        print("Saving history...")
        history_path = os.path.join(result_dir, "feature_selection_history.csv")
        pd.DataFrame(history, columns=["NumberOfFeatures", "Accuracy", "SizeMB"]).to_csv(history_path, index=False)
        print(f"Feature selection history saved at {history_path}")

        # Visualize the history
        print("Generating visualization for feature selection history...")

        history_df = pd.DataFrame(history, columns=["Number of Features", "Accuracy", "Size (MB)"])

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Accuracy vs. Number of Features
        ax1.plot(history_df["Number of Features"], history_df["Accuracy"], marker='o', color='blue', label="Accuracy")
        ax1.set_xlabel("Number of Features")
        ax1.set_ylabel("Accuracy", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Add Memory Usage on the same graph with a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(history_df["Number of Features"], history_df["Size (MB)"], marker='x', color='red', label="Memory Usage")
        ax2.set_ylabel("Size (MB)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Add titles and legend
        fig.suptitle("Feature Selection: Accuracy and Memory Usage vs. Number of Features")
        ax1.grid(True)
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        # Save the plot
        plot_path = os.path.join(result_dir, "feature_selection_accuracy_memory_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Feature selection accuracy and memory usage plot saved at {plot_path}")

        # Save training plot
        print("Saving training progress plot...")
        save_training_plot(result_dir, accuracies, "Accuracy Over Iterations", "accuracy_iterations.png")

        # Save sensor map
        print("Saving selected sensor map...")
        save_sensor_map(result_dir, selected_features)

        # Save metrics
        print("Saving metrics...")
        metrics = classification_report(y_test, y_pred, output_dict=True)
        with open(os.path.join(result_dir, "metrics.json"), "w") as f:
            json.dump({
                "selected_sensors": [int(s) for s in selected_features],  # Convert numpy.int64 to int
                "accuracy": float(accuracies[-1]),  # Ensure it's a float
                "metrics": metrics,
                "class_names": class_names
            }, f, indent=4)
        print(f"Results saved in {result_dir}")

    finally:
        # Stop resource profiling
        stop_event.set()
        profiler_thread.join()
        print("Resource profiling completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Feature Importance")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--target_param', type=str, choices=["accuracy", "sensor_count"], required=True)
    parser.add_argument('--target_value', type=float, required=True)
    args = parser.parse_args()

    print(f"Starting Random Forest Feature Importance with target: {args.target_param}={args.target_value}")
    random_forest_feature_importance(args.data_dir, args.target_param, args.target_value)
