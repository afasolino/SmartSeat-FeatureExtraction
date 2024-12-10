
import argparse
import tensorflow as tf
import numpy as np
import os
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import logging
from utils import create_output_directory, setup_logging, restore_stdout, load_dataset, compute_dataset_statistics, create_dataset
from plots import plot_confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model file (e.g., best_model.h5)")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()
    #tf.config.set_visible_devices([], 'GPU')

    output_dir = create_output_directory(base_dir="result", train=False, name="perf")
    log_file = setup_logging(output_dir, log_filename="performance.log")
    
    try:
        all_file_paths, all_labels = load_dataset(args.dataset_dir)
        compute_dataset_statistics(all_labels, os.path.join(output_dir,"metrics"))

        _, test_files, _, test_labels = train_test_split(all_file_paths, all_labels, test_size=0.9, random_state=42, stratify=all_labels)
        test_dataset = create_dataset(test_files, test_labels, batch_size=args.batch_size, shuffle=False, augment=False)

        print("Loading model:")
        model = tf.keras.models.load_model(args.model_path)
        class_names = sorted(os.listdir(args.dataset_dir))

        print("Evaluating the model:")
        restore_stdout()
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
        setup_logging(output_dir, log_filename="performance.log")
        logging.info(f"Test Loss: {test_loss:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        print("Generating predictions:")
        y_pred, y_true = [], []
        for images, labels in tqdm(test_dataset, desc="Predicting"):
            y_pred.append(np.argmax(model.predict(images, verbose=0), axis=-1))
            y_true.append(labels.numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        print("Saving metrics:")
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        confusion = confusion_matrix(y_true, y_pred)

        pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "metrics", "results.csv"))
        pd.DataFrame(confusion).to_csv(os.path.join(output_dir, "metrics", "confusion_matrix.csv"))
        plot_confusion_matrix(confusion, class_names, os.path.join(output_dir, "plots", "confusion_matrix.png"))

        logging.info("Evaluation complete. Outputs saved.")
        print(f"Evaluation complete. Outputs saved to {output_dir}")

    finally:
        restore_stdout()

if __name__ == "__main__":
    main()
