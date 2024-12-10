# utils.py
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

class TeeLogger:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_logging(output_dir, log_filename="training.log"):
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_filename)

    # Set up logging to a file and console
    sys.stdout = TeeLogger(log_file_path)
    sys.stderr = sys.stdout  # Redirect errors to the same log file

    return log_file_path

def restore_stdout():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Create output directory and subdirectories for augmented samples
def create_output_directory(base_dir="result", train=True,name="train"):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_dir = os.path.join(base_dir, f"{timestamp}_{name}")
    os.makedirs(output_dir, exist_ok=True)
    if train:
        os.makedirs(os.path.join(output_dir, "data_augmentation_samples/blurred"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data_augmentation_samples/mediated"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data_augmentation_samples/displaced"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    return output_dir

class TeeLogger:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def save_augmented_sample(output_dir, category, image, class_label, idx):
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
    class_label = tf.strings.as_string(class_label)
    filename = tf.strings.join([output_dir, "/", category, "/class_", class_label, "_example_", tf.strings.as_string(idx), ".png"])
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    png_encoded = tf.image.encode_png(image)
    tf.io.write_file(filename, png_encoded)

def save_augmented_image_per_class(dataset, output_dir, category):
    saved_classes = set()
    for image, label in dataset.take(len(dataset)):
        class_label = label.numpy()
        if class_label not in saved_classes:
            save_augmented_sample(os.path.join(output_dir, "data_augmentation_samples"), category, image, class_label, idx=0)
            saved_classes.add(class_label)

def load_dataset(dataset_dir):
    class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    all_file_paths, all_labels = [], []
    print("Loading dataset:")
    for label, class_folder in tqdm(enumerate(class_folders), total=len(class_folders), desc="Processing classes", unit="class"):
        class_path = os.path.join(dataset_dir, class_folder)
        file_paths = [os.path.join(class_path, f) for f in os.listdir(class_path)]
        all_file_paths.extend(file_paths)
        all_labels.extend([label] * len(file_paths))
    return np.array(all_file_paths), np.array(all_labels)

def compute_dataset_statistics(labels, output_dir):
    class_counts = pd.Series(labels).value_counts()
    stats = pd.DataFrame({
        "Class": class_counts.index,
        "Count": class_counts.values
    })
    stats.to_csv(os.path.join(output_dir, "dataset_statistics.csv"), index=False)


# Create a tf.data.Dataset and preprocess images
def create_dataset(file_paths, labels, batch_size=16, shuffle=False, augment=False,
                   blur_pct=0.4, median_pct=0.3, displacement_pct=0.3, 
                   output_dir=None):
    """
    Create a TensorFlow dataset with original images and specified percentages of augmented images.
    
    Includes features like dynamic resizing, saving augmented samples, and batching.

    Parameters:
        file_paths (list): List of image file paths.
        labels (list): List of labels corresponding to the images.
        batch_size (int): Number of images per batch.
        shuffle (bool): Whether to shuffle the dataset.
        blur_pct (float): Percentage of images to apply Gaussian blur.
        median_pct (float): Percentage of images to apply median filter.
        displacement_pct (float): Percentage of images to apply displacement.
        output_dir (str): Directory to save augmented samples (optional).

    Returns:
        tf.data.Dataset: Combined dataset of original and augmented images.
    """
    # Determine the target shape from the first image dynamically
    def count_images_in_dataset(dataset):
        count = dataset.cardinality().numpy()
        return count
    def get_image_shape(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)
        return tf.shape(image)[:2]

    target_shape = tf.numpy_function(
        func=lambda x: get_image_shape(x).numpy(),
        inp=[file_paths[0]],
        Tout=tf.int32
    )

    # Image preprocessing function
    def preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [target_shape[0], target_shape[1]])
        return image, label
    # Augmentation functions with error handling
    def blur_image(image):
        def apply_blur(img):
            try:
                blurred = scipy.ndimage.gaussian_filter(img, sigma=1.0)
                return tf.convert_to_tensor(blurred, dtype=tf.float32)
            except Exception as e:
                print(f"blur error\n")
                logging.error(f"Gaussian blur error: {e}")
                return tf.convert_to_tensor(img, dtype=tf.float32)

        blurred_image = tf.py_function(apply_blur, [image], tf.float32)
        blurred_image.set_shape(image.shape)
        return blurred_image

    def median_filter_image(image):
        def apply_median(img):
            kernel_size = tf.random.uniform([], minval=2, maxval=5, dtype=tf.int32)
            try:
                filtered = scipy.ndimage.median_filter(img, size=(kernel_size, kernel_size, 1))
                return tf.convert_to_tensor(filtered, dtype=tf.float32)
            except Exception as e:
                print(f"median error\n")
                logging.error(f"Median filter error: {e}")
                return tf.convert_to_tensor(img, dtype=tf.float32)

        median_filtered_image = tf.py_function(apply_median, [image], tf.float32)
        median_filtered_image.set_shape(image.shape)
        return median_filtered_image

    def displace_image(image):
        try:
            dx, dy = tf.random.uniform([], -4, 4, dtype=tf.int32), tf.random.uniform([], -4, 4, dtype=tf.int32)
            displaced_image = tf.roll(image, shift=[dx, dy], axis=[0, 1])
            return displaced_image
        except Exception as e:
            print(f"displace error\n")
            logging.error(f"Displacement error: {e}")
            return image

    # Count images in the dataset
    def count_images_in_dataset(dataset):
        count = dataset.cardinality().numpy()
        return count

    # Create augmentation datasets
    def augment_images(dataset, augment_fn, augment_pct, aug_type):
        augment_count = int(len(file_paths) * augment_pct)
        augmented_dataset = dataset.take(augment_count).map(
            lambda x, y: (augment_fn(x), y)
        )
        return augmented_dataset

    # Build the original dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)


    # Augmentation datasets
    blurred_dataset = augment_images(dataset, blur_image, blur_pct, "blurred")
    median_dataset = augment_images(dataset, median_filter_image, median_pct, "median")
    displaced_dataset = augment_images(dataset, displace_image, displacement_pct, "displaced")

    # Combine original and augmented datasets
    final_dataset = dataset
    if augment:
        final_dataset = final_dataset.concatenate(blurred_dataset)
        final_dataset = final_dataset.concatenate(median_dataset)
        final_dataset = final_dataset.concatenate(displaced_dataset)
        num_images = count_images_in_dataset(final_dataset)
        print(f"\nDataset augmented, total images:{num_images}\n")
        save_augmented_image_per_class(blurred_dataset, output_dir, "blurred")
        save_augmented_image_per_class(median_dataset, output_dir, "mediated")
        save_augmented_image_per_class(displaced_dataset, output_dir, "displaced")

    else:
        num_images = count_images_in_dataset(final_dataset)
        print(f"\nDataset not augmented, total images:{num_images}\n")

    # Shuffle dataset if required
    if shuffle:
        final_dataset = final_dataset.shuffle(buffer_size=num_images)

    # Batch, shuffle, and prefetch for training
    final_dataset = final_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return final_dataset

def save_history_to_json(history, output_dir, filename="training_history.json"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert history.history to JSON format and save it
    history_path = os.path.join(output_dir, filename)
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=4)
    
    print(f"Training history saved to {history_path}")

def parse_args_with_json(output_dir):
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Train a modified SqueezeNet on a custom dataset.")
    parser.add_argument('--dataset_dir', type=str, help="Path to the dataset directory")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for training (default: 0.0001)")
    parser.add_argument('--l_reg', type=float, default=0.005, help="L-regularization factor (default: 0.005)")
    parser.add_argument('--blur_pct', type=float, default=0.3, help="Percentage of samples to blur (default: 0.1)")
    parser.add_argument('--median_pct', type=float, default=0.3, help="Percentage of samples to apply median filter (default: 0.3)")
    parser.add_argument('--displacement_pct', type=float, default=0.3, help="Percentage of samples to displace (default: 0.3)")
    parser.add_argument('--data_aug', action='store_true', help="Enable data augmentation (default: False)")
    parser.add_argument('--verbosity', type=int, default=3, help="Set verbosity level (default: 3)")
    parser.add_argument('--patience', type=int, default=40, help="Set early stopping patience (default: 40)")
    parser.add_argument('--json_file', type=str, help="Path to a JSON file for parameters")

    # Parse initial arguments
    args = parser.parse_args()

    # Load JSON file if provided
    if args.json_file:
        with open(args.json_file, 'r') as f:
            json_params = json.load(f)
        for key, value in json_params.items():
            if getattr(args, key, None) is None:  # Only update if not set in CLI
                setattr(args, key, value)

    # Validate required parameters
    if not args.dataset_dir:
        raise ValueError("The dataset directory (--dataset_dir) must be specified either in the JSON file or as a command-line argument.")


    config = {
        "dataset_dir": args.dataset_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "l_reg": args.l_reg,
        "blur_pct": args.blur_pct,
        "median_pct": args.median_pct,
        "displacement_pct": args.displacement_pct,
        "data_aug": args.data_aug,
        "patience": args.patience,
        "verbosity": args.verbosity
    }

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")

    return args


