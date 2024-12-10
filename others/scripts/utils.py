# utils.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import threading
import time
import psutil
import GPUtil

def load_images_from_folder(folder_path):
    """Load images and labels from the dataset folder structure."""
    data = []
    labels = []
    class_names = sorted(os.listdir(folder_path))
    class_counts = []  # To track the number of images per class

    print("Loading images by class:")
    for class_idx, class_name in enumerate(tqdm(class_names, desc="Classes Progress", ncols=80)):
        class_path = os.path.join(folder_path, class_name)
        count = 0
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith(".png"):
                    file_path = os.path.join(class_path, file_name)
                    image = Image.open(file_path).convert("L")  # Ensure grayscale
                    data.append(np.array(image).flatten())  # Flatten to 1D
                    labels.append(class_idx)
                    count += 1
        class_counts.append(count)
        print(f" Class '{class_name}': {count} images loaded.")

    print(f"\nSummary of images loaded:")
    max_class_name_len = max(len(name) for name in class_names)
    for class_name, count in zip(class_names, class_counts):
        print(f"{class_name.ljust(max_class_name_len)}: {count} images")

    return np.array(data), np.array(labels), class_names

def ensure_results_directory(method_name):
    """Create a timestamped directory for results."""
    timestamp = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    result_dir = os.path.join("results", f"{timestamp}_{method_name}")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)  # Subfolder for plots
    return result_dir

def show_bar_chart(data, labels, title):
    """Display a bar chart interactively."""
    plt.figure(figsize=(10, 6))
    plt.bar(labels, data, color='skyblue')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    plt.show()
    plt.pause(0.001)  # Pause briefly to allow the plot to update
    plt.ioff()  # Disable interactive mode (optional, if only temporary)

def save_training_plot(results_dir, values, title, filename, xlabel="Iterations", ylabel="Value"):
    """Save training-related plots."""
    plt.figure(figsize=(10, 6))
    plt.plot(values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    output_path = os.path.join(results_dir, "plots", filename)
    plt.savefig(output_path)
    plt.close()

def save_sensor_map(results_dir, selected_sensors, grid_shape=(64, 158)):
    """Save a heatmap of selected sensors (pixels)."""
    sensor_map = np.zeros(grid_shape)
    for sensor in selected_sensors:
        x, y = divmod(sensor, grid_shape[1])
        sensor_map[x, y] = 1
    plt.figure(figsize=(12, 8))
    plt.imshow(sensor_map, cmap="viridis", aspect="auto")
    plt.title("Map of Selected Sensors")
    plt.colorbar(label="Importance")
    output_path = os.path.join(results_dir, "sensor_map.png")
    plt.savefig(output_path)
    plt.close()

def log_resource_usage():
    """Log CPU, RAM, and GPU usage."""
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # Convert bytes to GB
    ram_total = psutil.virtual_memory().total / (1024 ** 3)
    ram_percent = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()

    gpu_info = [
        {
            "id": gpu.id,
            "name": gpu.name,
            "memory_used": gpu.memoryUsed,
            "memory_total": gpu.memoryTotal,
            "load": gpu.load * 100
        }
        for gpu in gpus
    ] if gpus else []

    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage": cpu_usage,
        "ram_usage_gb": ram_usage,
        "ram_total_gb": ram_total,
        "ram_percent": ram_percent,
        "gpu_info": gpu_info
    }


def profile_resource_usage(interval=1, result_dir="results", stop_event=None):
    """Continuously profile resource usage and save logs."""
    log_file = os.path.join(result_dir, "resource_usage_log.jsonl")
    print(f"Resource profiling started. Logs will be saved to: {log_file}")

    with open(log_file, "w") as f:
        while not stop_event.is_set():
            usage = log_resource_usage()
            json.dump(usage, f)
            f.write("\n")
            time.sleep(interval)  # Log every `interval` seconds

    print("Resource profiling stopped.")