# Deep Learning Model for Classification and Analysis

This repository contains scripts for training, evaluating, and analyzing deep learning models for image classification tasks. The implementation includes a modified SqueezeNet architecture, custom metrics, and utility functions for preprocessing, data augmentation, and evaluation.

---

## Features

- **Modified SqueezeNet Model**: Tailored architecture with L2 regularization and dropout.
- **Custom Metric**: Includes an F1-score implementation for model evaluation.
- **Data Augmentation**: Options for Gaussian blur, median filtering, and random displacements.
- **Training and Evaluation Pipelines**: Modular scripts for efficient workflow.
- **Visualization**: Confusion matrices and learning curves plotting.

---

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Scripts Overview

### `scripts/model_def.py`
Contains the implementation of the modified SqueezeNet architecture and the F1-score metric.

### `scripts/train_tc.py`
Script for training the model:
- Supports custom datasets organized by class folders.
- Generates learning curves and evaluation metrics.

#### Example Usage:
```bash
python scripts/train_tc.py --dataset_dir ./data --num_epochs 50 --batch_size 32 --learning_rate 0.0001 --data_aug
```

### `scripts/performance.py` and `scripts/updated_performance.py`
Evaluate a pre-trained model on a test dataset, generating performance metrics and confusion matrices.

#### Example Usage:
```bash
python scripts/performance.py --model_path ./path/to/model.h5 --dataset_dir ./data --batch_size 16
```

### `scripts/utils.py`
Utility functions for logging, dataset creation, and saving results.

### `scripts/plots.py`
Functions for plotting learning curves and confusion matrices.

---

## Requirements

- Python 3.7+
- TensorFlow >= 2.4
- Additional dependencies listed in `requirements.txt`.

To install them, run:
```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Your dataset should be organized as follows:
```
data/
├── class_1/
│   ├── img1.jpg
│   ├── img2.jpg
├── class_2/
│   ├── img1.jpg
│   ├── img2.jpg
...
```

---

## Outputs

Each run creates an output folder with the following structure:
```
result/YYYY.MM.DD_HH.MM.SS/
├── data_augmentation_samples/
├── metrics/
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   ├── results.csv
├── models/
│   ├── best_model.h5
│   ├── final_model.h5
├── plots/
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## Contact

For questions or suggestions, please open an issue in this repository.
