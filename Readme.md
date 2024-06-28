# ISIC Skin Lesion Classification
This repository contains the code for training and evaluating a skin lesion classification model using the ISIC dataset. The model is based on a ResNet50 architecture, fine-tuned for the classification task. The training process employs K-Fold Cross-Validation to evaluate the model's performance.

## Getting Started
## Prerequisites
Install the required packages:
pip install -r requirements.txt
## Dataset
Download the ISIC dataset from the official website. You should have a CSV file containing the metadata and the image files in a separate directory.

## Usage
Training and Evaluation
Run the main function in the provided script to train and evaluate the model. Ensure that the csv_path and images_path variables are set to the correct paths for your dataset.
``` bash
python skin_classification.py
```

## Results
### The script will output the following:

Training and testing confusion matrices as images: overall_train_conf_mat.png, overall_test_conf_mat.png
Overall per class training and testing accuracy as images: overall_per_class_train.png, overall_per_class_test.png
t-SNE visualization of train and test features: train_tsne_plot.png, test_tsne_plot.png
ROC curve: ROC Curve.png
ROC curve with sensitivity and specificity: ROC_Sensitivity_Specificity.png
## Code Structure

ISICDataset: A custom PyTorch dataset class for loading and processing the ISIC dataset.
train: Function for training the model.
test: Function for evaluating the model on the test set.
get_stats: Function for computing the confusion matrix and per-class accuracy.
visulization: Function for generating and saving the confusion matrices and per-class accuracy plots.
plot_tsne: Function for generating and saving t-SNE visualizations of the train and test features.
ROC: Function for generating and saving ROC curves.
plot_sensitivity_specificity_auc: Function for generating and saving ROC curves with sensitivity and specificity.
main: The main function for running the training and evaluation process.

# baseline
This file contains the baseline model used in skin_classification task

# Distribution
This file plot the distribution of original ISIC dataset

## Usage
``` bash
python Distribution.py
```

# example
This file outputs several images with their annotations

## Usage
``` bash
python example.py
```