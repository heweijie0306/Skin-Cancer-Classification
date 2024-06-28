


import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random


# Load the CSV file
csv_path = Path("C:/Users/Ethan He/images/metadata.csv")
data = pd.read_csv(csv_path)

# Select one random data point
sample_data = data.sample(1)

# Retrieve the image path and all column labels
image_folder = Path("C:/Users/Ethan He/images")
image_path = None
labels = None

for idx, row in sample_data.iterrows():
    image_path = image_folder / row['filename']  # Replace 'filename' with the correct column name
    labels = row.drop('filename')  # Replace 'filename' with the correct column name

# Load the image and display it with all labels
img = Image.open(image_path)
plt.imshow(img)

# Create the title with all labels
title = ', '.join([f"{col}: {value}" for col, value in labels.items()])
plt.title(title)
plt.axis('off')
plt.show()

## data distribution
data = pd.read_csv('C:/Users/Ethan He/images/metadata.csv')
class_counts = data['diagnosis'].value_counts()
class_counts.plot(kind='bar')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Distribution of Classes in the Dataset')
plt.show()



