import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
csv_path = 'C:/Users/Ethan He/images/metadata.csv'

df = pd.read_csv(csv_path, low_memory=False)

# Create a figure to hold the subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop through the first 3 rows of the CSV file
for i in range(10):
    # Read the image
    image_name = df.iloc[i, 0] + ".jpg"
    image_path = os.path.join("C:/Users/Ethan He/images/", image_name)
    print(f"Trying to open image at path: {image_path}")
    image = Image.open(image_path)

    # Read all columns
    label = df.iloc[i].drop('isic_id').to_string(index=False).strip()

    # Add the labels to the image
    font = ImageFont.truetype("arial.ttf", 100)
    draw = ImageDraw.Draw(image)
    draw.text((5, 20), label, font=font, fill='white')

    # Create a new figure and plot the image
    plt.figure()
    plt.imshow(image)

    # Display the image in a separate window
    plt.show()