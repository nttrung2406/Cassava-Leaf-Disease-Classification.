import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
image_ids = train_df['image_id'].values
labels = train_df['label'].unique()

def displayLabel(train_df):
    label_counts = train_df['label'].value_counts()
    plt.bar(label_counts.index, label_counts.values)
    plt.xlabel('Label ID')
    plt.ylabel('Count')
    plt.title('Distribution of Label IDs')
    plt.show()

def plot_selected_images(train_df, labels):
    selected_images = []
    for label in labels:
        image_id = train_df[train_df['label'] == label]['image_id'].iloc[0]
        selected_images.append(image_id)
    
    fig, axes = plt.subplots(nrows=len(selected_images), figsize=(8, 8))
    
    for i, image_id in enumerate(selected_images):
        img_path = f'train_tfimages\\{image_id}'  # Assuming the images are stored in a directory named 'train_images'
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {labels[i]}')
    
    plt.tight_layout()
    plt.show()

plot_selected_images(train_df, labels)

data_directory = 'train_tfimages'
img = cv2.imread(os.path.join('train_tfimages', '6103.jpg'))
img.shape
