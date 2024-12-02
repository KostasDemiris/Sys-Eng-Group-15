import torch
import pandas as pd
import numpy as np
from PIL import Image
# from sklearn import train_test_split
# This is a CSV file in the format (Filename, width, height, class, xmin, ymin, xmax, ymax). Basically bounding boxes.
# annotations = pd.read_csv("Crosswalk.v7-crosswalk-t3.tensorflow/train/_annotations.csv") - moved to inside init


class CrossWalkDataset:
    def __init__(self, annotation_path, image_path, transform=None):
        self.annotations = pd.read_csv(annotation_path)
        self.image_dir = image_path
        self.transform = None  # In case later one we want to do normalisation etc. --> Someone else will look at this

        self.unique_labels = self.annotations["class"].unique()
        # There might be a more efficient method to do this -- come back to
        self.type_mapping = {type_value: idx for idx, type_value in enumerate(self.unique_labels)}
        # Labels have been converted to numerical class labels mapped by type mapping - for tensor conversion

        self.image_data = []
        self.bounding_boxes = []  # Not every image has a bbox, and some have multiple
        self.labels = []  # One-to-One mapping with bounding boxes by the way

        self.process_annotations()

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image = self.image_data[index]
        bounding_boxes = self.bounding_boxes[index]
        class_labels = self.labels[index]

        tensor_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        # From (H, W, C) to (C, H, W)  --> This is the format that pytorch uses.
        tensor_bbox = [torch.tensor(box, dtype=torch.float32) for box in bounding_boxes]
        tensor_label = [torch.tensor(label, dtype=torch.float32) for label in class_labels]

        if self.transform:
            tensor_image = self.transform(tensor_image)

        return tensor_image, tensor_bbox, tensor_label

    def process_annotations(self):
        for filename, group in self.annotations.groupby('filename'):
            completed_image_path = f"{self.image_dir}/{filename}"
            image = Image.open(completed_image_path)
            image_array = np.array(image)
            entity_annotations = []

            for _, row in group.iterrows():
                # We have to iterate through multiple datapoints, since most images have multiple objects in them

                # We can add additional classes in here - come back to later
                if row['class'] == "ZebraStyle":
                    numerical_class = self.type_mapping[row['class']]
                    entity_annotations.append([numerical_class, (row['xmin'], row['ymin'], row['xmax'], row['ymax'])])

            self.image_data.append(image_array)
            self.labels.append([entity_box for _, entity_box in entity_annotations])
            self.bounding_boxes.append([label for label, _ in entity_annotations])


dataset = CrossWalkDataset("Crosswalk.v7-crosswalk-t3.tensorflow/train/_annotations.csv",
                           "Crosswalk.v7-crosswalk-t3.tensorflow/train")

img, bounding_boxes, class_label = dataset[120]

print(bounding_boxes)
print(class_label)
print(np.shape(img))
