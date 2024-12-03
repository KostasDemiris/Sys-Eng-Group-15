import numpy as np
import torch
import torch.nn as nn
import torchvision
import Model
from Crosswalk_dataset import CrosswalkDataset


class BasicDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression_loss = torchvision.ops.generalized_box_iou_loss
        # Pretty good for bounding box regression - not for Rotated bounding boxes though, will have to update sometime
        self.classification_loss = nn.CrossEntropyLoss

    def forward(self, pred_bbox, targ_bbox, pred_lab, targ_lab):
        regressive_loss = self.regression_loss(pred_bbox, targ_bbox)
        label_loss = self.classification_loss(pred_lab, targ_lab)

        return regressive_loss + label_loss


# Maybe should move out of training - do so when we actually create a post_processing module
def post_processing(bbox_pred, label_pred, confidence_threshold=0.8, iou_threshold=0.3):
    # Confidence threshold is the minimum confidence of the model that there a crosswalk at a point
    # iou_threshold is the minimum number of (intersection/union) for two predictions to be considered the same pred.

    scores = label_pred[:, :, 1]  # Reminder that 1 is Crosswalk Label's confidence
    print(np.shape(scores), "scores")

    confident_predictions = scores >= confidence_threshold  # Removes background predictions
    kept_bbox = bbox_pred[confident_predictions]
    kept_labels = scores[confident_predictions]

    non_overlapping_boxes = torchvision.ops.nms(kept_bbox, kept_labels, iou_threshold)
    # nms --> Non-maximum suppression, used to remove overlapping boxes that pass
    # the min threshold to be considered the same box

    final_bboxes = kept_bbox[non_overlapping_boxes]
    final_labels = kept_labels[non_overlapping_boxes]

    return final_bboxes, final_labels


def collate_function(batch):
    images = []
    output_data = []

    for item in batch:
        images.append(item[0])  # img
        output_data.append(item[1])  # training data (labels, boxes)

    batched_images = torch.stack(images)
    return batched_images, output_data


def basic_train_model(model_to_train, dataset, epoch_number=25, loss_func=BasicDetectionLoss):
    optimiser = torch.optim.Adam(model_to_train.parameters())
    criteria = loss_func()

    # We need a batch function because there can be a variable number of bounding boxes in the training data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_function)

    # Main training loop
    for epoch in range(epoch_number):
        model_to_train.train()
        running_loss = 0.0

        for images, ground_truth in dataloader:
            optimiser.zero_grad()
            gt_class_labels, gt_bounding_boxes = ground_truth[0]

            pred_bbox, pred_labels = model(images)
            print(np.shape(pred_bbox), np.shape(pred_labels))
            nms_bbox, nms_labels = post_processing(pred_bbox, pred_labels)
            print("DONE W MODEL")
            print((np.shape(nms_bbox)), np.shape(nms_labels), "post-nms")
            print(np.shape(gt_bounding_boxes), np.shape(gt_class_labels))
            computed_loss = criteria(gt_bounding_boxes, gt_class_labels, nms_bbox, nms_labels)
            computed_loss.backwards()

            optimiser.step()
            running_loss += computed_loss

        print(f"Epoch [{epoch + 1}/{epoch_number}], Loss: {running_loss / len(dataloader)}")

    # Don't really have to return it, but why not
    return model_to_train


model = Model.BasicCrosswalkDetector()
crosswalk_dataset = CrosswalkDataset("Crosswalk.v7-crosswalk-t3.tensorflow/train/_annotations.csv",
                                     "Crosswalk.v7-crosswalk-t3.tensorflow/train")

model = basic_train_model(model, crosswalk_dataset)
