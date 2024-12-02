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

    scores = label_pred[:, 1]  # Reminder that 1 is Crosswalk Label's confidence

    confident_predictions = scores >= confidence_threshold  # Removes background predictions
    kept_bbox = bbox_pred[confident_predictions]
    kept_labels = scores[confident_predictions]

    non_overlapping_boxes = torchvision.ops.nms(kept_bbox, kept_labels, iou_threshold)
    # nms --> Non-maximum suppression, used to remove overlapping boxes that pass
    # the min threshold to be considered the same box

    final_bboxes = kept_bbox[non_overlapping_boxes]
    final_labels = kept_labels[non_overlapping_boxes]

    return final_bboxes, final_labels


def basic_train_model(model_to_train, dataset, epoch_number=25, loss_func=BasicDetectionLoss):
    optimiser = torch.optim.Adam(model_to_train.parameters())
    criteria = loss_func

    # We need a batch function because there can be a variable number of bounding boxes in the training data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=
        lambda batch: tuple(zip(*batch)))

    # Main training loop
    for epoch in range(epoch_number):
        model_to_train.train()

        for image, training_data in dataloader:
            pass

    # Don't really have to return it, but why not
    return model_to_train


model = Model.BasicCrosswalkDetector()
crosswalk_dataset = CrosswalkDataset("Crosswalk.v7-crosswalk-t3.tensorflow/train/_annotations.csv",
                                     "Crosswalk.v7-crosswalk-t3.tensorflow/train")
