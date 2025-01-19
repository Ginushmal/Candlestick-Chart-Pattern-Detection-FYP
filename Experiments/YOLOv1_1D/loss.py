# Implementation of the loss function for YOLO v1 changed for 1D input.

import torch
import torch.nn as nn
from utils import intersection_over_union

class Yolo1DLoss(nn.Module):
    """
    Calculate the loss for yolo 1D (v1) model
    """

    def __init__(self, S=7, B=2,C=8):
        super(Yolo1DLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of ts data (default 7),
        B is number of boxes (ranges in 1d case) (default 2),
        C is number of classes (default 8),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward( self,predictions, target):


        # predictions are shaped (BATCH_SIZE, S(C+B*3) when inputted
        predictions = predictions.reshape(-1, self.S, self.C + self.B * 3) # for each batch element, C number of classes and B number of boxes times 3 (probability , x, w)
        iou_b1 = intersection_over_union(predictions[..., 9:11], target[..., 9:11])
        iou_b2 = intersection_over_union(predictions[..., 12:14], target[..., 9:11])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 8].unsqueeze(2)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 12:14] # if 1st box is the best box
                + (1 - bestbox) * predictions[..., 9:11] # if 2nd box is the best box
            )
        )

        box_targets = exists_box * target[..., 9:11]

        # Take sqrt of width of boxes 
        box_predictions[..., 1:2] = torch.sign(box_predictions[..., 1:2]) * torch.sqrt(
            torch.abs(box_predictions[..., 1:2] + 1e-6)
        )
        box_targets[..., 1:2] = torch.sqrt(box_targets[..., 1:2])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 11:12] + (1 - bestbox) * predictions[..., 8:9]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 8:9]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 8:9], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 8:9], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 11:12], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 8:9], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :8], end_dim=-2,),
            torch.flatten(exists_box * target[..., :8], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
            