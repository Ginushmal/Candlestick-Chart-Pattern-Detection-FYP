import torch
import numpy as np
import matplotlib.patches as patches
from collections import Counter
from matplotlib import pyplot as plt
import mplfinance as mpf

import pandas as pd
import random

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union in 1D.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 2)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 2)
        box_format (str): midpoint/corners, if boxes (x,w) or (x1,x2)

    Returns:
        tensor: Intersection over union for all examples
    """
    
    if box_format ==  "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 1:2] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 1:2] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 1:2] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 1:2] / 2
        
    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_x2 = boxes_preds[..., 1:2]
        box2_x1 = boxes_labels[..., 0:1]
        box2_x2 = boxes_labels[..., 1:2]
        
    x1 = torch.max(box1_x1, box2_x1)
    x2 = torch.min(box1_x2, box2_x2)
    
    intersection = torch.clamp(x2 - x1, min=0)
    union = box1_x2 - box1_x1 + box2_x2 - box2_x1 - intersection
    
    return intersection / (union + 1e-6)
    
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x,w]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    
    # print ("bboxes in non_max_suppression--------------------------------", bboxes)     

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] # if class is not same
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    
    # print ("bboxes_after_nms in non_max_suppression--------------------------------", bboxes_after_nms)

    return bboxes_after_nms
    
    
def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=8
    ):
        """
        Calculates mean average precision 

        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x,w]
            true_boxes (list): Similar as pred_boxes except all the correct ones 
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes

        Returns:
            float: mAP value across all classes given a specific IoU threshold 
        """

        # list storing all AP for respective classes
        average_precisions = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)
            
            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                num_gts = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

        return sum(average_precisions) / len(average_precisions)








def plot_image(ohcl_data, boxes ):
    
    pattern_encoding = {'Double Top, Adam and Adam': 0, 'Triangle, symmetrical': 1, 'Double Bottom, Eve and Adam': 2, 'Head-and-shoulders top': 3, 'Double Bottom, Adam and Adam': 4, 'Head-and-shoulders bottom': 5, 'Flag, high and tight': 6, 'Cup with handle': 7}
    # Assuming pattern_encoding dictionary is available
    pattern_encoding_inv = pattern_encoding
    pattern_encoding_inv = {v: k for k, v in pattern_encoding_inv.items()}
    # Colors for the patterns
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'black', 'pink']
    
    
    
    # Transpose the tensor to make it [224, 4] and move to CPU
    tensor_transposed = ohcl_data.T.cpu()

    # Convert to a pandas DataFrame
    ohcl_data = pd.DataFrame(tensor_transposed.numpy(), columns=['Open', 'High', 'Low', 'Close'])
    
    # Create a candlestick plot using mplfinance
    ohlc_for_mpf = ohcl_data[['Open', 'High', 'Low', 'Close']].copy()
    ohlc_for_mpf.index = pd.to_datetime(ohcl_data.index)
    
    # Create the base plot (this returns a figure and axes)
    fig, axes = mpf.plot(ohlc_for_mpf, type='candle', style='charles', title=f'OHLC Chart with Patterns',
                         ylabel='Price', figsize=(12, 6), returnfig=True)  # Set figsize here
    
    ax = axes[0]  # Access the first (and only) axis object
    
    # Patterns data for the current instance
    # patters = boxes
    
    # Loop through the patterns and highlight them on the chart
    color_index = 0
    # for index, row in patters.iterrows():
    for box in boxes:   
        pattern_center = box[2]
        pattern_width = box[3]
        pattern_label = pattern_encoding_inv[box[0]]
        
        # Calculate pattern start and end positions (scaled for the width)
        pattern_start = (pattern_center - pattern_width/2) * len(ohlc_for_mpf)
        pattern_end = (pattern_center + pattern_width/2) * len(ohlc_for_mpf)
        
        # Add a vertical span (highlight the pattern) to the chart
        ax.axvspan(pattern_start, pattern_end, color=colors[color_index], alpha=0.2, label=pattern_label)
        color_index += 1

    # Customize the chart with grid, labels, and legend
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Patterns")
    
    # Show the chart
    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)
        
        # print("x : ",x)
        # print("lables :",labels)

        with torch.no_grad():
            predictions = model(x)

        # torch.set_printoptions(profile="full")
        # print("raw predictions : ", predictions)
        
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)
        # print("converted_preds : ", bboxes)

        for idx in range(batch_size):
            # print ("bboxes in get_bboxes--------------------------------", bboxes[idx])
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # print("box : ", box)
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes




def convert_cellboxes(predictions, S=7 , B=2, C=8):
    """
    convert the cell based baunding box cordinate values to the actual coordinate values relative the whole segment
    
    return :
    tensor of shape (batch_size, S, 8) where 8 is (class_prob, confidence, x, w)
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 14 ) # 7 x (8+(2 x 3))
    bboxes1 = predictions[..., 9:11]
    bboxes2 = predictions[..., 12:14]
    scores = torch.cat(
        (predictions[..., 8].unsqueeze(0), predictions[..., 11].unsqueeze(0)), dim=0
    ) # tenser of probablities of 2 bounding boxes in ech cell in two dimentions of a tenser
    best_box = scores.argmax(0).unsqueeze(-1) # get a tensor of 0 and 1 for which of the 2 bbox has the better score 
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    w_y = 1 / S * best_boxes[..., 1:2]
    converted_bboxes = torch.cat((x, w_y), dim=-1)
    predicted_class = predictions[..., :8].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 8], predictions[..., 11]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    
    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []
    # print("converted_pred reshaped : ", converted_pred)

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range( S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])