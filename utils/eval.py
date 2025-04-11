import pandas as pd


def intersection_over_union(start1, end1, start2, end2):
    """
    Compute Intersection over Union (IoU) between two date ranges.
    """
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    overlap = max(0, (earliest_end - latest_start).days + 1)
    union = (end1 - start1).days + (end2 - start2).days + 2 - overlap
    return overlap / union if union > 0 else 0  # Avoid division by zero

def mean_abselute_error(start1, end1, start2, end2):
    """
    Compute Mean Absolute Error (MAE) between two date ranges.
    """
    # check if start or end are NAT 
    if start1 is pd.NaT or end1 is pd.NaT or start2 is pd.NaT or end2 is pd.NaT:
        print("One of the dates is NaT")
        print(f"start1: {start1}, end1: {end1}, start2: {start2}, end2: {end2}")
        return None
    return (abs(start1 - start2).days + abs(end1 - end2).days) / 2