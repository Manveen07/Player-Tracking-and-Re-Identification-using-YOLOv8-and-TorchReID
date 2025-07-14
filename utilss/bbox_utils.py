def get_center_of_bbbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return x2 - x1

def clip_bbox(x1, y1, x2, y2, shape, pad=10):
    h, w = shape[:2]
    x1_clipped = max(0, x1 - pad)
    y1_clipped = max(0, y1 - pad)
    x2_clipped = min(w, x2 + pad)
    y2_clipped = min(h, y2 + pad)

    # Ensure bbox is still valid
    if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
        return None  # or raise ValueError("Invalid clipped bbox")

    return int(x1_clipped), int(y1_clipped), int(x2_clipped), int(y2_clipped)

