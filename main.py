# Import necessary libraries
import os
import cv2
import numpy as np
from ultralytics import YOLO  # For object detection and tracking
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchreid  # For re-identification
from utilss.bbox_utils import get_center_of_bbbox, get_bbox_width  # Custom utility functions

# Define paths
VIDEO_PATH = "assets/15sec_input_720p.mp4"
MODEL_PATH = "model/best.pt"
OUTPUT_DIR = "output/tracked_frames"
OUTPUT_VIDEO_PATH = "output/tracked_video_reid_final.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Make sure output directory exists

# Global tracking variables
global_id_counter = 0  # To assign unique global IDs to players
active_tracks = {}     # Stores currently active tracked objects
inactive_gallery = []  # Stores feature history of lost tracks (for ReID)

# Device setup for Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ReID model
reid_model = torchreid.models.build_model('osnet_ain_x1_0', num_classes=1000, pretrained=True)
reid_model.to(device)
reid_model.eval()

# Image transformation for ReID
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Extract normalized feature vector from player image crop
def extract_features(image_crop):
    try:
        img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = reid_model(tensor).cpu().numpy().flatten()
            features = features / np.linalg.norm(features)  # Normalize
        return features
    except:
        return None

# Try to match current player with existing identities using cosine similarity
def match_in_gallery(features, used_global_ids, threshold=0.7):
    if not inactive_gallery:
        return None
    filtered = [g for g in inactive_gallery if g['global_id'] not in used_global_ids]
    if not filtered:
        return None

    gallery_features = [get_mean_features(g) for g in filtered]
    gallery_ids = [g['global_id'] for g in filtered]
    sims = cosine_similarity([features], gallery_features)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] > threshold:
        print(f"Matched with ID {gallery_ids[best_idx]} (score: {sims[best_idx]:.3f})")
        return gallery_ids[best_idx]
    return None

# Update or create an entry in the gallery with new features
def update_gallery(global_id, new_feature, history_len=10):
    for entry in inactive_gallery:
        if entry['global_id'] == global_id:
            entry['features'].append(new_feature)
            if len(entry['features']) > history_len:
                entry['features'].pop(0)
            return
    inactive_gallery.append({
        'global_id': global_id,
        'features': [new_feature]
    })

# Get a weighted average of historical features
def get_mean_features(entry):
    features = np.array(entry['features'])
    weights = np.linspace(1, 2, num=len(features))  # Weight recent features more
    weights /= weights.sum()
    return np.average(features, axis=0, weights=weights)

# Draw triangle for the ball object
def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_of_bbbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
    return frame

# Draw ellipse and ID tag for a player
def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbbox(bbox)
    width = get_bbox_width(bbox)

    # Draw ellipse near player's feet
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    # Draw ID rectangle
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    if track_id is not None:
        cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    return frame

# Assign or re-identify a global ID for a track
def assign_global_id(track_id, bbox, frame, used_global_ids, frame_idx=0):
    global global_id_counter

    # If already active, reuse the ID
    if track_id in active_tracks:
        global_id = active_tracks[track_id]['global_id']
        if global_id not in used_global_ids:
            return global_id

    # Crop player region from frame
    pad = 10
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)
    crop = frame[y1:y2, x1:x2]

    # Extract features and match with gallery if enough frames have passed
    features = extract_features(crop)
    if features is None:
        return None

    matched_global_id = match_in_gallery(features, used_global_ids) if frame_idx >= 10 else None

    # Use matched ID or create a new one
    global_id = matched_global_id if matched_global_id else global_id_counter + 1
    if matched_global_id is None:
        global_id_counter += 1

    update_gallery(global_id, features)
    active_tracks[track_id] = {
        'global_id': global_id,
        'features': features,
        'age': 0
    }
    used_global_ids.add(global_id)
    return global_id

# Remove old/lost tracks that are inactive for a long time
def retire_lost_tracks(current_track_ids, max_age=30):
    for track_id in list(active_tracks.keys()):
        if track_id not in current_track_ids:
            active_tracks[track_id]['age'] += 1
            if active_tracks[track_id]['age'] > max_age:
                info = active_tracks[track_id]
                update_gallery(info['global_id'], info['features'])
                del active_tracks[track_id]
        else:
            active_tracks[track_id]['age'] = 0

# Main function to draw annotations for each frame
def draw_frame(frame, results, class_names, frame_idx):
    annotated = frame.copy()
    color_map = {
        'player': (255, 255, 255),
        'referee': (0, 215, 255),
        'goalkeeper': (0, 0, 255)
    }
    used_global_ids = set()
    current_frame_track_ids = []

    # Process each detection
    if results.boxes.id is not None:
        boxes = results.boxes.data.cpu().numpy()
        for *xyxy, track_id, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = int(track_id)
            cls_id = int(cls_id)
            label = class_names.get(cls_id, f"class{cls_id}")
            color = color_map.get(label, (128, 128, 128))

            if label == 'ball':
                draw_triangle(annotated, (x1, y1, x2, y2), color=color)
                continue

            current_frame_track_ids.append(track_id)
            global_id = assign_global_id(track_id, (x1, y1, x2, y2), frame, used_global_ids, frame_idx)
            if global_id is None:
                continue

            draw_ellipse(annotated, (x1, y1, x2, y2), color, global_id)

    retire_lost_tracks(current_frame_track_ids)
    return annotated

# Main tracking loop
def run_tracking(video_path, model_path, output_dir, output_video_path):
    model = YOLO(model_path)
    class_names = model.names

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_idx = 0

    # Read and process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracking
        results = model.track(source=frame, persist=True, conf=0.5, verbose=False)[0]

        # Draw results and write to output
        output_frame = draw_frame(frame, results, class_names, frame_idx)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), output_frame)
        out_writer.write(output_frame)
        frame_idx += 1

    cap.release()
    out_writer.release()
    print(f"Processed {frame_idx} frames. Video saved to {output_video_path}")

# Start the process
if __name__ == "__main__":
    run_tracking(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH)

