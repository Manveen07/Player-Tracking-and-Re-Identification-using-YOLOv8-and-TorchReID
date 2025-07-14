# # Import necessary libraries
# import os  
# import cv2  
# import numpy as np  #
# from ultralytics import YOLO  
# from torchvision import transforms  
# from PIL import Image  
# from sklearn.metrics.pairwise import cosine_similarity  
# import torch  
# import torchreid  
# from utilss.bbox_utils import get_center_of_bbbox, get_bbox_width


# # Path settings
# VIDEO_PATH = "assets/15sec_input_720p.mp4"  # Input video file
# MODEL_PATH = "model/best.pt"  # Trained YOLO model
# OUTPUT_DIR = "output/tracked_frames"  # Where to save processed frames
# OUTPUT_VIDEO_PATH = "output/tracked_video_reid_final.mp4"  # Output video file
# os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output folder if it doesn't exist

# # Tracking variables
# global_id_counter = 0  # Counter to assign unique IDs to players
# active_tracks = {}  # Stores currently tracked players
# inactive_gallery = []  # Stores features of players no longer in frame

# # Set up device (use GPU if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load re-identification model (to recognize players across frames)
# reid_model = torchreid.models.build_model('osnet_ain_x1_0', num_classes=1000, pretrained=True)
# reid_model.to(device)  # Move model to GPU if available
# reid_model.eval()  # Set model to evaluation mode

# # Image transformations for the re-identification model
# transform = transforms.Compose([
#     transforms.Resize((256, 128)),  # Resize to expected input size
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
# ])

# def extract_features(image_crop):
#     """Extract features from a player image crop for re-identification"""
#     try:
#         # Convert image format and apply transformations
#         img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
#         tensor = transform(img).unsqueeze(0).to(device)
        
#         # Extract features using the re-identification model
#         with torch.no_grad():
#             features = reid_model(tensor).cpu().numpy().flatten()
#             features = features / np.linalg.norm(features)  # Normalize features
#         return features
#     except:
#         return None

# def match_in_gallery(features, used_global_ids, threshold=0.7):
#     """Check if current player matches someone in the gallery (previously seen players)"""
#     if not inactive_gallery:
#         return None
    
#     # Filter out players already matched in this frame
#     filtered = [g for g in inactive_gallery if g['global_id'] not in used_global_ids]
#     if not filtered:
#         return None
    
#     # Get features and IDs from gallery
#     gallery_features = [get_mean_features(g) for g in filtered]
#     gallery_ids = [g['global_id'] for g in filtered]
    
#     # Calculate similarity scores
#     sims = cosine_similarity([features], gallery_features)[0]
#     best_idx = np.argmax(sims)  # Find most similar player
    
#     # If similarity is above threshold, return that player's ID
#     if sims[best_idx] > threshold:
#         print(f"Matched with ID {gallery_ids[best_idx]} (score: {sims[best_idx]:.3f})")
#         return gallery_ids[best_idx]
#     return None

# def update_gallery(global_id, new_feature, history_len=5):
#     """Update the gallery with new features for a player"""
#     # If player already in gallery, update their features
#     for entry in inactive_gallery:
#         if entry['global_id'] == global_id:
#             entry['features'].append(new_feature)
#             if len(entry['features']) > history_len:
#                 entry['features'].pop(0)  # Keep only recent features
#             return
    
#     # If new player, add to gallery
#     inactive_gallery.append({
#         'global_id': global_id,
#         'features': [new_feature]
#     })

# def get_mean_features(entry):
#     """Calculate average features for a gallery entry"""
#     return np.mean(entry['features'], axis=0)

# def assign_global_id(track_id, bbox, frame, global_id_to_track_id, frame_idx=0):
#     """Assign a persistent ID to a detected player"""
#     global global_id_counter
    
#     # If we're already tracking this player, return their existing ID
#     if track_id in active_tracks:
#         global_id = active_tracks[track_id]['global_id']
#         if global_id not in global_id_to_track_id:
#             global_id_to_track_id[global_id] = track_id
#             return global_id

#     # Extract player image with some padding around bounding box
#     pad = 10
#     x1, y1, x2, y2 = map(int, bbox)
#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(frame.shape[1], x2 + pad)
#     y2 = min(frame.shape[0], y2 + pad)
#     crop = frame[y1:y2, x1:x2]
    
#     # Extract features from player image
#     features = extract_features(crop)
#     if features is None:
#         return None

#     # For first few frames, don't try to match with gallery
#     if frame_idx < 10:
#         matched_global_id = None
#     else:
#         matched_global_id = match_in_gallery(features, global_id_to_track_id.keys())

#     # Prevent assigning same ID twice in one frame
#     if matched_global_id is not None:
#         if matched_global_id in global_id_to_track_id:
#             matched_global_id = None

#     # Assign ID - either matched from gallery or new ID
#     if matched_global_id is not None:
#         global_id = matched_global_id
#     else:
#         global_id_counter += 1
#         global_id = global_id_counter

#     # Update tracking info
#     update_gallery(global_id, features)
#     active_tracks[track_id] = {'global_id': global_id, 'features': features}
#     global_id_to_track_id[global_id] = track_id
#     return global_id

# def retire_lost_tracks(current_track_ids):
#     """Remove players who are no longer in the frame"""
#     lost_ids = set(active_tracks.keys()) - set(current_track_ids)
#     for tid in lost_ids:
#         info = active_tracks[tid]
#         update_gallery(info['global_id'], info['features'])  # Save their features
#         del active_tracks[tid]  # Remove from active tracking

# def draw_triangle(frame, bbox, color):
#     """Draw a triangle marker (used for the ball)"""
#     y = int(bbox[1])
#     x, _ = get_center_of_bbbox(bbox)

#     triangle_points = np.array([
#         [x, y],
#         [x - 10, y - 20],
#         [x + 10, y - 20],
#     ])
#     cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
#     cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

#     return frame

# def draw_ellipse(frame, bbox, color, track_id=None):
#     """Draw an ellipse and ID number for a player"""
#     y2 = int(bbox[3])
#     x_center, _ = get_center_of_bbbox(bbox)
#     width = get_bbox_width(bbox)

#     # Draw the ellipse at player's feet
#     cv2.ellipse(
#         frame,
#         center=(x_center, y2),
#         axes=(int(width), int(0.35 * width)),
#         angle=0.0,
#         startAngle=-45,
#         endAngle=235,
#         color=color,
#         thickness=2,
#         lineType=cv2.LINE_4
#     )

#     # Draw the ID number box
#     rectangle_width = 40
#     rectangle_height = 20
#     x1_rect = x_center - rectangle_width // 2
#     x2_rect = x_center + rectangle_width // 2
#     y1_rect = (y2 - rectangle_height // 2) + 15
#     y2_rect = (y2 + rectangle_height // 2) + 15

#     if track_id is not None:
#         # Draw filled rectangle
#         cv2.rectangle(frame,
#                       (int(x1_rect), int(y1_rect)),
#                       (int(x2_rect), int(y2_rect)),
#                       color,
#                       cv2.FILLED)

#         # Adjust text position based on ID number length
#         x1_text = x1_rect + 12
#         if track_id > 99:
#             x1_text -= 10

#         # Draw the ID number
#         cv2.putText(
#             frame,
#             f"{track_id}",
#             (int(x1_text), int(y1_rect + 15)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 0, 0),
#             2
#         )

#     return frame

# def draw_frame(frame, results, class_names, frame_idx):
#     """Process one frame - draw bounding boxes and IDs"""
#     annotated = frame.copy()
    
#     # Colors for different classes (players, referees, etc.)
#     color_map = {
#         'player': (255, 255, 255),  # White
#         'referee': (0, 215, 255),  # Yellow
#         'goalkeeper': (255, 255, 255),  # White
#         "ball": (0, 255, 0)  # Green
#     }
    
#     global_id_to_track_id = {}  # Maps persistent IDs to frame-specific track IDs
#     current_frame_track_ids = []  # Track IDs in current frame
    
#     if results.boxes.id is not None:
#         boxes = results.boxes.data.cpu().numpy()
#         for *xyxy, track_id, conf, cls_id in boxes:
#             x1, y1, x2, y2 = map(int, xyxy)
#             track_id = int(track_id)
#             cls_id = int(cls_id)
#             label = class_names.get(cls_id, f"class{cls_id}")
#             color = color_map.get(label, (128, 128, 128))  # Default gray
            
#             # Special handling for ball
#             if label == 'ball':
#                 draw_triangle(annotated, (x1, y1, x2, y2), color)
#                 continue  # Skip ReID for ball
                
#             current_frame_track_ids.append(track_id)
#             # Assign persistent ID to player
#             global_id = assign_global_id(track_id, (x1, y1, x2, y2), frame, global_id_to_track_id, frame_idx)
            
#             # Draw player marker with ID
#             draw_ellipse(annotated, (x1, y1, x2, y2), color, global_id)

#     # Clean up tracks for players who left the frame
#     retire_lost_tracks(current_frame_track_ids)
#     return annotated

# def run_tracking(video_path, model_path, output_dir, output_video_path):
#     """Main function to process video"""
#     # Load YOLO model
#     model = YOLO(model_path)
#     class_names = model.names  # Get class names
    
#     # Set up video input and output
#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Run object detection and tracking
#         results = model.track(source=frame, persist=True, conf=0.7, verbose=False)[0]
        
#         # Draw tracking results on frame
#         output_frame = draw_frame(frame, results, class_names, frame_idx=frame_idx)
        
#         # Save results
#         cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), output_frame)
#         out_writer.write(output_frame)
#         frame_idx += 1
    
#     # Clean up
#     cap.release()
#     out_writer.release()
#     print(f"Processed {frame_idx} frames. Video saved to {output_video_path}")

# if __name__ == "__main__":
#     run_tracking(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH)









import os
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchreid
from utilss.bbox_utils import get_center_of_bbbox, get_bbox_width


VIDEO_PATH = "assets/15sec_input_720p.mp4"
MODEL_PATH = "model/best.pt"
OUTPUT_DIR = "output/tracked_frames"
OUTPUT_VIDEO_PATH = "output/tracked_video_reid_final.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

global_id_counter = 0
active_tracks = {}
inactive_gallery = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model = torchreid.models.build_model('osnet_ain_x1_0', num_classes=1000, pretrained=True)
reid_model.to(device)
reid_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def extract_features(image_crop):
    try:
        img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = reid_model(tensor).cpu().numpy().flatten()
            features = features / np.linalg.norm(features)
        return features
    except:
        return None

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

def get_mean_features(entry):
    features = np.array(entry['features'])
    weights = np.linspace(1, 2, num=len(features))
    weights /= weights.sum()
    return np.average(features, axis=0, weights=weights)

def draw_triangle(frame, bbox, color):
    """Draw a triangle marker (used for the ball)"""
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

def draw_ellipse(frame, bbox, color, track_id=None):
    """Draw an ellipse and ID number for a player"""
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbbox(bbox)
    width = get_bbox_width(bbox)

    # Draw the ellipse at player's feet
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

    # Draw the ID number box
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    if track_id is not None:
        # Draw filled rectangle
        cv2.rectangle(frame,
                      (int(x1_rect), int(y1_rect)),
                      (int(x2_rect), int(y2_rect)),
                      color,
                      cv2.FILLED)

        # Adjust text position based on ID number length
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        # Draw the ID number
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


def assign_global_id(track_id, bbox, frame, used_global_ids, frame_idx=0):
    global global_id_counter
    if track_id in active_tracks:
        global_id = active_tracks[track_id]['global_id']
        if global_id not in used_global_ids:
            return global_id

    pad = 10
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)

    crop = frame[y1:y2, x1:x2]
    features = extract_features(crop)
    if features is None:
        return None

    matched_global_id = match_in_gallery(features, used_global_ids) if frame_idx >= 10 else None

    if matched_global_id is not None:
        global_id = matched_global_id
    else:
        global_id_counter += 1
        global_id = global_id_counter

    update_gallery(global_id, features)
    active_tracks[track_id] = {
        'global_id': global_id,
        'features': features,
        'age': 0  # New for aging logic
    }
    used_global_ids.add(global_id)
    return global_id

def retire_lost_tracks(current_track_ids, max_age=30):
    for track_id in list(active_tracks.keys()):
        if track_id not in current_track_ids:
            active_tracks[track_id]['age'] += 1
            if active_tracks[track_id]['age'] > max_age:
                info = active_tracks[track_id]
                update_gallery(info['global_id'], info['features'])
                del active_tracks[track_id]
        else:
            active_tracks[track_id]['age'] = 0  # Reset age if seen

def draw_frame(frame, results, class_names, frame_idx):
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_map = {
        'player': (255, 255, 255),
        'referee': (0, 215, 255),
        'goalkeeper': (0, 0, 255)
    }
    used_global_ids = set()
    current_frame_track_ids = []

    if results.boxes.id is not None:
        boxes = results.boxes.data.cpu().numpy()
        for *xyxy, track_id, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = int(track_id)
            cls_id = int(cls_id)
            label = class_names.get(cls_id, f"class{cls_id}")
            color = color_map.get(label, (128, 128, 128))
            if label == 'ball':
                draw_triangle(annotated, (x1, y1, x2, y2),color=color)
                continue
            current_frame_track_ids.append(track_id)
            global_id = assign_global_id(track_id, (x1, y1, x2, y2), frame, used_global_ids, frame_idx)
            if global_id is None:
                continue
            
            draw_ellipse(annotated, (x1, y1, x2, y2), color, global_id)


    retire_lost_tracks(current_frame_track_ids)
    return annotated

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(source=frame, persist=True, conf=0.5, verbose=False)[0]
        output_frame = draw_frame(frame, results, class_names, frame_idx)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), output_frame)
        out_writer.write(output_frame)
        frame_idx += 1

    cap.release()
    out_writer.release()
    print(f"Processed {frame_idx} frames. Video saved to {output_video_path}")

if __name__ == "__main__":
    run_tracking(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH)
