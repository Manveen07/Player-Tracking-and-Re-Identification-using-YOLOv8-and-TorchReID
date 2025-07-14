# Player-Tracking-and-Re-Identification-using-YOLOv8-and-TorchReID

Overview
This system detects and tracks soccer players, referees, and the ball in video footage, assigning persistent IDs to each player throughout the video. It combines object detection with player re-identification to maintain consistent tracking even when players leave and re-enter the frame.

git clone https://github.com/Manveen07/Player-Tracking-and-Re-Identification-using-YOLOv8-and-TorchReID.git
cd Player-Tracking-and-Re-Identification-using-YOLOv8-and-TorchReID

python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
or
pip install torch torchvision opencv-python ultralytics scikit-learn Pillow git+https://github.com/KaiyangZhou/deep-person-reid

Place best.pt in the model/ directory

Input Video
Place your input video in the assets/ folder and update VIDEO_PATH accordingly.

python main.py

The script will:

Detect and track players in the video

Assign consistent IDs even through occlusion using TorchReID

Save annotated frames in output/tracked_frames/

Save final video to output/tracked_video_reid_final1.mp4
