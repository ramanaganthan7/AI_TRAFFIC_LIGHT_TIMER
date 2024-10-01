import torch
import cv2
import os
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Specify the path to the local YOLOv5 directory
local_yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')

# Load the YOLOv5 model from the local directory
model = torch.hub.load(local_yolov5_path, 'yolov5s', source='local')

# Function to calculate vehicle density in a lane
def calculate_density(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    frame_count = 0
    vehicle_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count} of video {video_path}")

        # Perform detection
        results = model(frame)

        # Print detection results for the first frame
        if frame_count == 1:
            print(results.pandas().xyxy[0])  # Print the detection results

        # Filter out vehicles
        vehicles = results.pandas().xyxy[0].query("name in ['car', 'truck', 'bus', 'motorcycle']")
        vehicle_count += len(vehicles)

    cap.release()

    if frame_count == 0:
        print(f"No frames processed for video {video_path}")
        return 0

    density = vehicle_count / frame_count
    print(f"Density for {video_path}: {density}")
    return density

# Paths to the video files for each lane
project_folder = os.path.dirname(os.path.abspath(__file__))
video_folder = os.path.join(project_folder, 'data')

video_paths = [
    os.path.join(video_folder, "lane1.mp4"),
    os.path.join(video_folder, "lane2.mp4"),
    os.path.join(video_folder, "lane3.mp4"),
    os.path.join(video_folder, "lane4.mp4")
]

# Ensure the video files exist
for video in video_paths:
    if not os.path.exists(video):
        print(f"Video file not found: {video}")
        exit(1)

# Calculate densities for each lane
densities = [calculate_density(video) for video in video_paths]

# Function to calculate signal timings based on density
def calculate_signal_times(densities):
    total_density = sum(densities)
    green_times = [int((density / total_density) * 60) for density in densities]  # Total cycle time is 60 seconds
    yellow_time = 5  # Fixed yellow time
    red_times = [60 - (green_time + yellow_time) for green_time in green_times]

    return green_times, yellow_time, red_times

# Calculate signal timings
green_times, yellow_time, red_times = calculate_signal_times(densities)

# Print the timings for each lane
for i in range(4):
    print(f"Lane {i + 1}: Green time: {green_times[i]}s, Yellow time: {yellow_time}s, Red time: {red_times[i]}s")



cv2.destroyAllWindows()
