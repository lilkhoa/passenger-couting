import cv2
import numpy as np
import time
import sys
import os
from edge_impulse_linux.runner import ImpulseRunner
from tracker import CentroidTracker

# --- CONFIGURATION ---
MODEL_FILE = '../models/project-2-linux-x86_64-v5.eim'
INPUT_VIDEO = '../assets/videos/test_video_2.mp4'
OUTPUT_VIDEO = f'../assets/videos/result_counting_{INPUT_VIDEO.split("/")[-1].split(".")[0]}.mp4'

CONFIDENCE_THRESHOLD = 0.6

# --- COUNTING CONFIG ---
# Line position (0.0 to 1.0). 0.5 = Center of video.
# Adjust this based on where the bus door is in your video.
LINE_POSITION = 0.5 
LINE_OFFSET = 30  # Pixel buffer to detect crossing event
ct = CentroidTracker(maxDisappeared=30, maxDistance=80)

# Global variables
total_up = 0
total_down = 0
trackableObjects = {}

def main(argv):
    global total_up, total_down, trackableObjects
    
    runner = None
    try:
        # 1. Load Model
        print(f" -> Loading model: {MODEL_FILE}...")
        runner = ImpulseRunner(MODEL_FILE)
        model_info = runner.init()
        model_params = model_info['model_parameters']
        input_width = model_params.get('input_width') or model_params.get('image_input_width')
        input_height = model_params.get('input_height') or model_params.get('image_input_height')
        
        print(f" -> Model Input: {input_width}x{input_height}")
        print(" -> Mode: PACKED RGB (Video Counting)")

        # 2. Open Video File
        if not os.path.exists(INPUT_VIDEO):
            print("Error: Input video file not found.")
            return

        cap = cv2.VideoCapture(INPUT_VIDEO)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup Output Video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (orig_w, orig_h))

        # Calculate Line Y-Coordinate
        line_y = int(orig_h * LINE_POSITION)
        print(f" -> Counting Line set at Y={line_y}")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # --- A. DETECTION (Packed RGB Logic) ---
            img_resized = cv2.resize(frame, (input_width, input_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.uint32)
            packed_features = (img_rgb[:, :, 0] << 16) | (img_rgb[:, :, 1] << 8) | img_rgb[:, :, 2]
            features = packed_features.flatten().tolist()
            
            res = runner.classify(features)

            # Extract Centroids
            input_points = []
            if "bounding_boxes" in res["result"]:
                for bb in res["result"]["bounding_boxes"]:
                    if bb['value'] >= CONFIDENCE_THRESHOLD:
                        # Map to Original Video Size
                        cx = int((bb['x'] + bb['width']/2) / input_width * orig_w)
                        cy = int((bb['y'] + bb['height']/2) / input_height * orig_h)
                        input_points.append((cx, cy))

            # --- B. TRACKING ---
            # Assign IDs to points
            objects = ct.update(input_points)

            # --- C. COUNTING LOGIC ---
            # Draw the Counting Line (Yellow)
            cv2.line(frame, (0, line_y), (orig_w, line_y), (0, 255, 255), 2)

            for (objectID, centroid) in objects.items():
                # Retrieve object history
                to = trackableObjects.get(objectID, None)

                if to is None:
                    to = {"centroids": [centroid], "counted": False}
                else:
                    # Calculate direction: Y_current - Mean(Y_history)
                    # Positive = Moving Down, Negative = Moving Up
                    y_history = [c[1] for c in to["centroids"]]
                    direction = centroid[1] - np.mean(y_history)
                    to["centroids"].append(centroid)

                    # CHECK FOR LINE CROSSING
                    if not to["counted"]:
                        # Case: Going UP (Y is decreasing)
                        if direction < 0 and centroid[1] < line_y and centroid[1] > (line_y - LINE_OFFSET):
                            total_down += 1
                            to["counted"] = True
                            print(f" -> Event: Person ID {objectID} went DOWN.")

                        # Case: Going DOWN (Y is increasing)
                        elif direction > 0 and centroid[1] > line_y and centroid[1] < (line_y + LINE_OFFSET):
                            total_up += 1
                            to["counted"] = True
                            print(f" -> Event: Person ID {objectID} went UP.")

                # Save history back
                trackableObjects[objectID] = to

                # Visualize ID and Centroid
                text = f"ID {objectID}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # --- D. DASHBOARD ---
            info_text = f"UP: {total_up} | DOWN: {total_down}"
            # Black background for text
            cv2.rectangle(frame, (0, 0), (orig_w, 40), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            out.write(frame)

            # Log to console
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_curr = frame_count / elapsed
                sys.stdout.write(f"\rFrame {frame_count}/{total_frames} | FPS: {fps_curr:.1f} | UP: {total_up} DOWN: {total_down}")
                sys.stdout.flush()

        cap.release()
        out.release()
        print(f"\n\n -> Done! Result saved to: {OUTPUT_VIDEO}")

    finally:
        if runner: runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])