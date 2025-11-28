import cv2
import os
import sys
import numpy as np
from edge_impulse_linux.runner import ImpulseRunner
import time
from tracker import CentroidTracker 

MODEL_FILE = '../models/project-2-linux-x86_64-v5.eim'
INPUT_VIDEO = '../assets/videos/test_video.mp4'
OUTPUT_VIDEO = '../assets/videos/result_video.mp4'

CONFIDENCE_THRESHOLD = 0.6

def merge_close_points(detections, min_dist=60):
    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    final_detections = []
    
    while len(detections) > 0:
        current = detections.pop(0) 
        final_detections.append(current)

        detections = [
            d for d in detections
            if ((d['x'] - current['x'])**2 + (d['y'] - current['y'])**2)**0.5 > min_dist
        ]
    
    return final_detections

# maxDisappeared: If model loses detection for X frames, keep the ID. 
# Increased to 20 because Pi Zero FPS is low (frames are far apart in time).
# maxDistance: Max pixels an ID can "jump" between frames.
ct = CentroidTracker(maxDisappeared=30, maxDistance=80)

def main(argv):
    runner = None
    try:
        print(f" -> Loading model: {MODEL_FILE}...")
        runner = ImpulseRunner(MODEL_FILE)
        model_info = runner.init()
        
        model_params = model_info['model_parameters']
        input_width = model_params.get('input_width') or model_params.get('image_input_width')
        input_height = model_params.get('input_height') or model_params.get('image_input_height')
        
        print(f" -> Model Input Size: {input_width}x{input_height}")

        # Setting up video
        if not os.path.exists(INPUT_VIDEO):
            print("Error: Input video file not found.")
            return

        cap = cv2.VideoCapture(INPUT_VIDEO)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (orig_w, orig_h))

        print(" -> Starting Tracking Test...")
        
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # 1. Resize & Packed RGB Conversion (Correct Input Logic)
            img_resized = cv2.resize(frame, (input_width, input_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.uint32)
            packed_features = (img_rgb[:, :, 0] << 16) | (img_rgb[:, :, 1] << 8) | img_rgb[:, :, 2]
            features = packed_features.flatten().tolist()

            # 2. Inference
            res = runner.classify(features)

            # 3. Prepare Detections for Tracker
            raw_points = []
            
            if "bounding_boxes" in res["result"]:
                for bb in res["result"]["bounding_boxes"]:
                    if bb['value'] >= CONFIDENCE_THRESHOLD:
                        # Map coords
                        cx_model = bb['x'] + (bb['width'] / 2)
                        cy_model = bb['y'] + (bb['height'] / 2)
                        
                        cx_orig = int(cx_model / input_width * orig_w)
                        cy_orig = int(cy_model / input_height * orig_h)
                        
                        raw_points.append({'x': cx_orig, 'y': cy_orig, 'score': bb['value']})

            filtered_points = merge_close_points(raw_points, min_dist=50) 

            points_for_tracker = [(p['x'], p['y']) for p in filtered_points]

            # 4. Update Tracker
            objects = ct.update(points_for_tracker)

            # 5. Visualize Tracking
            for (objectID, centroid) in objects.items():
                # Draw the ID (Green text)
                text = f"ID {objectID}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                
                # Draw the Dot (Green)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # Print status
            sys.stdout.write(f"\rProcessing Frame: {frame_count}/{total_frames} | Active IDs: {len(objects)}")
            sys.stdout.flush()

            out.write(frame)

        cap.release()
        out.release()
        print(f"\n\n -> Done! Check the video: {OUTPUT_VIDEO}")

    finally:
        if runner: runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])