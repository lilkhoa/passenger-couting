import cv2
import os
import sys
import numpy as np
from edge_impulse_linux.runner import ImpulseRunner
import time

# --- CONFIGURATION ---
MODEL_FILE = '../models/project-2-linux-x86_64-v5.eim'
INPUT_VIDEO = '../assets/videos/test_video.mp4'
OUTPUT_VIDEO = '../assets/videos/result_video.mp4'

# Threshold
CONFIDENCE_THRESHOLD = 0.6

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
        print(" -> Mode: PACKED RGB (Raw Detection on Video)")

        # --- VIDEO SETUP ---
        if not os.path.exists(INPUT_VIDEO):
            print(f"Error: Input video file '{INPUT_VIDEO}' not found.")
            return

        cap = cv2.VideoCapture(INPUT_VIDEO)
        # Read original video dimensions
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f" -> Video Source: {orig_w}x{orig_h} | FPS: {fps:.2f} | Frames: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (orig_w, orig_h))

        print(" -> Starting inference...")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # 1. Resize (Squash)
            img_resized = cv2.resize(frame, (input_width, input_height))

            # 2. PACKED RGB CONVERSION
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Pack to uint32: (R << 16) | (G << 8) | B
            img_rgb = img_rgb.astype(np.uint32)
            packed_features = (img_rgb[:, :, 0] << 16) | (img_rgb[:, :, 1] << 8) | img_rgb[:, :, 2]
            features = packed_features.flatten().tolist()

            # 3. Inference
            res = runner.classify(features)

            # 4. Draw Raw Detections
            detect_count = 0
            if "bounding_boxes" in res["result"]:
                for bb in res["result"]["bounding_boxes"]:
                    score = bb['value']
                    
                    if score >= CONFIDENCE_THRESHOLD:
                        detect_count += 1
                        # Map coordinates back to Original Video Size
                        cx_model = bb['x'] + (bb['width'] / 2)
                        cy_model = bb['y'] + (bb['height'] / 2)
                        
                        cx_orig = int(cx_model / input_width * orig_w)
                        cy_orig = int(cy_model / input_height * orig_h)
                        
                        # Draw Red Dot
                        cv2.circle(frame, (cx_orig, cy_orig), 4, (0, 0, 255), -1)
                        
                        # Draw Score
                        cv2.putText(frame, f"{score:.2f}", (cx_orig+10, cy_orig), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)

            # Write to output video
            out.write(frame)

            # Print progress
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                sys.stdout.write(f"\rProcessing: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Objects: {detect_count}   ")
                sys.stdout.flush()

        cap.release()
        out.release()
        print(f"\n\n -> Done! Result saved to: {OUTPUT_VIDEO}")

    finally:
        if runner: runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])