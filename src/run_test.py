import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
from edge_impulse_linux.runner import ImpulseRunner
from tracker import CentroidTracker
import psutil

# --- CONFIGURATION ---
MODEL_FILE = '../models/project-2-linux-x86_64-v5.eim'
VIDEO_DIR = '../testing_videos'
OUTPUT_DIR = '../assets/videos/results'

CONFIDENCE_THRESHOLD = 0.6

# --- COUNTING CONFIG ---
LINE_POSITION = 0.5  # Line position (0.0 to 1.0). 0.5 = Center of video.
LINE_OFFSET = 30  # Pixel buffer to detect crossing event

def process_video(video_path, runner, model_params):
    """
    Process a single video for people counting.
    Returns a dictionary with counting results.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*80}")
    
    input_width = model_params.get('input_width') or model_params.get('image_input_width')
    input_height = model_params.get('input_height') or model_params.get('image_input_height')
    
    # Initialize tracker and counters for this video
    ct = CentroidTracker(maxDisappeared=30, maxDistance=80)
    total_up = 0
    total_down = 0
    trackableObjects = {}
    
    # Open Video File
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup Output Video
    output_filename = f"result_counting_{video_path.stem}.mp4"
    output_path = Path(OUTPUT_DIR) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (orig_w, orig_h))
    
    # Calculate Line Y-Coordinate
    line_y = int(orig_h * LINE_POSITION)
    print(f" -> Video Size: {orig_w}x{orig_h} @ {fps:.1f} FPS")
    print(f" -> Total Frames: {total_frames}")
    print(f" -> Counting Line at Y={line_y}")
    
    # Memory tracking
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    peak_memory = initial_memory
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
        
        # Monitor memory usage
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        if current_memory > peak_memory:
            peak_memory = current_memory
        
        # Log to console
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_curr = frame_count / elapsed
            sys.stdout.write(f"\rFrame {frame_count}/{total_frames} | FPS: {fps_curr:.1f} | UP: {total_up} DOWN: {total_down} | RAM: {current_memory:.1f}MB")
            sys.stdout.flush()
    
    cap.release()
    out.release()
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    final_memory = process.memory_info().rss / (1024 * 1024)  # MB
    memory_used = final_memory - initial_memory
    
    print(f"\n -> Processed {frame_count} frames in {elapsed_time:.2f}s (avg {avg_fps:.1f} FPS)")
    print(f" -> Final Count: UP={total_up}, DOWN={total_down}")
    print(f" -> Memory: Initial={initial_memory:.1f}MB, Peak={peak_memory:.1f}MB, Used={memory_used:.1f}MB")
    print(f" -> Result saved to: {output_path}")
    
    return {
        'video_name': video_path.name,
        'up': total_up,
        'down': total_down,
        'frames': frame_count,
        'time': elapsed_time,
        'fps': avg_fps,
        'initial_memory': initial_memory,
        'peak_memory': peak_memory,
        'memory_used': memory_used,
        'output': str(output_path)
    }

def main(argv):
    runner = None
    try:
        # 1. Load Model
        print(f"Loading model: {MODEL_FILE}...")
        runner = ImpulseRunner(MODEL_FILE)
        model_info = runner.init()
        model_params = model_info['model_parameters']
        input_width = model_params.get('input_width') or model_params.get('image_input_width')
        input_height = model_params.get('input_height') or model_params.get('image_input_height')
        
        print(f"Model Input: {input_width}x{input_height}")
        print(f"Mode: PACKED RGB (Video Counting)")
        
        # 2. Get all video files from testing_videos directory
        video_dir = Path(VIDEO_DIR)
        if not video_dir.exists():
            print(f"Error: Video directory not found: {VIDEO_DIR}")
            return
        
        # Get all .mp4 files
        video_files = sorted(video_dir.glob('*.mp4'))
        
        if not video_files:
            print(f"No video files found in {VIDEO_DIR}")
            return
        
        print(f"\nFound {len(video_files)} video(s) to process.")
        
        # Get system information
        total_ram = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        print(f"System RAM: {total_ram:.2f} GB")
        
        # 3. Process each video sequentially
        results = []
        overall_start = time.time()
        
        for idx, video_path in enumerate(video_files, 1):
            print(f"\n[{idx}/{len(video_files)}] Starting video: {video_path.name}")
            result = process_video(video_path, runner, model_params)
            if result:
                results.append(result)
        
        # 4. Print Summary
        overall_time = time.time() - overall_start
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total videos processed: {len(results)}")
        print(f"Total time: {overall_time:.2f}s\n")
        
        print(f"{'Video Name':<40} {'UP':>6} {'DOWN':>6} {'Time(s)':>10} {'FPS':>8} {'PeakRAM':>10}")
        print(f"{'-'*90}")
        
        total_up_all = 0
        total_down_all = 0
        max_peak_memory = 0
        
        for r in results:
            print(f"{r['video_name']:<40} {r['up']:>6} {r['down']:>6} {r['time']:>10.2f} {r['fps']:>8.1f} {r['peak_memory']:>9.1f}MB")
            total_up_all += r['up']
            total_down_all += r['down']
            if r['peak_memory'] > max_peak_memory:
                max_peak_memory = r['peak_memory']
        
        print(f"{'-'*90}")
        print(f"{'TOTAL':<40} {total_up_all:>6} {total_down_all:>6}")
        print(f"\nMax Peak Memory Usage: {max_peak_memory:.1f} MB")
        print(f"All results saved to: {OUTPUT_DIR}")
        
        # 5. Write results to text file
        output_dir_path = Path(OUTPUT_DIR)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        results_file = output_dir_path / 'counting_results.txt'
        
        with open(results_file, 'w') as f:
            f.write("="*90 + "\n")
            f.write("PEOPLE COUNTING RESULTS\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System RAM: {total_ram:.2f} GB\n")
            f.write("="*90 + "\n\n")
            f.write(f"Total videos processed: {len(results)}\n")
            f.write(f"Total processing time: {overall_time:.2f}s\n")
            f.write(f"Max Peak Memory Usage: {max_peak_memory:.1f} MB\n\n")
            
            f.write(f"{'Video Name':<40} {'UP':>6} {'DOWN':>6} {'Time(s)':>10} {'FPS':>8} {'PeakRAM':>10}\n")
            f.write("-"*90 + "\n")
            
            for r in results:
                f.write(f"{r['video_name']:<40} {r['up']:>6} {r['down']:>6} {r['time']:>10.2f} {r['fps']:>8.1f} {r['peak_memory']:>9.1f}MB\n")
            
            f.write("-"*90 + "\n")
            f.write(f"{'TOTAL':<40} {total_up_all:>6} {total_down_all:>6}\n\n")
            
            f.write("\nDetailed Results:\n")
            f.write("="*90 + "\n")
            for r in results:
                f.write(f"\nVideo: {r['video_name']}\n")
                f.write(f"  - People going UP: {r['up']}\n")
                f.write(f"  - People going DOWN: {r['down']}\n")
                f.write(f"  - Total frames: {r['frames']}\n")
                f.write(f"  - Processing time: {r['time']:.2f}s\n")
                f.write(f"  - Average FPS: {r['fps']:.1f}\n")
                f.write(f"  - Memory - Initial: {r['initial_memory']:.1f}MB, Peak: {r['peak_memory']:.1f}MB, Used: {r['memory_used']:.1f}MB\n")
                f.write(f"  - Output file: {r['output']}\n")
        
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if runner:
            runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
