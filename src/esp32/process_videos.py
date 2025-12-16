import cv2
import serial
import json
import os
import glob
import time
import numpy as np

# CONFIG
SERIAL_PORT = 'COM3'
BAUD_RATE = 921600
VIDEO_FOLDER = 'D:/Research/passenger-couting/testing_videos'
MODEL_W = 160
MODEL_H = 160

def wait_for_ready(ser, timeout=10):
    buffer = ""
    start_wait = time.time()
    while True:
        if ser.in_waiting > 0:
            try:
                chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                buffer += chunk
                if "READY" in buffer or "Ready" in buffer: return True
                if len(buffer) > 500: buffer = buffer[-200:]
            except: pass
        if time.time() - start_wait > timeout: return False
        time.sleep(0.005)

def perform_hard_reset(ser):
    print("Sending Hard Reset...", end="")
    ser.write(b'RST') 
    
    time.sleep(1) 
    ser.reset_input_buffer()
    
    start_time = time.time()
    buffer = ""
    while time.time() - start_time < 5:
        if ser.in_waiting > 0:
            try:
                chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                buffer += chunk
                if "Ready" in buffer or "READY" in buffer:
                    print(" Done! RAM is 100% clean.")
                    ser.reset_input_buffer()
                    return True
            except: pass
        time.sleep(0.01)
    
    print(" Timeout (Failed. Try Reset button manually)!")
    return False

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=20, dsrdtr=False)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e: print(f"COM Error: {e}"); return
    
    ser.reset_input_buffer()
    
    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    if not video_files: print("No videos found!"); return
    
    # Open CSV file to write results
    f_csv = open("metrics_report.csv", "w")
    # CSV header
    f_csv.write("Video,Frames,UP,DOWN,AvgFPS,AvgLat_ms,PeakRAM_KB,Flash_KB,TotalEnergy_J\n")
    
    for video_path in video_files:
        print(f"\nVIDEO: {os.path.basename(video_path)}")

        if (not (os.path.basename(video_path).lower().startswith("7") or os.path.basename(video_path).lower().startswith("8") or os.path.basename(video_path).lower().startswith("9"))):
            continue
        
        # 1. HARD RESET BEFORE EACH VIDEO
        # Ensure a completely clean test environment
        wait_for_ready(ser, timeout=5)
        if not perform_hard_reset(ser): continue # If reset fails, skip this video
        
        # 2. Xử lý Video
        cap = cv2.VideoCapture(video_path)
        
        frames_count = 0
        total_energy = 0.0
        fps_list = []
        lat_list = []
        ram_peak = 0
        flash_size = 0
        last_up = 0
        last_down = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames_count += 1
            
            # Chờ ESP32 sẵn sàng nhận ảnh
            if not wait_for_ready(ser, timeout=10): continue

            # Chuẩn bị ảnh
            img = cv2.resize(frame, (MODEL_W, MODEL_H))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_u32 = img_rgb.astype(np.uint32)
            # Đóng gói: 0x00RRGGBB (Little Endian -> B, G, R, 0)
            packed = (img_u32[:,:,0] << 16) | (img_u32[:,:,1] << 8) | img_u32[:,:,2]
            
            # Gửi
            ser.write(b'IMG')
            ser.write(packed.flatten().tobytes())
            
            # Nhận kết quả
            try:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('{'):
                    data = json.loads(line)
                    
                    fps = data.get('fps', 0.0)
                    ram = data.get('ram_used', 0) / 1024.0 # Đổi ra KB
                    flash = data.get('flash_used', 0) / 1024.0 # Đổi ra KB
                    energy = data.get('energy_mj', 0.0) # mJ
                    latency = data.get('time_ms', 0.0)
                    
                    last_up = data.get('up', 0)
                    last_down = data.get('down', 0)
                    
                    total_energy += energy
                    fps_list.append(fps)
                    lat_list.append(latency)
                    if ram > ram_peak: ram_peak = ram
                    flash_size = flash
                    
                    print(f"\rFr {frames_count:<4}| FPS:{fps:.1f}| RAM:{ram:.0f}KB| E:{energy:.1f}mJ| UP:{last_up} DWN:{last_down}", end="")
            except: pass
            
        cap.release()
        
        # Tổng hợp số liệu video
        avg_fps = sum(fps_list)/len(fps_list) if fps_list else 0
        avg_lat = sum(lat_list)/len(lat_list) if lat_list else 0
        total_energy_joules = total_energy / 1000.0
        
        # Ghi vào báo cáo
        f_csv.write(f"{os.path.basename(video_path)},{frames_count},{last_up},{last_down},{avg_fps:.2f},{avg_lat:.2f},{ram_peak:.2f},{flash_size:.2f},{total_energy_joules:.4f}\n")
        f_csv.flush()
        print(f"\nDone: {avg_fps:.2f} FPS | {total_energy_joules:.2f} J | RAM: {ram_peak:.0f} KB")

    f_csv.close()
    ser.close()
    print("\nALL DONE!")
if __name__ == "__main__":
    main()