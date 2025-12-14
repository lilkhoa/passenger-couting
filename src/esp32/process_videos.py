import cv2
import serial
import struct
import json
import os
import glob
import time

# CONFIG
SERIAL_PORT = 'COM3'     
BAUD_RATE = 115200
VIDEO_FOLDER = 'D:/Research/passenger-couting/testing_videos'
MODEL_W = 160           
MODEL_H = 160

def wait_for_ready(ser):
    start_wait = time.time()
    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "READY" in line:
                    return True
                if line.startswith('{'):
                    return True 
            except:
                pass
        
        if time.time() - start_wait > 5:
            return False
        time.sleep(0.01)

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print(f"Connected to {SERIAL_PORT} successfully (Baud: {BAUD_RATE})")
    except Exception as e:
        print(f"Serial Error: {e}")
        return

    ser.dtr = False
    ser.rts = False
    time.sleep(0.1)
    ser.dtr = True
    ser.rts = True
    print("Resetting ESP32 and waiting for startup...")
    time.sleep(2) 

    ser.reset_input_buffer()

    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    f_report = open("result_esp32.csv", "w")
    f_report.write("Video,Frames,UP,DOWN,FPS,RAM_Free\n")

    for video_path in video_files:
        print(f"\nVideo: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        frames_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames_count += 1

            # 1. WAIT FOR READY SIGNAL FROM ESP32 BEFORE SENDING
            is_ready = wait_for_ready(ser)
            if not is_ready:
                print(f"Frame {frames_count}: ESP32 did not respond with 'READY'. Retrying...")
            
            # 2. Process image
            img = cv2.resize(frame, (MODEL_W, MODEL_H))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = []
            for y in range(MODEL_H):
                for x in range(MODEL_W):
                    r, g, b = img_rgb[y, x]
                    val = (r << 16) | (g << 8) | b
                    pixels.append(val)
            
            # 3. Send data
            ser.write(b'IMG')
            ser.write(struct.pack(f'<{len(pixels)}I', *pixels))

            # 4. Read response            
            try:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('{'):
                    data = json.loads(line)
                    print(f"\rFrame {frames_count} | FPS: {data['fps']:.1f} | UP: {data['up']} | DOWN: {data['down']}", end="")
                    
                    last_data = data
                else:
                    pass
            except Exception as e:
                pass

        cap.release()
        if 'last_data' in locals():
            f_report.write(f"{os.path.basename(video_path)},{frames_count},{last_data['up']},{last_data['down']},{last_data['fps']:.2f},{last_data['ram']}\n")
            f_report.flush()

    f_report.close()
    ser.close()
    print("\n\n Processing completed. Results saved to result_esp32.csv")

if __name__ == "__main__":
    main()