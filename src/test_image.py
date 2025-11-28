import cv2
import os
import sys
import numpy as np
from edge_impulse_linux.runner import ImpulseRunner

# --- CẤU HÌNH ---
MODEL_FILE = './project-2-linux-x86_64-v5.eim' # Model x86 của bạn
IMAGE_FILE = 'test_image_160x160.jpg' 
CONFIDENCE_THRESHOLD = 0.6 # Để thấp để test

def main(argv):
    runner = None
    try:
        print(f" -> Loading model: {MODEL_FILE}")
        runner = ImpulseRunner(MODEL_FILE)
        model_info = runner.init()
        
        model_params = model_info['model_parameters']
        input_width = model_params.get('input_width') or model_params.get('image_input_width')
        input_height = model_params.get('input_height') or model_params.get('image_input_height')
        
        print(f" -> Model Input: {input_width}x{input_height}")
        print("LOG: Phát hiện PACKED RGB Mode (Dựa trên Hex data từ Web)")

        # ĐỌC ẢNH
        frame = cv2.imread(IMAGE_FILE)
        if frame is None: return
        orig_h, orig_w = frame.shape[:2]

        # 1. Resize
        img_resized = cv2.resize(frame, (input_width, input_height))

        # 2. CHUYỂN ĐỔI MÀU VÀ ĐÓNG GÓI (PACKING) - KHẮC PHỤC LỖI CHÍNH
        # Chuyển BGR (OpenCV) sang RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Dùng Numpy để đóng gói 3 kênh (R, G, B) thành 1 số nguyên (Int32)
        # Công thức: (R << 16) + (G << 8) + B
        img_rgb = img_rgb.astype(np.uint32)
        packed_features = (img_rgb[:, :, 0] << 16) | (img_rgb[:, :, 1] << 8) | img_rgb[:, :, 2]
        
        # Làm phẳng mảng
        features = packed_features.flatten().tolist()

        # DEBUG: In thử 5 giá trị đầu để so sánh với Web (Nó sẽ ra số thập phân lớn, convert sang Hex sẽ giống web)
        print(f"DEBUG: 5 giá trị đầu (Decimal): {features[:5]}")
        print(f"DEBUG: 5 giá trị đầu (Hex): {[hex(x) for x in features[:5]]}")
        
        # 3. Inference
        res = runner.classify(features)

        # 4. Vẽ kết quả
        print("\n--- KẾT QUẢ ---")
        if "bounding_boxes" in res["result"]:
            for bb in res["result"]["bounding_boxes"]:
                val = bb['value']
                label = bb['label']
                print(f"Found: {label} ({val:.2f})")

                if val >= CONFIDENCE_THRESHOLD:
                    cx_model = bb['x'] + (bb['width'] / 2)
                    cy_model = bb['y'] + (bb['height'] / 2)
                    
                    cx_orig = int(cx_model / input_width * orig_w)
                    cy_orig = int(cy_model / input_height * orig_h)

                    cv2.circle(frame, (cx_orig, cy_orig), 8, (0, 255, 0), -1)
                    cv2.putText(frame, f"{val:.2f}", (cx_orig+10, cy_orig), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Hiển thị
        cv2.imshow("Packed RGB Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        if runner: runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])