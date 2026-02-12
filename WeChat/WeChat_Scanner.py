import cv2
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

detect_proto = os.path.join(current_dir, "detect.prototxt")
detect_model = os.path.join(current_dir, "detect.caffemodel")
sr_proto = os.path.join(current_dir, "sr.prototxt")
sr_model = os.path.join(current_dir, "sr.caffemodel")

# Initialize WeChat Detector
detector = cv2.wechat_qrcode_WeChatQRCode(
    detect_proto, detect_model, sr_proto, sr_model
)

cap = cv2.VideoCapture(0)

# Force high-res
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify what res we actually got
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera Resolution set to: {int(actual_w)}x{int(actual_h)}")

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE helps with lighting gradients caused by tilting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Convert back to BGR (WeChat detector expects 3 channels usually)
    input_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Detection function
    res, points = detector.detectAndDecode(input_frame)

    # If detection found print and display it (also adds bounding boxes)
    if len(res) > 0:
        for i, content in enumerate(res):
            print(f"Detected: {content}")
            pts = np.int32(points[i]).reshape(-1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            cv2.putText(frame, content, (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Adds how the frame looks like to the computer as well as user side-by-side
    scale = 0.5 # Scale down for display if 1080p fits poorly on screen
    small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    small_enhanced = cv2.resize(input_frame, (0,0), fx=scale, fy=scale)

    combined = np.hstack((small_frame, small_enhanced))
    return combined

# cv2 inference loop
while True:
    ret, frame = cap.read()
    if not ret: break

    debug_view = process_frame(frame)
    cv2.imshow('Left: Normal | Right: Enhanced (Computer Vision)', debug_view)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()