import cv2
import numpy as np

# Initialize WeChat Detector
detector = cv2.wechat_qrcode_WeChatQRCode(
    "detect.prototxt", "detect.caffemodel",
    "sr.prototxt", "sr.caffemodel"
)

# Initialize Camera
cap = cv2.VideoCapture(0)

# --- FIX 1: FORCE HIGH RESOLUTION ---
# OpenCV defaults to 640x480. We need more pixels to see tilted details.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify what we actually got (some webcams maximize at 720p)
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera Resolution set to: {int(actual_w)}x{int(actual_h)}")

def process_frame(frame):
    # --- FIX 2: PRE-PROCESSING ---
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This helps ENORMOUSLY with lighting gradients caused by tilting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Convert back to BGR (WeChat detector expects 3 channels usually)
    input_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Detect
    res, points = detector.detectAndDecode(input_frame)

    if len(res) > 0:
        for i, content in enumerate(res):
            print(f"Detected: {content}")
            pts = np.int32(points[i]).reshape(-1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            cv2.putText(frame, content, (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the "Enhanced" view so you can see what the computer sees
    # (Optional: stack them side-by-side)
    scale = 0.5 # Scale down for display if 1080p fits poorly on screen
    small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    small_enhanced = cv2.resize(input_frame, (0,0), fx=scale, fy=scale)

    combined = np.hstack((small_frame, small_enhanced))
    return combined

while True:
    ret, frame = cap.read()
    if not ret: break

    debug_view = process_frame(frame)
    cv2.imshow('Left: Normal | Right: Enhanced (Computer Vision)', debug_view)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()