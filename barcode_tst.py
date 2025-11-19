from __future__ import print_function
import pyzbar.pyzbar as pyzbar
from pyzbar.pyzbar import ZBarSymbol
import numpy as np
import cv2

# This function is UNCHANGED from your original code
def decode(im) :
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im, symbols=[ZBarSymbol.QRCODE])

  # Print results
  for obj in decodedObjects:
    print('Type : ', obj.type)
    print('Data : ', obj.data,'\n')

  return decodedObjects

# This function is MODIFIED
# It no longer calls imshow or waitKey
def display(im, decodedObjects, status_text="Scanning"):
  cv2.putText(im, f"Mode: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

  # Loop over all decoded objects
  for decodedObject in decodedObjects:
    points = decodedObject.polygon

    # Number of points in the convex hull
    n = len(points)

    # Draw the convex hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
      
  # We return the image with drawings, but DO NOT show it here
  return im

def apply_sharpening(image):
  # Create a sharpening kernel
  kernel = np.array([[0, -1, 0],
                      [-1, 5,-1],
                      [0, -1, 0]])
  return cv2.filter2D(image, -1, kernel)

# Main
if __name__ == '__main__':

  # 1. Create a video capture object
  # 0 is the default webcam
  # CAP_MSMF: windows only driver change for other systems
  cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
  
  # Check if webcam opened successfully
  if not cap.isOpened():
      print("Error: Could not open video device.")
      exit()

  print("Starting video stream... Press 'q' to quit.")

  N = 5
  frame_counter = 0 
  decodedObjects = []
  current_mode_text = "Waiting"

  # 2. Create a continuous loop
  while True:
    
    # 3. Read a frame from the webcam
    ret, frame = cap.read()

    # If frame is not read correctly, break the loop
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)
    frame_counter += 1
    # 4. Decode the frame
    if frame_counter % N == 0:
      current_mode_text = "Standard"
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      decodedObjects = decode(gray)
      
      if not decodedObjects:
        thresh_im = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        decodedObjects = decode(thresh_im)
        if decodedObjects:
          current_mode_text = "Adaptive Thresholding"

      if not decodedObjects:
        sharpened = apply_sharpening(gray)
        decodedObjects = decode(sharpened)
        if decodedObjects:
          current_mode_text = "Sharpening"

      if not decodedObjects:
          current_mode_text = "Searching..."

    frame = display(frame, decodedObjects, current_mode_text)

    cv2.imshow("Live QR Code Scanner", frame)

    if cv2.waitKey(1) == ord('q'):
      break

  print("Stopping stream.")
  cap.release()
  cv2.destroyAllWindows()