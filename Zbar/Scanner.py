from __future__ import print_function
import pyzbar.pyzbar as pyzbar
from pyzbar.pyzbar import ZBarSymbol
import numpy as np
import cv2

def decode(im) :
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im, symbols=[ZBarSymbol.QRCODE])

  for obj in decodedObjects:
    print('Type : ', obj.type)
    print('Data : ', obj.data,'\n')

  return decodedObjects

def display(im, decodedObjects, status_text="Scanning"):
  cv2.putText(im, f"Mode: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

  for decodedObject in decodedObjects:
    points = decodedObject.polygon

    n = len(points)

    # Draw the convex hull
    for j in range(0,n):
      cv2.line(im, points[j], points[ (j+1) % n], (255,0,0), 3)

  # We return the image with drawings
  return im

if __name__ == '__main__':

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

  while True:

    # Read a frame from the webcam
    ret, frame = cap.read()

    # If frame is not read correctly, break the loop
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)
    frame_counter += 1
    # Decode every fifth frame (for optimization purposes)
    if frame_counter % N == 0:
      current_mode_text = "Standard"
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      decodedObjects = decode(gray)

      if not decodedObjects:
          current_mode_text = "Searching..."

    frame = display(frame, decodedObjects, current_mode_text)

    cv2.imshow("Live QR Code Scanner", frame)

    if cv2.waitKey(1) == ord('q'):
      break

  print("Stopping stream.")
  cap.release()
  cv2.destroyAllWindows()
