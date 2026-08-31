import keyboard 
import time
import cv2 
import numpy as np
import math

cap = cv2.VideoCapture(1)

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height


# Manual Control via Keyboard
def check_keys():
    if keyboard.is_pressed('w'):
        print("The 'w' key is pressed.")
        time.sleep(0.1)
    if keyboard.is_pressed('a'):
        print("The 'a' key is pressed.")
        time.sleep(0.1)
    if keyboard.is_pressed('d'):
        print("The 'd' key is pressed.")
        time.sleep(0.1)
    if keyboard.is_pressed('s'):
        print("The 's' key is pressed.")
        time.sleep(0.1)

    if keyboard.is_pressed('x'):
        print("Bye")
        time.sleep(0.5)

## Visual Detection Functions ##

# Function to detect blue and red lines
def detect_patb(frame):

    # 1) Convert to HSV
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2) Threshold for BLUE
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 3) ROI rectangel
    x1 = w//2 - 180
    y1 = h//2 - 100
    x2 = w//2 + 180
    y2 = h//2

    # Drawing camera reference lines and center
    cv2.circle(frame, (w//2, h//2), 6, (0, 255, 0), -1)
    cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 170), 2)
    cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 170), 2)

    # 4) Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # 5) Create rectangle mask
    roi_mask = np.zeros(blue_mask.shape, dtype=np.uint8)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)

    # 6) Apply ROI to color mask
    mask_roi = cv2.bitwise_and(blue_mask, roi_mask)

    # 7) Find contours of the blue region in mask_roi
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Defaults
    deg_path = 0
    deg_camara = 0
    cx = 0
    cy = 0

    if len(contours) > 0:
        # Tomar contorno más grande (la línea)
        c = max(contours, key=cv2.contourArea)      

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        angle = rect[2]

        if rect[1][0] < rect[1][1]:   # width < height
            angle = angle + 90

        angle = angle - 90

        if angle >= 80 or angle <= -80:
            angle = 0
        
        if angle > 50 or angle < -50:
            print("Be careful, steep angle detected:", angle)

        deg_path = angle

        cv2.putText(frame, f"Angle: {angle:.2f}", (int(rect[0][0]), int(rect[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Calcular centro de la línea
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Dibujar centroide
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        cv2.imshow("Mask ROI", blue_mask)
        

    # Return frame, center, and angle of the LONG side
    return frame, cx, cy, deg_path, deg_camara

def error_calculation(frame, cx, cy, deg_path, deg_camara):
    if cx is not None:
        error_angle = deg_path - deg_camara
        error_posx = cx - frame.shape[1] // 2
        error_posy = cy - frame.shape[0] // 2

        cv2.putText(frame, f"Angle Diff: {error_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Angle Diff: {error_posx:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Angle Diff: {error_posy:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "No angle found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def main():
    try:
        # Main loop to capture frames and process them
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.resize(frame, (480, 320))

            if not ret or frame is None:
                print("Failed to read frame from camera")
                break

            # Process the frame to detect blue and red lines
            frame, cx, cy, deg_path, deg_camara = detect_patb(frame)
            
            # Print the angle difference
            error_calculation(frame, cx, cy, deg_path, deg_camara)

            # Display the processed frame
            cv2.imshow("Frame with Detected Lines", frame)
            
            # Exit the loop when the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            check_keys()

        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
            
    except KeyboardInterrupt:
        print("Program terminated by user.")

if __name__ == "__main__":
    main()