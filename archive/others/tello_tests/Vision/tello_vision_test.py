import keyboard 
import time
import cv2 
import numpy as np
import math

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Manual keyboard test
def check_keys():
    if keyboard.is_pressed('w'):
        print("W pressed")
        time.sleep(0.1)
    if keyboard.is_pressed('a'):
        print("A pressed")
        time.sleep(0.1)
    if keyboard.is_pressed('d'):
        print("D pressed")
        time.sleep(0.1)
    if keyboard.is_pressed('s'):
        print("S pressed")
        time.sleep(0.1)
    if keyboard.is_pressed('x'):
        print("Bye")
        time.sleep(0.5)


# ========================================================
# BLUE PATH DETECTION (3 ROIs + CENTROIDS + ANGLES)
# ========================================================
def detect_patb(frame):

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue threshold
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ========= ROI Sizes ==========
    roi_width = 40
    roi_height = 100

    # Center ROI
    cx1 = w//2 - roi_width
    cy1 = h//2 - roi_height
    cx2 = w//2 + roi_width
    cy2 = h//2

    # Left ROI
    lx1 = w//2 - 3*roi_width
    ly1 = cy1
    lx2 = w//2 - roi_width
    ly2 = cy2

    # Right ROI
    rx1 = w//2 + roi_width
    ry1 = cy1
    rx2 = w//2 + 3*roi_width
    ry2 = cy2

    # Draw rectangles
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 100, 0), 2)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 100, 255), 2)

    # =====================================================
    # Helper: return contour, centroid, angle of that ROI
    # =====================================================
    def process_roi_and_angle(mask, x1, y1, x2, y2):
        roi_mask = np.zeros_like(mask)
        cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)

        masked = cv2.bitwise_and(mask, roi_mask)
        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, None, None, None

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        if M['m00'] == 0:
            return c, None, None, None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

        # Angle calculation
        rect = cv2.minAreaRect(c)
        angle = rect[2]

        if rect[1][0] < rect[1][1]:  # width < height
            angle += 90
        angle -= 90

        if angle >= 80 or angle <= -80:
            angle = 0

        return c, cx, cy, angle

    # Process each ROI
    cC, cxC, cyC, angleC = process_roi_and_angle(blue_mask, cx1, cy1, cx2, cy2)
    cL, cxL, cyL, angleL = process_roi_and_angle(blue_mask, lx1, ly1, lx2, ly2)
    cR, cxR, cyR, angleR = process_roi_and_angle(blue_mask, rx1, ry1, rx2, ry2)

    # Show angles on screen
    if angleC is not None:
        cv2.putText(frame, f"C Angle: {angleC:.1f}", (10, 230), 0, 0.6, (0,255,255), 2)
    if angleL is not None:
        cv2.putText(frame, f"L Angle: {angleL:.1f}", (10, 250), 0, 0.6, (255,150,0), 2)
    if angleR is not None:
        cv2.putText(frame, f"R Angle: {angleR:.1f}", (10, 270), 0, 0.6, (0,150,255), 2)

    return frame, cxC, cyC, angleC, cxL, cyL, angleL, cxR, cyR, angleR


# ========================================================
# ERROR CALCULATION (uses center centroid)
# ========================================================
def error_calculation(frame, cx, cy, angleC):
    if cx is not None:
        error_angle = angleC
        error_posx = cx - frame.shape[1] // 2
        error_posy = cy - frame.shape[0] // 2

        cv2.putText(frame, f"Angle: {error_angle:.2f}", (10, 30), 0, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"X Diff: {error_posx:.2f}", (10, 60), 0, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Y Diff: {error_posy:.2f}", (10, 90), 0, 0.7, (0,255,255), 2)
    else:
        cv2.putText(frame, "No center path found", (10, 30), 0, 0.7, (0,255,255), 2)


# ========================================================
# MAIN LOOP
# ========================================================
def main():
    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (480, 320))

            if not ret or frame is None:
                print("Camera read error")
                break

            # Now we get all centroid + angle values
            frame, cxC, cyC, angleC, cxL, cyL, angleL, cxR, cyR, angleR = detect_patb(frame)

            error_calculation(frame, cxC, cyC, angleC)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            check_keys()

        cap.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("Program stopped by user.")


if __name__ == "__main__":
    main()
