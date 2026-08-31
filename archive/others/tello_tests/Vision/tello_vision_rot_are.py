import cv2
import numpy as np
import math

def detect_blue_line_angle(frame):

    h, w = frame.shape[:2]

    # 1) BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2) Blue mask (adjust ranges to your line)
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Optional: clean a bit
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # ROI
    x1 = w // 2 - 140
    y1 = h // 2 - 200
    x2 = w // 2 + 140
    y2 = h // 2

    # Guides
    cv2.circle(frame, (w // 2, h // 2), 6, (0, 255, 0), -1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 170), 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 170), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    roi_mask = np.zeros_like(blue_mask, dtype=np.uint8)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
    mask_roi = cv2.bitwise_and(blue_mask, roi_mask)

    # 3) Biggest blue blob
    cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cx = cy = 0
    angle_deg = 0

    if not cnts:
        cv2.putText(frame, "No blue line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        angle_deg = cx = cy = 0
        return frame, angle_deg, cx, cy

    cnt = max(cnts, key=cv2.contourArea)

    # Ignore very small contours
    MIN_AREA = 4000
    if cv2.contourArea(cnt) < MIN_AREA:
        cv2.putText(frame, "Line too small", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        angle_deg = cx = cy = 0
        return frame, angle_deg, cx, cy

    # 4) Fit line to the contour
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)

    # Force direction upwards: vy <= 0
    if vy > 0:
        vx, vy = -vx, -vy

    # 5) Angle relative to "up"
    #   0°  = vertical
    #  +°   = tilted to the right
    #  -°   = tilted to the left
    angle_rad = math.atan2(vx, -vy)
    angle_raw = math.degrees(angle_rad)  # range [-90, +90]

    angle_deg = angle_raw  # Default to raw angle

    # 6) Rectangle and centroid (for visualization)
    if len(cnt) >0:
        c = max(cnt, key=cv2.contourArea)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.polylines(frame, [box], True, (255, 0, 0), 2)

    cx, cy = rect[0]
    cx, cy = int(cx), int(cy)
    
    return frame, angle_deg, cx, cy


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    last_angle = 0  # Track the last valid angle
    error_angle = 0
    angle_deg = 0
    angle_raw = 0

    print("✔ Webcam opened. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read error")
            break

        error_angle = angle_raw - last_angle

        if error_angle > 90 or error_angle < -90:
            angle_raw = last_angle
            angle_deg = angle_raw
        else:
            last_angle = angle_raw  # Update last valid angle
            angle_deg = angle_raw

        frame, angle_raw, cx, cy = detect_blue_line_angle(frame)

        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
        # Show angle
        cv2.putText(frame, f"angle = {angle_deg:.1f}, cx {cx}, cy {cy}",
                (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Blue rect + centroid + angle", frame)
        

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()