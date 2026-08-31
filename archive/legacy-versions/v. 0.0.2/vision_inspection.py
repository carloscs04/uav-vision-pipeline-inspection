import cv2
import numpy as np
import math

def detect_white_line_angle(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 245])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    x1 = w // 2 - 200
    y1 = h // 2 - 280
    x2 = w // 2 + 200
    y2 = h // 2 - 90

    cv2.circle(frame, (w // 2, h // 2), 6, (0, 255, 0), -1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 170), 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 170), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # ROI azul para PID

    roi_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
    mask_roi = cv2.bitwise_and(mask, roi_mask)

    cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cx = cy = 0
    angle_deg = 0
    detected = False

    if not cnts:
        cv2.putText(frame, "No white line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, angle_deg, cx, cy, detected

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1700:
        cv2.putText(frame, "Line too small", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, angle_deg, cx, cy, detected

    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)
    if vy > 0:
        vx, vy = -vx, -vy

    angle_rad = math.atan2(vx, -vy)
    angle_deg = math.degrees(angle_rad)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.polylines(frame, [box], True, (255, 0, 0), 2)

    cx, cy = rect[0]
    w, h = rect[1]
    cx = int(cx)
    cy = int(cy)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    if abs(h/w) > 0.85:
        angle_deg = 0
        print("hola")

    return frame, angle_deg, cx, cy, True

def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✔ Webcam opened. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read error")
            break

        frame, angle_deg, cx, cy, a = detect_white_line_angle(frame)

        # Example usage for the drone:
        # if angle is not None:
        #     yaw_error = angle                  # [-90..90], saturated at ±90
        #     yaw_speed = int(-0.4 * yaw_error)
        #     yaw_speed = max(-50, min(50, yaw_speed))
        #     t.send_rc_control(0, 0, 0, yaw_speed)

        cv2.putText(frame, f"Angle: {angle_deg:.1f} deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Blue rect + centroid + angle", frame)
        

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
