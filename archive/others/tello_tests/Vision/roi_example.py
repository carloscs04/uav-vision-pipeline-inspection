import cv2
import numpy as np

# --- Parámetros del color azul en HSV ---
lower_blue = np.array([90, 60, 60])
upper_blue = np.array([130, 255, 255])

cap = cv2.VideoCapture(0)   # 0 = cámara web

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (480, 320))
    h, w = frame.shape[:2]

    # ========= 1. COLOR MASK =========
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ========= 2. DEFINE ROI RECTANGLE =========
    x1 = w//2 - 100
    x2 = w//2 + 100
    y1 = h//2
    y2 = h//2 + 100

    # Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # ========= 3. CREATE RECTANGLE MASK =========
    roi_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)

    # ========= 4. APPLY ROI TO COLOR MASK =========
    mask_roi = cv2.bitwise_and(mask, roi_mask)

    # ========= 5. FIND CONTOURS IN mask_roi =========
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)

        # ---- PCA ORIENTATION (recommended) ----
        pts = c.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
        vx, vy = eigenvectors[0]

        raw_angle = np.degrees(np.arctan2(vy, vx))
        angle = raw_angle + 90
        if angle > 180: angle -= 360
        if angle < -180: angle += 360

        cx, cy = int(mean[0][0]), int(mean[0][1])

        cv2.putText(frame, f"Angle: {angle:.2f}", (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # ========= 6. CENTROID =========
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
            cv2.line(frame, (w//2, 0), (w//2, h), (0,255,255), 2)

            error = cx - w//2

            if error > 30:
                direction = "RIGHT"
            elif error < -30:
                direction = "LEFT"
            else:
                direction = "FORWARD"

            cv2.putText(frame, direction, (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Mask ROI", mask_roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
