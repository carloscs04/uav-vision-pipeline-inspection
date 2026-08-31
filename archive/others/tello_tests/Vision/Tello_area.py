import cv2
import numpy as np

def detect_line_3roi(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue range
    lower = np.array([90, 60, 60])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Narrow/short ROIs near the TOP of the image
    roi_x = 45
    roi_y = 100
    
    cx1 = w//2 - roi_x
    cy1 = h//2 - 70     ## Offset Altura cuadro central
    cy2 = h//2 - roi_y  ## Altura cuadro central
    y1 = h//2 - roi_y - 15   ## Altura cuadros laterales
    cx2 = w//2 + roi_x   ## Ancho de cuadro central
    y2 = h//2 - 70     ## Offset ancho cuadros laterales

    lx1 = cx1
    lx2 = cx1 - 2*roi_x -20  ## Anchos de laterales

    rx1 = cx2 
    rx2 = cx2 + 2*roi_x + 20    ## Anchos de laterales

    # Left last rectangle
    llx1 = lx2 
    llx2 = lx2 + 10  # Width of the rectangle

    # Left last rectangle
    lrx1 = rx2 
    lrx2 = rx2 - 10 # Width of the rectangle

    # Draw ROIs
    cv2.rectangle(frame, (lx1, y1), (lx2, y2), (255, 100, 0), 2)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
    cv2.rectangle(frame, (rx1, y1), (rx2, y2), (0, 100, 255), 2)
    cv2.rectangle(frame, (lrx1, y1), (lrx2, y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (llx1, y1), (llx2, y2), (0,255,0), 2)

    # Helper for each ROI
    def roi_area(x1, y1, x2, y2):
        # Ensure proper ordering
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        
        roi = mask[y1:y2, x1:x2]
        area = cv2.countNonZero(roi)
        if area > 4000:
            area = 4000  # Cap area to max value for consi8tency
        area = area / 4000  # Normalize to 0-1
        area = int(area * 80)
        flag = False
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"]) + x1
                cy = int(M["m01"]/M["m00"]) + y1
                cv2.circle(frame, (cx,cy), 4, (255,255,255), -1)
            flag = True
        else:
            flag = False

        return flag, area

    has_left, area_left,    = roi_area(lx1, y1, lx2, y2)
    has_center, area_center = roi_area(cx1, cy1, cx2, cy2)
    has_right, area_right  = roi_area(rx1, y1, rx2, y2)
    has_last_left, area_last_left  = roi_area(llx1, y1, llx2, y2)
    has_last_right, area_last_right  = roi_area(lrx1, y1, lrx2, y2)

    cv2.putText(frame, f"AL:{area_left} AR:{area_right} L:{has_left} C:{has_center} R:{has_right} LL:{has_last_left} LR:{has_last_right} ", 
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    return frame, area_left, area_right, has_left, has_center, has_right, has_last_left, has_last_right

# Example usage:
def main():
    cap = cv2.VideoCapture(1)   # webcam

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✔ Webcam opened. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read error")
            break

        # CALL YOUR FUNCTION HERE
        frame, areaL, areaR, hasL, hasC, hasR, hasLL, hasLR = detect_line_3roi(frame)

        # Show frame with rectangles + areas
        cv2.imshow("Webcam ROI Detection", frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()