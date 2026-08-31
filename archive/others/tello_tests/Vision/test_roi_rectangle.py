import cv2

def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Could not open camera")
        return

    roi_w = 80   # width of each rectangle
    roi_h = 100  # height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

        # Narrow/short ROIs near the TOP of the image
        roi_x = 70
        roi_y = 70
        
        cx1 = w//2 - roi_x
        y1 = h//2 + roi_y
        cx2 = w//2 + roi_x
        y2 = h//2

        lx1 = cx1
        lx2 = cx1 - 2*roi_x

        rx1 = cx2 
        rx2 = cx2 + 2*roi_x


        # Draw rectangles (non-overlapping, touching)
        cv2.rectangle(frame, (lx1, y1), (lx2, y2), (255, 150, 0), 2)   # LEFT
        cv2.rectangle(frame, (cx1, y1), (cx2, y2), (0, 255, 255), 2)   # CENTER
        cv2.rectangle(frame, (rx1, y1), (rx2, y2), (0, 150, 255), 2)   # RIGHT

        # Draw exact center of screen
        cv2.circle(frame, (center_x, center_y), 4, (0,255,0), -1)
        cv2.line(frame, (center_x,0), (center_x,h), (0,255,0), 1)

        cv2.imshow("Aligned ROIs with center touch", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
