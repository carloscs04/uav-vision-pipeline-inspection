import cv2
import numpy as np
from pupil_apriltags import Detector

# =======================
# AprilTag detector setup
# =======================

tag_detector = Detector(
    families="tag36h11",   # make sure your printed tag is this family
    nthreads=1,
    quad_decimate=0.2,     # smaller -> better for small/far tags, slower
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25
)

def detect_apriltag(frame):
    # Safety
    if frame is None:
        print("detect_apriltag: frame is None")
        return False, None, None, None, None

    # Convert BGR -> GRAY (required by detector)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run detection
    tags = tag_detector.detect(gray)
    print("detect_apriltag: tags found =", len(tags))

    if len(tags) == 0:
        return False, None, None, None, None

    # Choose the tag with best decision margin (most confident)
    tag = max(tags, key=lambda t: t.decision_margin)

    cx, cy = tag.center
    tag_id = tag.tag_id

    (ptA, ptB, ptC, ptD) = tag.corners
    dx = ptB[0] - ptA[0]
    dy = ptB[1] - ptA[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Draw on the frame you are showing
    cv2.polylines(frame, [np.int32(tag.corners)], True, (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
    cv2.putText(frame, f"TAG {tag_id}", (int(cx) - 20, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Also draw angle for debugging
    cv2.putText(frame, f"ang={angle:.1f}", (int(cx) - 20, int(cy) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return True, int(tag_id), int(cx), int(cy), angle

# =======================
# Main webcam loop
# =======================

def main():
    cap = cv2.VideoCapture(1)  # 0 = default webcam
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam")
            break

        # Detect AprilTag in the frame
        detected, tag_id, cx, cy, angle = detect_apriltag(frame)

        if detected:
            cv2.putText(frame, f"Detected TAG ID: {tag_id}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No tag detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("AprilTag tag36h11 Webcam Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
