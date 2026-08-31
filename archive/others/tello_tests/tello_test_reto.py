"""
Tello + Blue-line vision overlay + manual keyboard control
- Vertical mirror (flip around X-axis)
- Low-latency source prefers get_frame_read(); falls back to OpenCV UDP if needed
- Keys:
    SPACE = takeoff
    L     = land
    W/S   = forward/back
    A/D   = left/right
    R/F   = up/down
    Q/E   = yaw left/right
    ESC/Q = quit (safe cleanup)
"""

import cv2
import time
import numpy as np
import keyboard
from djitellopy import Tello

# =========================
# Vision helper functions
# =========================

def detect_patb(frame):
    """Detect a BLUE line within a central ROI; draw box, centroid, and angle."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue range (tune if needed)
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ROI rectangle (bottom-center band)
    x1 = w // 2 - 180
    y1 = h // 2 - 100
    x2 = w // 2 + 180
    y2 = h // 2

    # camera guides
    cv2.circle(frame, (w // 2, h // 2), 6, (0, 255, 0), -1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 170), 2)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 170), 2)

    # draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # mask to ROI only
    roi_mask = np.zeros_like(blue_mask, dtype=np.uint8)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
    mask_roi = cv2.bitwise_and(blue_mask, roi_mask)

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    deg_path = 0
    deg_camara = 0
    cx = None
    cy = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        angle = rect[2]
        # normalize: long side angle
        if rect[1][0] < rect[1][1]:
            angle = angle + 90
        angle = angle - 90

        if angle >= 80 or angle <= -80:
            angle = 0
        if angle > 50 or angle < -50:
            cv2.putText(frame, "Steep angle!", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        deg_path = float(angle)

        cv2.putText(frame, f"Angle: {angle:.1f}", (int(rect[0][0]), int(rect[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # show basic mask (optional)
        cv2.imshow("Mask ROI (blue)", blue_mask)

    return frame, cx, cy, deg_path, deg_camara


def error_calculation(frame, cx, cy, deg_path, deg_camara):
    """Overlay error numbers (angle + pixel offsets) only for visualization."""
    if cx is not None and cy is not None:
        error_angle = deg_path - deg_camara
        error_posx = cx - frame.shape[1] // 2
        error_posy = cy - frame.shape[0] // 2

        cv2.putText(frame, f"Angle Err: {error_angle:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"dx: {error_posx}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"dy: {error_posy}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "No line found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# =========================
# Main
# =========================

def main():
    t = Tello()
    airborne = False

    # Connect and tune stream for latency
    t.connect()
    print("Battery:", t.get_battery())
    try:
        t.set_resolution(t.RESOLUTION_480P)  # or RESOLUTION_720P
    except Exception:
        pass
    try:
        t.set_video_bitrate(t.BITRATE_1M)  # lower bitrate -> fewer drops
    except Exception:
        pass

    # Start stream
    t.streamon()
    time.sleep(1.0)

    # Prefer low-latency reader (PyAV). If it fails on your setup, we’ll fall back.
    use_frame_read = True
    frame_read = None
    try:
        frame_read = t.get_frame_read()  # no arguments in your djitellopy
        _ = frame_read.frame  # touch once
    except Exception as e:
        print("[WARN] get_frame_read failed on this system -> falling back to OpenCV UDP:", e)
        use_frame_read = False
        cap = cv2.VideoCapture(
            "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000&fflags=nobuffer&flags=low_delay",
            cv2.CAP_FFMPEG
        )
        # warm-up
        deadline = time.time() + 4.0
        ok, _ = cap.read()
        while not ok and time.time() < deadline:
            ok, _ = cap.read()
        if not ok:
            print("[ERROR] Could not open video stream from Tello.")
            # we can still fly blind if you want, but better to stop here
            t.streamoff()
            t.end()
            return

    print("SPACE=Takeoff  L=Land  W/A/S/D move  R/F up/down  Q/E yaw  ESC/Q=quit")

    speed = 30  # [-100..100]
    try:
        while True:
            # ----- Read a frame -----
            if use_frame_read:
                frame = frame_read.frame
                if frame is None or frame.size == 0:
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    continue

            # Vertical mirror (flip around X-axis)
            frame = cv2.flip(frame, 0)

            # Resize for consistent processing/display (optional)
            frame = cv2.resize(frame, (640, 480))

            # ---- Vision overlay (no control yet) ----
            frame, cx, cy, deg_path, deg_cam = detect_patb(frame)
            error_calculation(frame, cx, cy, deg_path, deg_cam)

            # ----- Keyboard flight control (manual only) -----
            lr = fb = ud = yaw = 0

            # one-shot actions
            if keyboard.is_pressed('space') and not airborne:
                try:
                    print("Taking off...")
                    t.takeoff()
                    airborne = True
                    time.sleep(0.3)
                except Exception as e:
                    print("Takeoff failed:", e)
                    time.sleep(0.3)

            if keyboard.is_pressed('l') and airborne:
                try:
                    print("Landing...")
                    t.land()
                    airborne = False
                    time.sleep(0.3)
                except Exception as e:
                    print("Land failed:", e)
                    time.sleep(0.3)

            # continuous rc control (only when airborne)
            if airborne:
                if keyboard.is_pressed('w'): fb = speed
                if keyboard.is_pressed('s'): fb = -speed
                if keyboard.is_pressed('a'): lr = -speed
                if keyboard.is_pressed('d'): lr = speed
                if keyboard.is_pressed('r'): ud = speed
                if keyboard.is_pressed('f'): ud = -speed
                if keyboard.is_pressed('q'): yaw = -speed
                if keyboard.is_pressed('e'): yaw = speed
                t.send_rc_control(lr, fb, ud, yaw)
            else:
                # keep motors idle when on ground (prevents SDK timeouts)
                t.send_rc_control(0, 0, 0, 0)

            # ----- Show windows -----
            cv2.imshow("Tello (vertical mirror + vision)", frame)

            # ----- Exit -----
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                break

            # modest loop rate to reduce CPU
            time.sleep(0.03)  # ~33 Hz

    finally:
        # Safe cleanup
        try:
            t.send_rc_control(0, 0, 0, 0)
        except: pass

        if airborne:
            try:
                print("Landing on exit...")
                t.land()
            except: pass

        try:
            t.streamoff()
        except: pass

        try:
            if not use_frame_read:
                cap.release()
        except: pass

        cv2.destroyAllWindows()
        try:
            t.end()
        except: pass


if __name__ == "__main__":
    main()