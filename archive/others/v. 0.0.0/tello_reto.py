"""
Tello + BLUE line + P-control with staged behavior:
1) First align yaw to the line (angle error -> yaw).
2) Only when |angle_err| ≤ ALIGN_DEG, allow centroid-based lr/fb.

- lr  = Kp_lr  * err_x         (only when aligned)
- fb  = Kp_fb  * (-err_y)      (only when aligned)
- yaw = Kp_yaw * angle_err     (always, with deadband)

Keys:
  SPACE = takeoff
  L     = land
  T     = toggle FOLLOW (P-control) [OFF by default]
  W/A/S/D, R/F, Q/E = manual when FOLLOW is OFF
  ESC/Q = quit
"""

import cv2
import time
import numpy as np
import keyboard
from djitellopy import Tello

# =======================
# Gains, thresholds & limits
# =======================
Kp_lr    = 0.1     # px -> stick (left/right)
Kp_fb    = 0.1     # px -> stick (forward/back)
Kp_yaw   = 2     # deg -> stick (yaw turn)

DEAD_PIX = 6        # deadband (px) for lr/fb
DEAD_DEG = 2.0      # deadband (deg) for yaw command
ALIGN_DEG = 0.5     # window: only if |angle_err| <= ALIGN_DEG we allow lr/fb

MAX_STICK = 70      # clamp magnitude per axis (<=100)
AREA_MIN = 120      # ignore tiny blobs

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# =======================
# BLUE line detector (centroid + angle)
# =======================
def detect_blue_line_and_angle(frame):
    """
    Returns: frame (drawn), cx, cy, angle_deg
    angle_deg: orientation of the line (longer side), ~[-90, +90], + = CCW
    cx, cy: centroid (None if not found)
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

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

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = cy = None
    angle_deg = 0.0

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > AREA_MIN:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # Normalize angle to longer side, into [-90, +90]
            raw_angle = rect[2]                 # [-90, 0)
            w_rect, h_rect = rect[1]
            if w_rect < h_rect:
                raw_angle += 90.0
            angle_deg = raw_angle - 90.0
            if angle_deg < -90: angle_deg += 180
            if angle_deg >  90: angle_deg -= 180
            if angle_deg >= 90 or angle_deg <= -90:
                angle_deg = 0.0                 # reject near-vertical artifacts

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(frame, f"angle: {angle_deg:+.1f} deg",
                        (int(rect[0][0]), int(rect[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame, cx, cy, angle_deg

# =======================
# Main
# =======================
def main():
    t = Tello()
    airborne = False
    follow = False  # start manual

    t.connect()
    print("Battery:", t.get_battery())

    # Latency-friendly settings (best-effort)
    try: t.set_resolution(t.RESOLUTION_480P)
    except: pass
    try: t.set_video_bitrate(t.BITRATE_1M)
    except: pass

    t.streamon()
    time.sleep(1.0)

    # Use low-latency frame reader if available
    use_frame_read = True
    frame_read = None
    cap = None
    try:
        frame_read = t.get_frame_read()
        _ = frame_read.frame
    except Exception as e:
        print("[WARN] get_frame_read failed; fallback to VideoCapture:", e)
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
            print("[ERROR] Could not open video stream.")
            t.streamoff(); t.end()
            return

    print("SPACE=Takeoff  L=Land  T=Follow toggle  (WASD/RF/QE when Follow OFF)  ESC/Q=quit")
    speed_manual = 30

    try:
        while True:
            # --- frame in ---
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
            frame = cv2.resize(frame, (640, 480))
            h, w = frame.shape[:2]
            cx_img, cy_img = w // 2, h // 2

            # --- vision ---
            frame, cx, cy, angle_deg = detect_blue_line_and_angle(frame)

            # --- pixel errors ---
            err_x = err_y = None
            if cx is not None and cy is not None:
                err_x = cx - cx_img
                err_y = cy - cy_img
                cv2.arrowedLine(frame, (cx_img, cy_img), (cx, cy),
                                (0, 255, 255), 2, tipLength=0.25)
                cv2.putText(frame, f"err_x: {err_x:+d}px", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"err_y: {err_y:+d}px", (10, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "No line detected", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"angle: {angle_deg:+.1f}", (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"FOLLOW: {'ON' if follow else 'OFF'}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

            # --- one-shot keys ---
            if keyboard.is_pressed('space') and not airborne:
                try:
                    print("Taking off...")
                    t.takeoff(); airborne = True
                    time.sleep(0.3)
                except Exception as e:
                    print("Takeoff failed:", e)
                    time.sleep(0.3)

            if keyboard.is_pressed('l') and airborne:
                try:
                    print("Landing...")
                    t.land(); airborne = False
                    time.sleep(0.3)
                except Exception as e:
                    print("Land failed:", e)
                    time.sleep(0.3)

            if keyboard.is_pressed('t'):
                follow = not follow
                print("Follow:", "ON" if follow else "OFF")
                time.sleep(0.25)

            # --- RC build ---
            lr = fb = ud = yaw = 0

            if airborne and follow:
                # --- Yaw from ANGLE ERROR (always active with deadband) ---
                angle_err = angle_deg
                yaw = 0 if abs(angle_err) <= DEAD_DEG else int(clamp(Kp_yaw * angle_err, -MAX_STICK, MAX_STICK))

                # --- Only allow lr/fb when yaw is nearly aligned ---
                if abs(angle_err) <= ALIGN_DEG and (err_x is not None and err_y is not None):
                    # Left/Right (x error)
                    if abs(err_x) <= DEAD_PIX:
                        lr = 0
                    else:
                        lr = int(clamp(Kp_lr * err_x, -MAX_STICK, MAX_STICK))

                    # Forward/Back (y error; invert sign to make "line below center" => move forward)
                    if abs(err_y) <= DEAD_PIX:
                        fb = 0
                    else:
                        fb = int(clamp(Kp_fb * (-err_y), -MAX_STICK, MAX_STICK))
                else:
                    # Not aligned yet: freeze translations
                    lr = 0
                    fb = 0

                t.send_rc_control(lr, fb, ud, yaw)

                # HUD of applied sticks
                cv2.putText(frame, f"lr:{lr:+d} fb:{fb:+d} ud:{ud:+d} yaw:{yaw:+d}",
                            (w - 330, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 255), 2)

            elif airborne:
                # Manual when FOLLOW is off
                if keyboard.is_pressed('w'): fb =  speed_manual
                if keyboard.is_pressed('s'): fb = -speed_manual
                if keyboard.is_pressed('a'): lr = -speed_manual
                if keyboard.is_pressed('d'): lr =  speed_manual
                if keyboard.is_pressed('r'): ud =  speed_manual
                if keyboard.is_pressed('f'): ud = -speed_manual
                if keyboard.is_pressed('q'): yaw = -speed_manual
                if keyboard.is_pressed('e'): yaw =  speed_manual
                t.send_rc_control(lr, fb, ud, yaw)
            else:
                # keep link alive on ground
                t.send_rc_control(0, 0, 0, 0)

            # --- show ---
            cv2.imshow("Tello (vertical mirror) - Align->Translate P control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            time.sleep(0.02)  # ~50 Hz loop

    finally:
        try: t.send_rc_control(0,0,0,0)
        except: pass
        if airborne:
            try:
                print("Landing on exit...")
                t.land()
            except: pass
        try: t.streamoff()
        except: pass
        try:
            if cap is not None: cap.release()
        except: pass
        cv2.destroyAllWindows()
        try: t.end()
        except: pass

if __name__ == "__main__":
    main()
