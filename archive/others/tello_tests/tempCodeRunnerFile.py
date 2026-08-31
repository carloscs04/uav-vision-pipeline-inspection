"""
Tello system (NO PyAV, FAST, Manual + Auto):

AUTO logic:
1) ALIGN phase:
   - Detect largest BLUE line in frame
   - Compute angle with YOUR logic
   - Only rotate (yaw) until angle ~ 0
   - GREEN is also detected (for visualization)  

2) GOTO_GREEN phase:
   - Still detect BLUE + GREEN
   - Keep angle small with yaw
   - Now move in X,Y toward GREEN centroid

3) FOLLOW_LINE phase:
   - 3 ROIs at TOP (LEFT, CENTER, RIGHT)
   - Follow BLUE line using the ROIs
   - Draw centroids inside each ROI

Manual mode:
- W/S/A/D = forward/back/left/right
- R/F = up/down
- Q/E = yaw left/right

Keys:
  SPACE = takeoff
  L     = land
  T     = toggle AUTO (ALIGN→GOTO_GREEN→FOLLOW_LINE)
  ESC/Q = quit
"""

import cv2
import time
import numpy as np
import keyboard
from djitellopy import Tello
import threading

# =====================================================================
# REAL-TIME VIDEO THREAD (OpenCV only, no PyAV)
# =====================================================================

frame_global = None

def video_thread():
    """Reads Tello video stream using OpenCV with minimal delay."""
    global frame_global

    cap = cv2.VideoCapture(
        "udp://0.0.0.0:11111",
        cv2.CAP_FFMPEG
    )

    if not cap.isOpened():
        print("[ERROR] Could not open video stream.")
        return

    while True:
        ok, frame = cap.read()
        if ok:
            frame_global = cv2.resize(frame, (640, 480))
        else:
            time.sleep(0.01)


# =====================================================================
# HELPERS
# =====================================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =====================================================================
# BLUE ANGLE DETECTOR (YOUR EXACT ANGLE LOGIC)
# =====================================================================

def detect_blue_angle(frame):
    """
    Detect largest blue contour and compute angle using YOUR formula.

    Returns:
        frame (with box drawn),
        angle (float, degrees) or None if not found
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 60, 60])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angle = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

        angle_raw = rect[2]       # OpenCV angle
        w_rect, h_rect = rect[1]

        # --- YOUR ANGLE LOGIC ---
        angle_calc = angle_raw
        if w_rect < h_rect:       # width < height
            angle_calc = angle_calc + 90

        angle_calc = angle_calc - 90

        if angle_calc >= 75 or angle_calc <= -75:
            angle_calc = 0

        angle = angle_calc

        cv2.putText(frame, f"Blue angle: {angle:+.1f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

    return frame, angle


# =====================================================================
# GREEN DETECTOR (unchanged)
# =====================================================================

def detect_green(frame):
    """
    Detects a green blob in the frame and returns its centroid.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 60, 60])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = cy = None
    found = False

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            found = True

            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    return frame, cx, cy, found


# =====================================================================
# BLUE LINE 3 ROIs WITH CENTROIDS  (TOP, not tall)
# =====================================================================

def detect_line_3roi(frame):
    """
    Detects blue presence in 3 ROIs at the TOP: LEFT, CENTER, RIGHT.
    Draws centroid of largest contour in each ROI.

    Returns:
        frame, has_left, has_center, has_right
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 60, 60])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Narrow/short ROIs near the TOP of the image
    roi_x = 70
    roi_y = 100
    
    cx1 = w//2 - roi_x
    cy1 = h//2 - 70     ## Offset Altura cuadro central
    cy2 = h//2 - roi_y  ## Altura cuadro central
    y1 = h//2 - roi_y - 15   ## Altura cuadros laterales
    cx2 = w//2 + roi_x   ## Ancho de cuadro central
    y2 = h//2 - 70     ## Offset ancho cuadros laterales

    lx1 = cx1
    lx2 = cx1 - roi_x - 20  ## Anchos de laterales

    rx1 = cx2 
    rx2 = cx2 + roi_x + 20    ## Anchos de laterales

    # Draw ROIs
    cv2.rectangle(frame, (lx1, y1), (lx2, y2), (255, 100, 0), 2)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
    cv2.rectangle(frame, (rx1, y1), (rx2, y2), (0, 100, 255), 2)

    def check_roi(mask, x1, y1, x2, y2, color):
        roi_mask = np.zeros_like(mask)
        cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
        masked = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 80:
            return False

        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 4, color, -1)

        return True

    has_left = check_roi(mask, lx1, y1, lx2, y2, (255, 255, 255))
    has_center = check_roi(mask, cx1, cy1, cx2, cy2, (255, 255, 0))
    has_right = check_roi(mask, rx1, y1, rx2, y2, (255, 255, 255))

    txt = f"L:{int(has_left)} C:{int(has_center)} R:{int(has_right)}"
    cv2.putText(frame, txt, (10, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame, has_left, has_center, has_right


# =====================================================================
# MAIN
# =====================================================================

def main():
    global frame_global

    t = Tello()
    t.connect()
    print("Battery:", t.get_battery())

    t.streamon()
    time.sleep(1.0)

    # Start video thread
    threading.Thread(target=video_thread, daemon=True).start()

    airborne = False
    auto = False
    phase = "ALIGN"  # ALIGN -> GOTO_GREEN -> FOLLOW_LINE

    print("\nControls:")
    print(" SPACE = takeoff")
    print(" L     = land")
    print(" T     = toggle AUTO")
    print(" WASD/RF/QE = manual")
    print(" ESC/Q = quit\n")

    max_yaw = 25
    max_fb = 20
    max_lr = 20
    align_threshold = 2.0  # deg



    try:
        while True:
            frame = frame_global
            # keep your vertical mirror exactly as requested
            frame = cv2.flip(frame, 0)
            if frame is None:
                time.sleep(0.01)
                continue

            f = frame.copy()
            h, w = f.shape[:2]
            cx_img = w // 2
            cy_img = h // 2
            cv2.circle(f, (cx_img, cy_img), 4, (0, 255, 0), -1)
 
            # ---------------- BASIC KEYS ----------------
            if keyboard.is_pressed("space") and not airborne:
                t.takeoff()
                airborne = True
                auto = False
                phase = "ALIGN"
                time.sleep(0.3)

            if keyboard.is_pressed("l") and airborne:
                t.land()
                airborne = False
                auto = False
                time.sleep(0.3)

            if keyboard.is_pressed("t"):
                auto = not auto
                phase = "ALIGN"
                print("AUTO =", auto)
                time.sleep(0.3)

            if keyboard.is_pressed("esc") or keyboard.is_pressed("q"):
                break

            lr = fb = ud = yaw = 0

            # ====================== AUTO MODE ======================
            if airborne and auto:

                # ---------- PHASE 1: ALIGN TO BLUE ANGLE ----------
                if phase == "ALIGN":
                    f, angle = detect_blue_angle(f)
                    f, gx, gy, green_found = detect_green(f)

                    cv2.putText(f, "PHASE: ALIGN", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if angle is None:
                        yaw = 14
                        time.sleep(0.5)
                    else:
                        yaw = int(clamp(angle * 0.8, -max_yaw, max_yaw))
                        if yaw <= 3 and yaw >= 0:
                            yaw = 5
                        if yaw <= 0 and yaw >= -3:
                            yaw = -5
                        if abs(angle) < align_threshold and green_found:
                            print(">>> ALIGN done + GREEN seen -> GOTO_GREEN")
                            phase = "GOTO_GREEN"

                    t.send_rc_control(lr, fb, ud, yaw)

                # ---------- PHASE 2: GO TO GREEN ----------
                elif phase == "GOTO_GREEN":
                    f, angle = detect_blue_angle(f)
                    f, gx, gy, green_found = detect_green(f)

                    if abs(angle) > align_threshold and green_found or green_found is False:
                            print(">>> ALIGN done + GREEN seen -> GOTO_GREEN")
                            phase = "ALIGN"
                            continue

                    cv2.putText(f, "PHASE: GOTO_GREEN", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if green_found:
                        dx = gx - cx_img
                        dy = gy - cy_img

                        if angle is None or abs(angle) > align_threshold:
                            lr = 0
                            fb = 0
                        else:
                            lr = int(clamp(dx * 0.12, -max_lr, max_lr))
                            fb = int(clamp(-dy * 0.15, -max_fb, max_fb))
                            if lr <= 3 and lr >= 0:
                                lr = 7
                            if lr <= 0 and lr >= -3:
                                lr = -7
                            if fb <= 3 and fb >= 0:
                                lr = 7
                            if fb <= 0 and fb >= -3:
                                fb = -7  

                            if abs(dx) < 45 and abs(dy) < 30:
                                print(">>> GREEN reached -> FOLLOW_LINE")
                                phase = "FOLLOW_LINE"

                    t.send_rc_control(lr, fb, ud, yaw)

                # ---------- PHASE 3: FOLLOW BLUE LINE (TOP ROIs) ----------
                elif phase == "FOLLOW_LINE":
                    f, L, C, R = detect_line_3roi(f)

                    cv2.putText(f, "PHASE: FOLLOW_LINE", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if not (L or C or R):
                        fb = 0
                        yaw = 0
                    else:
                        if C and not L and not R:
                            fb = 10
                        elif C and L:
                            fb = 15
                            yaw = -45
                            lr = -20
                            last = "L" 
                        elif C and R:
                            fb = 15
                            yaw = +45
                            lr = 20  
                            last = "R"
                        elif L and not C:
                            fb = 0
                            lr = -10
                            last = "L"
                        elif R and not C:
                            fb = 0
                            lr = 10
                            last = "R"
                        elif not L and not R and not C:
                            if last == "L":
                                lr = 13
                                yaw = 10
                            if last == "R":
                                lr = 13
                                yaw = -10

                    t.send_rc_control(lr, fb, ud, yaw)

            # ====================== MANUAL MODE ======================
            elif airborne:
                if keyboard.is_pressed("w"): fb = max_fb
                if keyboard.is_pressed("s"): fb = -max_fb
                if keyboard.is_pressed("a"): lr = -max_lr
                if keyboard.is_pressed("d"): lr = max_lr
                if keyboard.is_pressed("r"): ud = max_lr
                if keyboard.is_pressed("f"): ud = -max_lr
                if keyboard.is_pressed("q"): yaw = -max_yaw
                if keyboard.is_pressed("e"): yaw = max_yaw

                t.send_rc_control(lr, fb, ud, yaw)

            else:
                t.send_rc_control(0, 0, 0, 0)

            cv2.imshow("Tello – ALIGN->GREEN->LINE (ROIs at TOP)", f)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.012)

    finally:
        try:
            t.send_rc_control(0, 0, 0, 0)
        except:
            pass

        if airborne:
            try:
                t.land()
            except:
                pass

        try:
            t.streamoff()
        except:
            pass

        try:
            cv2.destroyAllWindows()
        except:
            pass

        try:
            t.end()
        except:
            pass


if __name__ == "__main__":
    main()