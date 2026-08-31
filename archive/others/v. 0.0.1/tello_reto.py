"""
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
import math
from djitellopy import Tello
import threading
import time

# =======================
# Gains, thresholds & limits
# =======================

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

# =======================
# BLUE line detector (centroid + angle)
# =======================

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
    x1 = w // 2 - 170
    y1 = h // 2 - 180
    x2 = w // 2 + 170
    y2 = h // 2 - 60

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
        detection = False
        return frame, angle_deg, cx, cy, detection

    cnt = max(cnts, key=cv2.contourArea)

    # Ignore very small contours
    MIN_AREA = 1500
    if cv2.contourArea(cnt) < MIN_AREA:
        cv2.putText(frame, "Line too small", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        angle_deg = cx = cy = 0
        detection = False
        return frame, angle_deg, cx, cy, detection

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

    detection = True
    
    return frame, angle_deg, cx, cy, detection

# =======================
# Main
# =======================
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
    detection = False

    print("\nControls:")
    print(" SPACE = takeoff")
    print(" L     = land")
    print(" T     = toggle AUTO")
    print(" WASD/RF/QE = manual")
    print(" ESC/Q = quit\n")

    speed_manual = 40

    # Inicial Parameters
    last_angle = 0 
    error_angle = 0
    angle_deg = 0
    angle_raw = 0

    # Inicial Conditions
    integralcx = 0
    integralcy = 0
    integrala = 0
    derivativocx = 0
    derivativocy = 0
    derivativoa = 0
    eintbx = 0
    eintby = 0
    eintba = 0
    incx = 0
    incy = 0
    inan = 0
    ebx = 0
    eby = 0
    eba = 0
    cxe = 0
    cye = 0
    angle_error = 0

    # Gains Proporcional
    Kpcx = 0.18
    Kpcy = -0.138   
    Kpan = 1.815

    # Gains Integral
    Kicx = 0.46 
    Kicy = -0.1
    Kian = 0.1
    
    # Derivative gains
    Kdcx = 0.1
    Kdcy = -0.01
    Kda = 0.45

    # Max Values
    max_yaw = 120
    max_fb = 50  
    max_lr = 50  

    last_time = time.perf_counter()

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

            if keyboard.is_pressed("esc"):
                break

            lr = fb = ud = yaw = 0

            # Error Calculations

            error_angle = angle_raw - last_angle

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if error_angle > 90 or error_angle < -90:
                angle_raw = last_angle
                angle_deg = angle_raw
            else:
                last_angle = angle_raw  # Update last valid angle
                angle_deg = angle_raw

            frame, angle_raw, cx, cy, detection = detect_blue_line_angle(frame)
            
            # ====================== AUTO MODE ======================
            if airborne and auto and detection:
                # Actual error
                ebx = cxe
                eby = cye
                eba = angle_error
                cxe = cx - cx_img
                cye = cy - cy_img
                angle_error = angle_raw 

                # Input with P + I and if error is less than 3 and 1
                if abs(cxe) > 1:
                    integralcx = eintbx + cxe*dt
                    eintbx = cxe*dt
                    derivativocx = (cxe - ebx)/dt
                    if integralcx > 35: integralcx = 35
                    incx = Kpcx*cxe + Kicx*integralcx + Kdcx*derivativocx
                    incx = int(clamp(incx, -max_lr, max_lr))
                
                if abs(cye) > 3:
                    integralcy = eintby + cye*dt
                    eintby = cye*dt
                    derivativocy = (cye - eby)/dt
                    if integralcy > 35: integralcy = 35
                    incy = Kpcy*cye + Kicy*integralcy + Kdcy*derivativocy
                    incy = int(clamp(incy, -max_fb, max_fb))

                if abs(angle_error) > 1:
                    integrala = eintba + angle_error*dt
                    eintba = angle_error*dt
                    derivativoa = (angle_error - eba)/dt
                    if integrala > 50: integrala = 50
                    inan = Kpan*angle_error + Kian*integrala + Kda*derivativoa
                    inan = int(clamp(inan, -max_yaw, max_yaw))

                lr = int(incx)
                fb = int(incy)
                yaw = int(inan)

                t.send_rc_control(lr, fb, ud, yaw)

                # HUD of applied sticks
                cv2.putText(frame, f"y_er:{int(cye):+d} x_er:{int(cxe):+d} a_er:{int(angle_error):+d}",
                            (w - 330, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 255), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
                # Show angle
                cv2.putText(frame, f"angle = {angle_deg:.1f}, cx {cx}, cy {cy}",
                        (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
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

            time.sleep(0.012)  # ~50 Hz loop

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
        try: cv2.destroyAllWindows()
        except: pass
        try: t.end()
        except: pass

if __name__ == "__main__":
    main() 
