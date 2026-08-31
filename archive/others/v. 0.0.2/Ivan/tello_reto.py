"""
Tello AUTO:
1) ALIGN → Alinear con la línea azul
2) FOLLOW_LINE (PID)
3) Si se pierde la línea:
      - Siempre se lee AprilTag
      - Si pasan >= 1 s sin línea y el CENTROIDE del tag está en el ROI MORADO central (66%),
        se pasa a ALIGN_TAG
4) ALIGN_TAG
5) FORWARD_SEARCH / LEFT_SEARCH / RIGHT_SEARCH según el ID del tag
   (y durante estas búsquedas también puede leer nuevos tags con la misma condición)
"""

import cv2
import time
import numpy as np
import keyboard
from djitellopy import Tello
import threading
from pupil_apriltags import Detectorx 
import math


# =====================================================================
# APRILTAG DETECTOR
# =====================================================================

tag_detector = Detector(families="tag36h11")


def detect_apriltag(frame):
    """
    Devuelve:
    found(bool), tag_id(int or None), cx(int or None), cy(int or None), angle(float or None)
    Siempre dibuja contorno y centroide si hay tag.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = tag_detector.detect(gray)

    if len(tags) == 0:
        return False, None, None, None, None

    tag = tags[0]
    cx, cy = tag.center
    tag_id = tag.tag_id

    (ptA, ptB, ptC, ptD) = tag.corners
    dx = ptB[0] - ptA[0]
    dy = ptB[1] - ptA[1]
    angle = np.degrees(np.arctan2(dy, dx))

    cv2.polylines(frame, [np.int32(tag.corners)], True, (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
    cv2.putText(
        frame,
        f"TAG {tag_id}",
        (int(cx) - 20, int(cy) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return True, tag_id, int(cx), int(cy), angle


# =====================================================================
# VIDEO THREAD
# =====================================================================

frame_global = None


def video_thread():
    global frame_global
    cap = cv2.VideoCapture("udp://0.0.0.0:11111", cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[ERROR] Stream failed")
        return

    while True:
        ok, frame = cap.read()
        if ok:
            frame_global = cv2.resize(frame, (640, 480))
        else:
            time.sleep(0.01)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =====================================================================
# BLUE ANGLE DETECTOR (para ALIGN)
# =====================================================================

def detect_blue_angle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 60, 60])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    angle = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return frame, None

        rect = cv2.minAreaRect(c)
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

        angle_raw = rect[2]
        w, h = rect[1]

        angle_calc = angle_raw
        if w < h:
            angle_calc += 90
        angle_calc -= 90

        if angle_calc > 75 or angle_calc < -75:
            angle_calc = 0

        angle = angle_calc

        cv2.putText(
            frame,
            f"ALIGN angle: {angle:+.1f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    return frame, angle


# =====================================================================
# BLUE LINE DETECTOR (PID) → devuelve cx, cy, angle
# =====================================================================

def detect_blue_line_angle(frame):
    """
    Busca la línea azul SOLO dentro de un ROI amarillo grande
    que cubre la parte frontal.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # ROI AMARILLO (para la línea)
    x1 = w // 2 - 180
    y1 = h // 2 - 280
    x2 = w // 2 + 180
    y2 = h // 2 - 40

    # ROI visual amarillo
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    roi_mask = np.zeros_like(blue_mask, dtype=np.uint8)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
    mask_roi = cv2.bitwise_and(blue_mask, roi_mask)

    cnts, _ = cv2.findContours(
        mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not cnts:
        return frame, 0, 0, 0

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1500:
        return frame, 0, 0, 0

    [vx, vy, _, _] = cv2.fitLine(
        cnt, cv2.DIST_L2, 0, 0.01, 0.01
    )

    if vy > 0:
        vx, vy = -vx, -vy

    angle_rad = math.atan2(vx, -vy)
    angle_raw = math.degrees(angle_rad)

    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]

    cx = int(cx)
    cy = int(cy)

    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

    cv2.putText(
        frame,
        f"PID angle: {angle_raw:+.1f}",
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return frame, angle_raw, cx, cy


# =====================================================================
# SEARCH 3-ROIs (para reenganchar línea)
# =====================================================================

def detect_line_3roi(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([90, 60, 60]), np.array([130, 255, 255])
    )

    roi_x = 70
    roi_y = 100

    cx1 = w // 2 - roi_x -40  ## 0
    cx2 = w // 2 + roi_x + 80##  40
    y1 = h // 2 - roi_y - 55
    y2 = h // 2 - 70

    lx1 = cx1
    lx2 = cx1 - roi_x
    rx1 = cx2
    rx2 = cx2 + roi_x

    def check(mask_local, x1_local, y1_local, x2_local, y2_local):
        roi_mask = np.zeros_like(mask_local)
        cv2.rectangle(
            roi_mask, (x1_local, y1_local), (x2_local, y2_local), 255, -1
        )
        masked = cv2.bitwise_and(mask_local, roi_mask)
        cnts_local, _ = cv2.findContours(
            masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts_local:
            return False
        c_local = max(cnts_local, key=cv2.contourArea)
        return cv2.contourArea(c_local) > 70

    L = check(mask, lx1, y1, lx2, y2)
    C = check(mask, cx1, h // 2 - 70, cx2, h // 2 - roi_y)
    R = check(mask, rx1, y1, rx2, y2)

    cv2.putText(
        frame,
        f"L:{int(L)} C:{int(C)} R:{int(R)}",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    return frame, L, C, R


# =====================================================================
# MAIN
# =====================================================================

def main():
    global frame_global

    t = Tello()
    t.connect()
    print("Battery:", t.get_battery())

    t.streamon()
    threading.Thread(target=video_thread, daemon=True).start()

    airborne = False
    auto = False
    phase = "ALIGN"
    stored_tag = None

    max_yaw = 120
    max_fb = 50
    max_lr = 50

    # Tiempo desde la última vez que se vio la línea
    last_line_seen_time = None
    NO_LINE_TAG_DELAY = 6.0  # segundos sin línea para permitir acción de AprilTag

    # PID variables
    last_angle = 0
    angle_error = 0
    cxe = 0
    cye = 0

    integralcx = 0
    integralcy = 0
    integrala = 0
    eintbx = 0
    eintby = 0
    eintba = 0

    Kpcx = 0.18
    Kpcy = -0.116
    Kpan = 1.815

    Kicx = 0.46
    Kicy = -0.1
    Kian = 0.1

    Kdcx = 0.1
    Kdcy = -0.01
    Kda = 0.45

    lr_last = 0
    fb_last = 0
    yaw_last = 0

    last_time = time.perf_counter()

    try:
        while True:
            frame = frame_global
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 0)
            f = frame.copy()

            h, w = f.shape[:2]
            cx_img = w // 2
            cy_img = h // 2
            cv2.circle(f, (cx_img, cy_img), 4, (0, 255, 0), -1)

            # ===================== ROI MORADO (AprilTag) =====================
            tag_roi_w = int(w * 0.66)
            tag_roi_h = int(h * 0.66)
            tag_x1 = cx_img - tag_roi_w // 2 - 80
            tag_y1 = cy_img - tag_roi_h // 2 - 80
            tag_x2 = cx_img + tag_roi_w // 2 + 80
            tag_y2 = cy_img + tag_roi_h // 2 + 80

            cv2.rectangle(f, (tag_x1, tag_y1), (tag_x2, tag_y2), (255, 0, 255), 2)

            # Tiempo dt
            now = time.perf_counter()
            dt = now - last_time
            if dt <= 0:
                dt = 1e-3
            last_time = now

            # Teclas
            if keyboard.is_pressed("space") and not airborne:
                t.takeoff()
                airborne = True
                auto = False
                phase = "ALIGN"
                last_line_seen_time = None
                time.sleep(0.3)

            if keyboard.is_pressed("l") and airborne:
                t.land()
                return

            if keyboard.is_pressed("t"):
                auto = not auto
                print("AUTO:", auto)
                phase = "ALIGN"
                last_line_seen_time = None
                time.sleep(0.3)

            if keyboard.is_pressed("esc"):
                break

            lr = fb = ud = yaw = 0

            # ================= SIEMPRE LEER APRILTAG =================
            tag_found, tag_id, tx, ty, tang = detect_apriltag(f)

            tag_in_roi = False
            if tag_found and tx is not None and ty is not None:
                if tag_x1 <= tx <= tag_x2 and tag_y1 <= ty <= tag_y2:
                    tag_in_roi = True
                    cv2.circle(f, (tx, ty), 8, (255, 0, 255), 2)

            # =====================================================================
            # ========================== AUTO MODE ================================
            # =====================================================================

            if airborne and auto:
                # -----------------------------------------------------
                # ALIGN
                # -----------------------------------------------------
                if phase == "ALIGN":
                    f, angle = detect_blue_angle(f)
                    cv2.putText(
                        f,
                        "PHASE: ALIGN",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    if angle is None:
                        yaw = 15
                    else:
                        yaw = int(clamp(angle * 1.0, -max_yaw, max_yaw))
                        # Tolerancia de alineación inicial
                        if abs(angle) < 5:
                            print(">> ALIGN → FOLLOW_LINE")
                            phase = "FOLLOW_LINE"
                            last_line_seen_time = time.time()

                    t.send_rc_control(0, 0, 0, yaw)

                # -----------------------------------------------------
                # FOLLOW_LINE (PID + memoria + AprilTag gating)
                # -----------------------------------------------------
                elif phase == "FOLLOW_LINE":
                    cv2.putText(
                        f,
                        "PHASE: FOLLOW_LINE (PID)",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    f, angle_meas, cx, cy = detect_blue_line_angle(f)

                    line_found = not (cx == 0 and cy == 0 and angle_meas == 0)

                    if line_found:
                        last_line_seen_time = time.time()
                        no_line_elapsed = 0.0
                    else:
                        if last_line_seen_time is None:
                            no_line_elapsed = 999.0
                        else:
                            no_line_elapsed = time.time() - last_line_seen_time

                    cv2.putText(
                        f,
                        f"x_er:{int(cxe):+d}  y_er:{int(cye):+d}  a_er:{int(angle_error):+d}",
                        (w - 330, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (50, 220, 255),
                        2,
                    )

                    # ====================== CASO: NO HAY LÍNEA ======================
                    if not line_found:
                        if abs(cxe) < 40:
                            lr =0 
                            fb = 2
                            yaw = 0
                            cv2.putText(
                                f,
                                "MEMORIA: QUIETO",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 200, 255),
                                2,
                            )
                        else:
                            lr = int(lr_last * 0.20)
                            fb = int(fb_last * 0.15)+2
                            yaw = int(yaw_last * 0.20)
                            cv2.putText(
                                f,
                                f"MEMORIA PID lr={lr} fb={fb} yaw={yaw}",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 220, 255),
                                2,
                            )

                        # Gating para saltar a ALIGN_TAG
                        if (
                            tag_found
                            and tag_in_roi
                            and no_line_elapsed >= NO_LINE_TAG_DELAY
                        ):
                            print(
                                ">> NO LINE + TAG EN ROI + DELAY → ALIGN_TAG con tag",
                                tag_id,
                            )
                            stored_tag = tag_id
                            phase = "ALIGN_TAG"
                            t.send_rc_control(0, 0, 0, 0)
                        else:
                            t.send_rc_control(lr, fb, 0, yaw)

                        cv2.imshow("Tello", f)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                        continue

                    # ====================== CASO: HAY LÍNEA ======================
                    no_line_elapsed = 0.0

                    ebx = cxe
                    eby = cye
                    eba = angle_error

                    cxe = cx - cx_img
                    cye = cy - cy_img
                    angle_raw = angle_meas

                    if abs(angle_raw - last_angle) < 90:
                        last_angle = angle_raw

                    angle_error = angle_raw

                    # PID X
                    if abs(cxe) > 1:
                        integralcx = eintbx + cxe * dt
                        eintbx = cxe * dt
                        dvx = (cxe - ebx) / dt
                        if integralcx > 25:
                            integralcx = 25
                        lr = int(
                            clamp(
                                Kpcx * cxe + Kicx * integralcx + Kdcx * dvx,
                                -max_lr,
                                max_lr,
                            )
                        )
                    else:
                        lr = 0

                    # PID Y
                    if abs(cye) > 3:
                        integralcy = eintby + cye * dt
                        eintby = cye * dt
                        dvy = (cye - eby) / dt
                        if integralcy > 25:
                            integralcy = 25
                        fb = int(
                            clamp(
                                Kpcy * cye + Kicy * integralcy + Kdcy * dvy,
                                -max_fb,
                                max_fb,
                            )
                        )
                    else:
                        fb = 0

                    # PID yaw
                    if abs(angle_error) > 1:
                        integrala = eintba + angle_error * dt
                        eintba = angle_error * dt
                        dva = (angle_error - eba) / dt
                        if integrala > 50:
                            integrala = 50
                        yaw = int(
                            clamp(
                                Kpan * angle_error
                                + Kian * integrala
                                + Kda * dva,
                                -max_yaw,
                                max_yaw,
                            )
                        )
                    else:
                        yaw = 0

                    lr_last = lr
                    fb_last = fb
                    yaw_last = yaw

                    t.send_rc_control(lr, fb+3, 0, yaw)

                # -----------------------------------------------------
                # ALIGN_TAG
                # -----------------------------------------------------
                elif phase == "ALIGN_TAG":
                    cv2.putText(
                        f,
                        "PHASE: ALIGN_TAG",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )

                    tag_found2, tag_id2, tx2, ty2, tang2 = detect_apriltag(f)

                    if not tag_found2:
                        t.send_rc_control(0, 0, 0, 0)
                        cv2.putText(
                            f,
                            "No TAG → QUIETO",
                            (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        dx = tx2 - cx_img
                        dy = ty2 - cy_img
                        tang = tang2

                        lr = int(clamp(dx * 0.17, -max_lr, max_lr))
                        fb = int(clamp(-dy * 0.12, -max_fb, max_fb))
                        yaw = int(clamp(tang * 0.87, -max_yaw, max_yaw))
 
                        if abs(dx) < 25:
                            lr = 0
                        if abs(dy) < 35:
                            fb = 0
                        if abs(tang) < 5:
                            yaw = 0

                        t.send_rc_control(lr, fb, 0, yaw)

                        if abs(dx) < 35 and abs(dy) < 35 and abs(tang) < 5:
                            cv2.putText(
                                f,
                                "TAG ALIGNED!",
                                (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )

                            if stored_tag == 1:
                                phase = "FORWARD_SEARCH"
                            elif stored_tag == 2:
                                phase = "LEFT_SEARCH"
                            elif stored_tag == 3:
                                phase = "RIGHT_SEARCH"
                            elif stored_tag == 4:
                                t.send_rc_control(0, 0, 0, 0)
                                t.land()
                                return

                            # empezamos a contar tiempo sin línea desde ahora
                            last_line_seen_time = time.time()

                # -----------------------------------------------------
                # FORWARD_SEARCH
                # -----------------------------------------------------
                elif phase == "FORWARD_SEARCH":
                    cv2.putText(
                        f,
                        "PHASE: FORWARD_SEARCH",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    f, L, C, R = detect_line_3roi(f)
                    line_found = L or C or R

                    if line_found:
                        phase = "FOLLOW_LINE"
                        last_line_seen_time = time.time()
                        t.send_rc_control(0, 0, 0, 0)
                    else:
                        if last_line_seen_time is None:
                            no_line_elapsed = 999.0
                        else:
                            no_line_elapsed = time.time() - last_line_seen_time

                        if (
                            tag_found
                            and tag_in_roi
                            and no_line_elapsed >= NO_LINE_TAG_DELAY
                        ):
                            print(
                                ">> FORWARD_SEARCH: TAG EN ROI + DELAY → ALIGN_TAG con tag",
                                tag_id,
                            )
                            stored_tag = tag_id
                            phase = "ALIGN_TAG"
                            t.send_rc_control(0, 0, 0, 0)
                        else:
                            lr = 3
                            fb = 12
                            yaw = 8
                            t.send_rc_control(lr, fb, 0, yaw)

                # -----------------------------------------------------
                # LEFT_SEARCH
                # -----------------------------------------------------
                elif phase == "LEFT_SEARCH":
                    cv2.putText(
                        f,
                        "PHASE: LEFT_SEARCH",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    f, L, C, R = detect_line_3roi(f)
                    line_found = L or C or R

                    if line_found:
                        phase = "FOLLOW_LINE"
                        last_line_seen_time = time.time()
                        t.send_rc_control(0, 0, 0, 0)
                    else:
                        if last_line_seen_time is None:
                            no_line_elapsed = 999.0
                        else:
                            no_line_elapsed = time.time() - last_line_seen_time

                        if (
                            tag_found
                            and tag_in_roi
                            and no_line_elapsed >= NO_LINE_TAG_DELAY
                        ):
                            print(
                                ">> LEFT_SEARCH: TAG EN ROI + DELAY → ALIGN_TAG con tag",
                                tag_id,
                            )
                            stored_tag = tag_id
                            phase = "ALIGN_TAG"
                            t.send_rc_control(0, 0, 0, 0)
                        else:
                            lr = -14
                            fb = 6
                            yaw = -11
                            t.send_rc_control(lr, fb, 0, yaw)

                # -----------------------------------------------------
                # RIGHT_SEARCH
                # -----------------------------------------------------
                elif phase == "RIGHT_SEARCH":
                    cv2.putText(
                        f,
                        "PHASE: RIGHT_SEARCH",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    f, L, C, R = detect_line_3roi(f)
                    line_found = L or C or R

                    if line_found:
                        phase = "FOLLOW_LINE"
                        last_line_seen_time = time.time()
                        t.send_rc_control(0, 0, 0, 0)
                    else:
                        if last_line_seen_time is None:
                            no_line_elapsed = 999.0
                        else:
                            no_line_elapsed = time.time() - last_line_seen_time

                        if (
                            tag_found
                            and tag_in_roi
                            and no_line_elapsed >= NO_LINE_TAG_DELAY
                        ):
                            print(
                                ">> RIGHT_SEARCH: TAG EN ROI + DELAY → ALIGN_TAG con tag",
                                tag_id,
                            )
                            stored_tag = tag_id
                            phase = "ALIGN_TAG"
                            t.send_rc_control(0, 0, 0, 0)
                        else:
                            lr = 14
                            fb = 4
                            yaw = 8
                            t.send_rc_control(lr, fb, 0, yaw)

            # =====================================================================
            # =========================== MANUAL MODE =============================
            # =====================================================================
            elif airborne:

                if keyboard.is_pressed("w"):
                    fb = max_fb
                if keyboard.is_pressed("s"):
                    fb = -max_fb
                if keyboard.is_pressed("a"):
                    lr = -max_lr
                if keyboard.is_pressed("d"):
                    lr = max_lr
                if keyboard.is_pressed("q"):
                    yaw = -max_yaw
                if keyboard.is_pressed("e"):
                    yaw = max_yaw
                if keyboard.is_pressed("r"):
                    ud = max_lr
                if keyboard.is_pressed("f"):
                    ud = -max_lr

                t.send_rc_control(lr, fb, ud, yaw)

            cv2.imshow("Tello", f)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        try:
            t.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        try:
            t.land()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            t.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()