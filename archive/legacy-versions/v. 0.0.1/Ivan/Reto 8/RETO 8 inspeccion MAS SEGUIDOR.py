import time
import math
import io
import contextlib

import cv2
import numpy as np
import keyboard
import threading
from djitellopy import Tello

from ultralytics import YOLO

from pywifi import PyWiFi, const, Profile

# ============================================================
#  YOLO MODEL (3 CLASES: 0=FIRE, 1=ANOMALY, 2=SHOE)
# ============================================================
model = YOLO(r"D:\carlo\Documents\Python\Concentration\Reto\v. 0.0.1\Ivan\Reto 8\best.pt")   # nuevo best.pt con fire, anomaly, shoe


# ============================================================
#  VIDEO THREAD COMPARTIDO PARA DRON 1
# ============================================================
frame_global = None

def video_thread():
    global frame_global
    cap = cv2.VideoCapture("udp://0.0.0.0:11111", cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[ERROR] Unable to open stream")
        return

    while True:
        ok, frame = cap.read()
        if ok:
            frame_global = cv2.resize(frame, (640, 480))
        else:
            time.sleep(0.01)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
#  DETECCIÓN DE TUBERÍA (ANGLE PARA ALIGN)
# ============================================================
def detect_white_angle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angle = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
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

        cv2.putText(frame, f"White angle: {angle:+.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, angle


# ============================================================
#  DETECCIÓN DE LÍNEA BLANCA (PID)
# ============================================================
def detect_white_line_angle(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
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
    if cv2.contourArea(cnt) < 1500:
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
    cx = int(cx)
    cy = int(cy)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame, angle_deg, cx, cy, True


# ============================================================
#  YOLO FIRE + ANOMALY + SHOE
# ============================================================
def detect_yolo(frame, conf_thresh=0.55):
    """
    Clases esperadas en best.pt:
      0 = FIRE
      1 = ANOMALY
      2 = SHOE

    Returns:
        fire_detected, anomaly_detected, shoe_detected,
        fire_cx, fire_cy, anomaly_cx, anomaly_cy, shoe_cx, shoe_cy, frame
    """
    fire_detected = False
    anomaly_detected = False
    shoe_detected = False

    fx = fy = ax = ay = sx = sy = None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = model(frame, imgsz=320, verbose=False)

    if not results:
        return False, False, False, None, None, None, None, None, None, frame

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return False, False, False, None, None, None, None, None, None, frame

    for b in r.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if cls == 0:  # FIRE
            fire_detected = True
            fx, fy = cx, cy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        elif cls == 1:  # ANOMALY
            anomaly_detected = True
            ax, ay = cx, cy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "ANOMALY", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        elif cls == 2:  # SHOE
            shoe_detected = True
            sx, sy = cx, cy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, "SHOE", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

    return fire_detected, anomaly_detected, shoe_detected, fx, fy, ax, ay, sx, sy, frame


# ============================================================
#  MISIÓN COMPLETA DRON 1 (TELLO-9A57E0)
# ============================================================
def drone1_mission():
    """
    Ejecuta:
    - ALIGN a tubería
    - FOLLOW_LINE con PID + MEMORY
    - INSPECT en 8 puntos usando anomaly + fire
    - PAUSA por SHOE (se detiene, no consume cooldown)
    - Genera traj_segments.txt
    """
    global frame_global

    # Parámetros globales
    COOLDOWN_INSPECTION = 4.0
    INSPECTION_DURATION = 4.0
    MAX_INSPECTIONS = 8

    t = Tello()
    t.connect()
    print("[DRON 1] Battery:", t.get_battery())
    t.streamon()
    threading.Thread(target=video_thread, daemon=True).start()

    airborne = False
    auto = False
    phase = "ALIGN"

    max_yaw = 120
    max_fb = 50
    max_lr = 50

    # --- inspections ---
    inspection_index = 0
    inspection_results = ["-"] * MAX_INSPECTIONS
    cooldown_inspection = -999.0
    fire_seen = False
    inspect_start = 0.0
    current_point = None

    # --- PID state ---
    last_angle = 0
    error_angle = 0
    angle_deg = 0
    angle_raw = 0

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

    # Ganancias PID (como en tu versión buena)
    Kpcx = 0.13  ###                            0.13
    Kpcy = -0.065 ###                   0.065
    Kpan = 0.82##                   1.82    0.82

    Kicx = 0.15         ###         0.15
    Kicy = -0.023 ##                    0.043
    Kian = 0.4 ##                   0.1   0.4

    Kdcx = 0.15         ##  0.1
    Kdcy = -0.01
    Kda = 0.15 ###                    0.45  

    last_time = time.perf_counter()

    # --- memory mode ---
    last_lr = 0
    last_fb = 0
    last_yaw = 0
    lost_line_time = 0
    in_memory_mode = False
    MAX_MEMORY_TIME = 8.0

    # --- LOG DE SEGMENTOS PARA SEGUNDO DRON ---
    segments = []           # lista de (seg_id, fb_avg, duration)
    segment_active = False  # True cuando estamos midiendo un tramo entre inspecciones
    segment_start_time = 0.0
    segment_fb_sum = 0.0
    segment_fb_count = 0

    # --- PAUSA POR SHOE ---
    in_shoe_pause = False
    shoe_pause_start = 0.0

    try:
        print("\n[DRON 1] Presiona SPACE para despegar y usar modo AUTO con 't'")
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

            # YOLO (fire + anomaly + shoe)
            fire_now, anomaly_now, shoe_now, fx, fy, ax, ay, sx, sy, f = detect_yolo(
                f, conf_thresh=0.45
            )

            # CONTROLES BÁSICOS
            if keyboard.is_pressed("space") and not airborne:
                t.takeoff()
                airborne = True
                auto = False
                phase = "ALIGN"
                time.sleep(0.3)

            if keyboard.is_pressed("l") and airborne:
                t.land()
                airborne = False
                break

            if keyboard.is_pressed("t"):
                auto = not auto
                print("[DRON 1] AUTO =", auto)
                phase = "ALIGN"
                time.sleep(0.3)

            if keyboard.is_pressed("esc") or keyboard.is_pressed("q"):
                break

            lr = fb = ud = yaw = 0

            # ========================= AUTO =========================
            if airborne and auto:

                # ---------- MANEJO DE SHOE (PAUSA GLOBAL) ----------
                # Si detectamos shoe y NO estamos en inspección:
                if shoe_now and phase != "INSPECT":
                    if not in_shoe_pause:
                        in_shoe_pause = True
                        shoe_pause_start = time.time()
                        print("[DRON 1] SHOE DETECTED → PAUSE")

                    t.send_rc_control(0, 0, 0, 0)
                    cv2.putText(f, "SHOE DETECTED - PAUSE",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 0, 255), 2)

                    cv2.imshow("Tello – PID + Anomaly + Memory + Shoe", f)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    time.sleep(0.012)
                    continue  # saltamos toda la lógica de fase

                # Si ya NO hay shoe y veníamos de una pausa:
                if in_shoe_pause and (not shoe_now):
                    pause_dt = time.time() - shoe_pause_start
                    cooldown_inspection += pause_dt  # para que el cooldown no se "consuma"
                    in_shoe_pause = False
                    print(f"[DRON 1] SHOE desapareció, reanudando (pause={pause_dt:.1f}s)")

                # ===== ALIGN =====
                if phase == "ALIGN":
                    f, angle = detect_white_angle(f)

                    cv2.putText(f, "PHASE: ALIGN", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if angle is None:
                        fb = 10
                    else:
                        yaw = int(clamp(angle * 0.8, -25, 25))
                        # Cuando ya está alineado, pasamos DIRECTO a FOLLOW_LINE
                        if abs(angle) < 3.0:
                            phase = "FOLLOW_LINE"
                            cooldown_inspection = time.time()  # arrancar cooldown desde aquí

                    t.send_rc_control(0, 0, 0, yaw)

                # ===== FOLLOW_LINE (PID + MEMORY + ANOMALY ROI) =====
                elif phase == "FOLLOW_LINE":

                    cv2.putText(f, "PHASE: FOLLOW_LINE (PID)",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # ROI de anomaly (igual que antes usabas para verde)
                    big_x1 = cx_img - 200
                    big_x2 = cx_img + 200
                    big_y1 = cy_img - 80
                    big_y2 = cy_img + 190
                    cv2.rectangle(f, (big_x1, big_y1),
                                  (big_x2, big_y2), (0, 255, 0), 1)

                    anomaly_in_roi = False
                    if anomaly_now and ax is not None and ay is not None:
                        if big_x1 <= ax <= big_x2 and big_y1 <= ay <= big_y2:
                            anomaly_in_roi = True
                            cv2.circle(f, (ax, ay), 6, (0, 0, 255), 2)

                    # PID timing
                    now = time.perf_counter()
                    dt = now - last_time
                    last_time = now
                    if dt <= 0:
                        dt = 1e-3

                    error_angle = angle_raw - last_angle
                    if abs(error_angle) > 90:
                        angle_raw = last_angle
                        angle_deg = angle_raw
                    else:
                        last_angle = angle_raw
                        angle_deg = angle_raw

                    # detect pipe
                    f, angle_raw, cx, cy, detected_line = detect_white_line_angle(f)

                    # ====================== PIPE DETECTED ======================
                    if detected_line:
                        in_memory_mode = False

                        ebx = cxe
                        eby = cye
                        eba = angle_error

                        cxe = cx - cx_img
                        cye = cy - cy_img
                        angle_error = angle_raw

                        # LATERAL
                        if abs(cxe) > 1:
                            integralcx = eintbx + cxe * dt
                            eintbx = cxe * dt
                            derivativocx = (cxe - ebx) / dt
                            integralcx = min(integralcx, 25)
                            incx = Kpcx * cxe + Kicx * integralcx + Kdcx * derivativocx
                            incx = int(clamp(incx, -max_lr, max_lr))
                        else:
                            incx = 0

                        # FORWARD / BACKWARDS
                        if abs(cye) > 3:
                            integralcy = eintby + cye * dt
                            eintby = cye * dt
                            derivativocy = (cye - eby) / dt
                            integralcy = min(integralcy, 25)
                            incy = Kpcy * cye + Kicy * integralcy + Kdcy * derivativocy
                            incy = int(clamp(incy, -max_fb, max_fb))
                        else:
                            incy = 0

                        # ANGLE
                        if abs(angle_error) > 1:
                            integrala = eintba + angle_error * dt
                            eintba = angle_error * dt
                            derivativoa = (angle_error - eba) / dt
                            integrala = min(integrala, 50)
                            inan = (Kpan * angle_error +
                                    Kian * integrala +
                                    Kda * derivativoa)
                            inan = int(clamp(inan, -max_yaw, max_yaw))
                        else:
                            inan = 0

                        lr = int(incx)
                        fb = int(incy)
                        yaw = int(inan)

                        t.send_rc_control(lr, fb+10, ud, yaw)

                        # Guardar último comando para memoria
                        last_lr = lr
                        last_fb = fb
                        last_yaw = yaw

                        # Si hay segmento activo, acumular fb para promedio
                        if segment_active:
                            segment_fb_sum += fb
                            segment_fb_count += 1

                        cv2.putText(f,
                                    f"y_er:{int(cye)} x_er:{int(cxe)} a_er:{int(angle_error)}",
                                    (w - 330, h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (50, 220, 255), 2)

                    # ====================== PIPE LOST → MEMORY ======================
                    else:
                        cv2.putText(f, "NO WHITE LINE (MEMORY MODE)",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if not in_memory_mode:
                            in_memory_mode = True
                            lost_line_time = time.time()

                        elapsed_mem = time.time() - lost_line_time

                        # Si ya hizo 8 inspecciones y se perdió la línea → LAND
                        if inspection_index >= MAX_INSPECTIONS and elapsed_mem >= MAX_MEMORY_TIME:
                            print(">>> 8 INSPECTIONS COMPLETE — LOST PIPE — LANDING")
                            t.send_rc_control(0, 0, 0, 0)
                            t.land()
                            break

                        if elapsed_mem <= MAX_MEMORY_TIME:
                            lr = int(last_lr * 0.5) - 11
                            fb = int(last_fb * 0.5) + 7
                            yaw = int(last_yaw * 0.5) - 14
                            t.send_rc_control(lr, fb, ud, yaw)

                            cv2.putText(f, f"MEMORY MOVE ({elapsed_mem:.1f}s)",
                                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            t.send_rc_control(0, 0, 0, 0)
                            cv2.putText(f, "MEMORY EXPIRED → STOP",
                                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # ====================== INSPECTION TRIGGER ======================
                    

                    if (anomaly_in_roi
                        and inspection_index < MAX_INSPECTIONS
                        and (time.time() - cooldown_inspection) >= COOLDOWN_INSPECTION):

                        # TERMINAR SEGMENTO ACTUAL (si existía)
                        if segment_active and segment_fb_count > 0:
                            seg_duration = time.time() - segment_start_time
                            fb_avg = segment_fb_sum / segment_fb_count
                            seg_id = len(segments) + 1
                            segments.append((seg_id, fb_avg, seg_duration))
                            print(f"[SEGMENT {seg_id}] FB_AVG={fb_avg:.3f}, TIME={seg_duration:.3f}s")
                        segment_active = False

                        print(f">> INSPECTING anomaly at point {inspection_index + 1}")
                        phase = "INSPECT"
                        inspect_start = time.time()
                        fire_seen = False
                        current_point = inspection_index
                        t.send_rc_control(0, 0, 0, 0)
                        continue

                # ===== INSPECT =====
                elif phase == "INSPECT":

                    cv2.putText(f, f"PHASE: INSPECT (Point {current_point + 1})",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # En INSPECT ignoramos shoe (como pediste)
                    if fire_now:
                        fire_seen = True

                    t.send_rc_control(0, 0, 0, 0)

                    elapsed = time.time() - inspect_start
                    cv2.putText(f, f"t_inspect={elapsed:.1f}s",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if elapsed >= INSPECTION_DURATION:
                        # Clasificación del punto
                        if fire_seen:
                            inspection_results[current_point] = "MALO"
                            print(f">> Point {current_point + 1}: MALO (FIRE)")
                        else:
                            inspection_results[current_point] = "BUENO"
                            print(f">> Point {current_point + 1}: BUENO (NO FIRE)")

                        inspection_index += 1
                        cooldown_inspection = time.time()

                        # A partir de AQUÍ empieza el siguiente segmento:
                        if inspection_index < MAX_INSPECTIONS:
                            segment_active = True
                            segment_start_time = time.time()
                            segment_fb_sum = 0.0
                            segment_fb_count = 0

                        phase = "FOLLOW_LINE"
                        continue

            # ====================== MANUAL ======================
            elif airborne:
                if keyboard.is_pressed("w"): fb = max_fb
                if keyboard.is_pressed("s"): fb = -max_fb
                if keyboard.is_pressed("a"): lr = -max_lr
                if keyboard.is_pressed("d"): lr = max_lr
                if keyboard.is_pressed("q"): yaw = -max_yaw
                if keyboard.is_pressed("e"): yaw = max_yaw
                if keyboard.is_pressed("r"): ud = max_lr
                if keyboard.is_pressed("f"): ud = -max_lr

                t.send_rc_control(lr, fb, ud, yaw)

            cv2.imshow("Tello – PID + Anomaly + Memory + Shoe", f)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.012)

    finally:
        try:
            t.send_rc_control(0, 0, 0, 0)
        except:
            pass
        try:
            if airborne:
                t.land()
        except:
            pass
        cv2.destroyAllWindows()
        try:
            t.end()
        except:
            pass

        print("\n===== INSPECTION SUMMARY (DRON 1) =====")
        for i, res in enumerate(inspection_results, start=1):
            print(f"Point {i}: {res}")

        # ====================== GUARDAR ARCHIVO TXT ======================
        try:
            with open("traj_segments.txt", "w") as ftxt:
                ftxt.write("SEGMENTS (between inspections, starting AFTER first inspect):\n")
                for seg_id, fb_avg, duration in segments:
                    ftxt.write(f"SEGMENT {seg_id}\n")
                    ftxt.write(f"FB_AVG={fb_avg:.3f}\n")
                    ftxt.write(f"TIME={duration:.3f}\n")
                    # TURNs para segundo dron: después del segmento 2 y 4
                    if seg_id in (2, 4):
                        ftxt.write("TURN_90_LEFT=1\n")
                    ftxt.write("\n")

                ftxt.write("POINT_RESULTS:\n")
                for i, res in enumerate(inspection_results, start=1):
                    ftxt.write(f"POINT {i}={res}\n")

            print("\n>>> Archivo 'traj_segments.txt' generado correctamente.")
        except Exception as e:
            print("Error writing traj_segments.txt:", e)


# ============================================================
#  LECTURA DE PUNTOS BUENOS/MALOS PARA DRON 2
# ============================================================
def read_fire_points(filename):
    fire = []
    inside = False

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if ("POINT_RESULTS" in line) and not inside:
                inside = True
                continue

            if inside:
                if line.startswith("POINT") and "=" in line:
                    left, result = line.split("=", 1)
                    parts = left.strip().split()
                    if len(parts) < 2:
                        continue
                    try:
                        idx = int(parts[1])
                    except:
                        continue

                    fire.append(1 if result.upper() == "MALO" else 0)

    print("\n=== FIRE LISTA GENERADA (DRON 2) ===")
    print(fire)
    print("====================================\n")
    return fire


# ============================================================
#  MISIÓN GEOMÉTRICA DEL DRON 2 (TELLO-FE193A)
# ============================================================
def second_drone_mission(fire):
    t = Tello()
    t.connect()
    print("Drone 2 conectado!")
    t.takeoff()
    time.sleep(2)

    # Distancias corregidas que tú ajustaste
    c = 360
    d = 410 ##              430

    coords = [
        (0, 0),
        (0, c/2),
        (0, c),
        ((-d-5)/2, c),
        (-(3*d-10)/4, c),
        (-d -10, c),
        (-d, (c/2)-10),
        (-d, 0)
    ]

    fp = [0]
    for i, val in enumerate(fire):
        if val == 1:
            fp.append(i)
    fp.append(0)

    print(f"Puntos a visitar: {fp}")

    last_yaw = 0

    for i in range(len(fp) - 1):
        p1 = fp[i]
        p2 = fp[i + 1]

        x1, y1 = coords[p1]
        x2, y2 = coords[p2]

        dx = x2 - x1
        dy = y2 - y1

        print(f"\nVECTOR: ({dx}, {dy})")

        angle_world = math.degrees(math.atan2(dx, dy))
        angle_world = -angle_world

        rotation = angle_world - last_yaw
        rotation = (rotation + 180) % 360 - 180

        dist = int(math.sqrt(dx * dx + dy * dy))

        print(f"Rotate = {int(rotation)} deg")
        print(f"Move   = {dist} cm")

        last_yaw += rotation
        last_yaw = (last_yaw + 180) % 360 - 180
        print(f"New yaw = {int(last_yaw)}")

        try:
            if abs(rotation) > 1:
                if rotation > 0:
                    t.rotate_counter_clockwise(int(abs(rotation)))
                else:
                    t.rotate_clockwise(int(abs(rotation)))
                time.sleep(1.5)

            if dist > 0:
                t.move_forward(dist)
                time.sleep(2)

        except Exception as e:
            print(f"Error enviando comandos al dron 2: {e}")

    t.land()
    print("Drone 2 finalizó su misión.")


# ============================================================
#  WIFI HANDLING (TP-LINK + INTERFACE 1)
# ============================================================
def connect_wifi(ssid, interface_index=1):
    wifi = PyWiFi()
    interfaces = wifi.interfaces()

    if interface_index >= len(interfaces):
        print(f"Error: invalid interface index {interface_index}.")
        return None

    iface = interfaces[interface_index]
    print(f"Connecting to {ssid} via {iface.name}")

    profile = Profile()
    profile.ssid = ssid
    profile.auth = const.AUTH_ALG_OPEN
    profile.akm.append(const.AKM_TYPE_NONE)
    profile.cipher = const.CIPHER_TYPE_NONE

    iface.remove_all_network_profiles()
    iface.add_network_profile(profile)
    iface.connect(profile)

    time.sleep(5)

    if iface.status() == const.IFACE_CONNECTED:
        print(f"Connected to {ssid}!")
        return iface
    else:
        print(f"Failed to connect to {ssid}")
        return None


def disconnect_wifi(interface_index=1):
    wifi = PyWiFi()
    interfaces = wifi.interfaces()

    if interface_index >= len(interfaces):
        print(f"Error: invalid interface index {interface_index}.")
        return

    iface = interfaces[interface_index]
    iface.remove_all_network_profiles()
    print(f"Disconnected from WiFi on {iface.name}")


# ============================================================
#  MAIN: INTERCONEXIÓN DOS DRONES
# ============================================================
def main():

    # =============================
    #   ✈️ DRON 1 (TELLO-9A57E0)
    # =============================
    if connect_wifi("TELLO-9A57E0", interface_index=1):
        print("\n[MAIN] Ejecutando misión DRON 1 (inspecciones + TXT)...")
        drone1_mission()
        disconnect_wifi(interface_index=1)

    # =============================
    #   ✈️ DRON 2 (TELLO-FE193A)
    # =============================
    if connect_wifi("TELLO-FE193A", interface_index=1):
        print("\n[MAIN] Ejecutando misión DRON 2 (trayectoria con MALO)...")
        fire = read_fire_points("traj_segments.txt")
        second_drone_mission(fire)
        disconnect_wifi(interface_index=1)


if __name__ == "__main__":
    main()
