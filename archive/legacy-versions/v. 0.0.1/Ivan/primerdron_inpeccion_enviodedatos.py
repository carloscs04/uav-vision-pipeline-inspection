"""
Tello – Tubería blanca + puntos verdes de inspección + YOLO (fuego) + SEGUIDOR PID + MEMORIA

AUTO logic:
1) ALIGN       - Alinearse con la tubería blanca usando solo el ángulo
2) FOLLOW_LINE - Seguir tubería blanca con PID (centro + ángulo)
                 Si se pierde la línea:
                     - hasta MAX_MEMORY_TIME s: usa el ÚLTIMO comando (lr, fb, yaw) / 2
                     - después: se detiene (0,0,0,0)
                 En cuanto vuelve a detectar línea, el PID continúa normal.
3) INSPECT     - Cuando se detecta un punto verde dentro de
                 una ROI grande de inspección y ya pasó el cooldown,
                 se detiene, inspecciona fuego con YOLO y marca BUENO/MALO.

- YOLO corre en TODOS los frames en AUTO para dibujar fuego.
- Ya NO se usa un punto verde inicial para arrancar el recorrido.
"""

import cv2
import time
import numpy as np
import keyboard
from djitellopy import Tello
import threading
from ultralytics import YOLO
import contextlib
import io
import math

# =====================================================
#  PARÁMETROS GLOBALES
# =====================================================

COOLDOWN_INSPECTION = 6.0     # segundos entre inspecciones (puedes cambiarlo)
INSPECTION_DURATION = 4.0     # duración de cada inspección en segundos
MAX_INSPECTIONS = 8           # puntos 1..8

# carga modelo YOLO entrenado para fuego
model = YOLO("best.pt")

# =====================================================
#  VIDEO THREAD
# =====================================================

frame_global = None

def video_thread():
    """
    Hilo que recibe el stream del Tello por UDP usando OpenCV (sin PyAV).
    """
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


# =====================================================
#  DETECCIÓN DE TUBERÍA BLANCA (ÁNGULO GLOBAL) – PARA ALIGN
# =====================================================

def detect_white_angle(frame):
    """
    Detecta el contorno blanco principal (tubería) y calcula su ángulo.

    Retorna:
        frame (con rectángulo dibujado),
        angle (float en grados) o None si no hay línea.
    """
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


# =====================================================
#  DETECCIÓN DEL PUNTO VERDE
# =====================================================

def detect_green(frame):
    """
    Detecta un blob verde (punto) y regresa su centro.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # verde un poco más claro
    lower = np.array([35, 80, 80])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = cy = None
    found = False

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 40:  # filtrar ruido
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                found = True

                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    return frame, cx, cy, found


# =====================================================
#  DETECCIÓN TUBERÍA BLANCA PARA PID (CENTRO + ÁNGULO)
# =====================================================

def detect_white_line_angle(frame):
    """
    Versión adaptada del detect_blue_line_angle para tubería BLANCA.
    Devuelve:
      frame_modificado, angle_deg, cx, cy, detected (bool)
    """
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blanco: baja saturación, alto valor
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Opcional: limpieza
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ROI para la línea (zona frontal)
    x1 = w // 2 -200
    y1 = h // 2 - 280
    x2 = w // 2 + 200
    y2 = h // 2 - 90

    # Guías visuales
    cv2.circle(frame, (w // 2, h // 2), 6, (0, 255, 0), -1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 170), 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 170), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

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

    # Ignorar contornos muy pequeños
    MIN_AREA = 1500
    if cv2.contourArea(cnt) < MIN_AREA:
        cv2.putText(frame, "Line too small", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, angle_deg, cx, cy, detected

    # Ajuste de línea
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)

    # Forzar dirección hacia arriba (vy <= 0)
    if vy > 0:
        vx, vy = -vx, -vy

    angle_rad = math.atan2(vx, -vy)
    angle_deg = math.degrees(angle_rad)  # [-90, +90]

    # Dibujar rectángulo y centroide para visualización
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.polylines(frame, [box], True, (255, 0, 0), 2)

    cx, cy = rect[0]
    cx, cy = int(cx), int(cy)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    detected = True
    return frame, angle_deg, cx, cy, detected


# =====================================================
#  DETECTAR FUEGO (YOLO)
# =====================================================

def detect_fire(frame, conf_thresh=0.6):
    """
    Ejecuta YOLO para detectar fuego en el frame.
    Devuelve:
        frame (con bbox dibujado si hay fuego),
        fire_detected (bool),
        cx, cy del fuego (o None, None)
    """
    fire_detected = False
    fx = fy = None

    # Evitar spam de consola capturando stdout de YOLO
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = model(frame, imgsz=320, verbose=False)

    if not results:
        return frame, False, None, None

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return frame, False, None, None

    # Tomar la detección con mayor confianza
    best_conf = 0.0
    best_box = None

    for b in r.boxes:
        conf = float(b.conf[0])
        if conf < conf_thresh:
            continue
        if conf > best_conf:
            best_conf = conf
            best_box = b

    if best_box is None:
        return frame, False, None, None

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    fx = (x1 + x2) // 2
    fy = (y1 + y2) // 2
    fire_detected = True

    # Dibujar bounding box en rojo
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.circle(frame, (fx, fy), 4, (0, 0, 255), -1)
    cv2.putText(frame, "FIRE", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, fire_detected, fx, fy


# =====================================================
#  MAIN
# =====================================================

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

    max_yaw = 120   # usando los límites del PID original
    max_fb  = 50
    max_lr  = 50

    # ---- MEMORIA DE INSPECCIONES ----
    inspection_index = 0
    inspection_results = ["-"] * MAX_INSPECTIONS
    cooldown_inspection = -999.0
    fire_seen = False
    inspect_start = 0.0
    current_point = None

    # ---- ESTADO PID (del primer código) ----
    last_angle = 0
    error_angle = 0
    angle_deg = 0
    angle_raw = 0

    # Inicial Conditions
    integralcx = integralcy = integrala = 0
    derivativocx = derivativocy = derivativoa = 0
    eintbx = eintby = eintba = 0
    incx = incy = inan = 0
    ebx = eby = eba = 0
    cxe = cye = angle_error = 0

    # Gains Proporcional
    Kpcx = 0.13        ## 16
    Kpcy = -0.065
    Kpan = 1.82

    # Gains Integral
    Kicx = 0.15         ##25
    Kicy = -0.03
    Kian = 0.1

    # Gains Derivativos
    Kdcx = 0.1
    Kdcy = -0.01
    Kda = 0.45

    last_time = time.perf_counter()

    # ---- MEMORIA DE LÍNEA PERDIDA ----
    last_lr = 0
    last_fb = 0
    last_yaw = 0
    last_cmd_time = 0.0
    lost_line_time = 0.0
    in_memory_mode = False
    MAX_MEMORY_TIME = 4.0  # segundos de "memoria"

    try:
        while True:

            frame = frame_global
            if frame is None:
                time.sleep(0.01)
                continue

            # Mantener flip vertical
            frame = cv2.flip(frame, 0)
            f = frame.copy()

            h, w = f.shape[:2]
            cx_img = w // 2
            cy_img = h // 2
            cv2.circle(f, (cx_img, cy_img), 4, (0, 255, 0), -1)

            # Detección de fuego (en todo momento en AUTO para dibujar)
            f, fire_now, fx, fy = detect_fire(f, conf_thresh=0.55)

            # ================== CONTROLES ==================
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
                print("AUTO =", auto)
                phase = "ALIGN"
                time.sleep(0.3)

            if keyboard.is_pressed("esc") or keyboard.is_pressed("q"):
                break

            lr = fb = ud = yaw = 0

            # ================== AUTO MODE ==================
            if airborne and auto:

                # ----------------- ALIGN -----------------
                if phase == "ALIGN":
                    f, angle = detect_white_angle(f)

                    cv2.putText(f, "PHASE: ALIGN", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                    if angle is None:
                        yaw = 14
                    else:
                        yaw = int(clamp(angle * 0.8, -25, 25))
                        # Cuando ya está alineado, pasamos DIRECTO a FOLLOW_LINE
                        if abs(angle) < 2.0:
                            phase = "FOLLOW_LINE"
                            cooldown_inspection = time.time()  # arrancar cooldown desde aquí

                    t.send_rc_control(0, 0, 0, yaw)

                # ----------------- FOLLOW_LINE (PID + MEMORIA) -----------------
                elif phase == "FOLLOW_LINE":

                    cv2.putText(f, "PHASE: FOLLOW_LINE (PID)", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)

                    # 1) Detección de punto verde (para inspección)
                    f, gx, gy, green_here = detect_green(f)

                    # ROI GRANDE de inspección (debajo del centro)
                    big_x1 = cx_img - 200
                    big_x2 = cx_img + 200
                    big_y1 = cy_img - 80  # un poco por debajo del centro
                    big_y2 = cy_img + 60  # hacia abajo

                    cv2.rectangle(f, (big_x1, big_y1),
                                     (big_x2, big_y2),
                                     (0, 255, 0), 1)

                    green_in_roi = False
                    if green_here and gx is not None and gy is not None:
                        if (big_x1 <= gx <= big_x2) and (big_y1 <= gy <= big_y2):
                            green_in_roi = True
                            cv2.circle(f, (gx, gy), 6, (0, 0, 255), 2)

                    # 2) PID: cálculo de dt y error angular previos
                    now = time.perf_counter()
                    dt = now - last_time
                    last_time = now
                    if dt <= 0:
                        dt = 1e-3

                    error_angle = angle_raw - last_angle
                    if error_angle > 90 or error_angle < -90:
                        angle_raw = last_angle
                        angle_deg = angle_raw
                    else:
                        last_angle = angle_raw
                        angle_deg = angle_raw

                    # 3) Detección de tubería blanca para PID
                    f, angle_raw, cx, cy, detected_line = detect_white_line_angle(f)

                    if detected_line:
                        # *** VOLVIÓ A VER LÍNEA → SALIR DE MEMORIA ***
                        in_memory_mode = False

                        # Error actual
                        ebx = cxe
                        eby = cye
                        eba = angle_error
                        cxe = cx - cx_img
                        cye = cy - cy_img
                        angle_error = angle_raw

                        # Eje X (lateral)
                        if abs(cxe) > 1:
                            integralcx = eintbx + cxe * dt
                            eintbx = cxe * dt
                            derivativocx = (cxe - ebx) / dt
                            if integralcx > 25:
                                integralcx = 25
                            incx = Kpcx * cxe + Kicx * integralcx + Kdcx * derivativocx
                            incx = int(clamp(incx, -max_lr, max_lr))
                        else:
                            incx = 0

                        # Eje Y (adelante/atrás)
                        if abs(cye) > 3:
                            integralcy = eintby + cye * dt
                            eintby = cye * dt
                            derivativocy = (cye - eby) / dt
                            if integralcy > 25:
                                integralcy = 25
                            incy = Kpcy * cye + Kicy * integralcy + Kdcy * derivativocy
                            incy = int(clamp(incy, -max_fb, max_fb))
                        else:
                            incy = 0

                        # Ángulo
                        if abs(angle_error) > 1:
                            integrala = eintba + angle_error * dt
                            eintba = angle_error * dt
                            derivativoa = (angle_error - eba) / dt
                            if integrala > 50:
                                integrala = 50
                            inan = Kpan * angle_error + Kian * integrala + Kda * derivativoa
                            inan = int(clamp(inan, -max_yaw, max_yaw))
                        else:
                            inan = 0

                        lr = int(incx)
                        fb = int(incy)
                        yaw = int(inan)

                        t.send_rc_control(lr, fb, ud, yaw)

                        # Guardar último comando para memoria
                        last_lr = lr
                        last_fb = fb
                        last_yaw = yaw
                        last_cmd_time = time.time()

                        # HUD
                        cv2.putText(f,
                                    f"y_er:{int(cye):+d} x_er:{int(cxe):+d} a_er:{int(angle_error):+d}",
                                    (w - 330, h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (50, 220, 255), 2)
                        cv2.putText(f,
                                    f"angle = {angle_deg:.1f}, cx {cx}, cy {cy}",
                                    (cx + 8, cy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2)

                    else:
                        # *** NO SE DETECTA LÍNEA → MEMORIA ***
                        cv2.putText(f, "NO WHITE LINE (MEMORY MODE)", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)

                        if not in_memory_mode:
                            in_memory_mode = True
                            lost_line_time = time.time()

                        elapsed_mem = time.time() - lost_line_time

                        if in_memory_mode and elapsed_mem <= MAX_MEMORY_TIME:
                            # Movimiento suave con el último comando / 2
                            lr = int(last_lr * 0.5) - 10
                            fb = int(last_fb * 0.5) + 10
                            yaw = int(last_yaw * 0.5) - 10
                            t.send_rc_control(lr, fb, ud, yaw)

                            cv2.putText(f,
                                        f"MEMORY MOVE ({elapsed_mem:.1f}s)",
                                        (10, 85),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2)
                        else:
                            # Memoria expirada → detener
                            t.send_rc_control(0, 0, 0, 0)
                            cv2.putText(f,
                                        "MEMORY EXPIRED → STOP",
                                        (10, 85),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2)

                    # 4) DISPARO DE INSPECCIÓN POR PUNTO VERDE
                    if (green_in_roi and
                        inspection_index < MAX_INSPECTIONS and
                        (time.time() - cooldown_inspection >= COOLDOWN_INSPECTION)):

                        print(f">> Iniciando INSPECCIÓN en punto {inspection_index+1}")
                        phase = "INSPECT"
                        inspect_start = time.time()
                        fire_seen = False
                        current_point = inspection_index
                        t.send_rc_control(0, 0, 0, 0)
                        continue

                # ----------------- INSPECT -----------------
                elif phase == "INSPECT":

                    cv2.putText(f, f"PHASE: INSPECT (Punto {current_point+1})",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                    if fire_now:
                        fire_seen = True

                    t.send_rc_control(0, 0, 0, 0)

                    elapsed = time.time() - inspect_start
                    cv2.putText(f, f"t_inspect = {elapsed:.1f}s",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                    if elapsed >= INSPECTION_DURATION:
                        if fire_seen:
                            inspection_results[current_point] = "MALO"
                            print(f">> Punto {current_point+1}: MALO (FUEGO)")
                        else:
                            inspection_results[current_point] = "BUENO"
                            print(f">> Punto {current_point+1}: BUENO (SIN FUEGO)")

                        inspection_index += 1
                        cooldown_inspection = time.time()
                        phase = "FOLLOW_LINE"
                        continue

            # ================== MANUAL MODE ==================
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

            cv2.imshow("Tello – WHITE PIPE + FIRE INSPECTION (PID+MEM)", f)
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

        # Resumen final de inspecciones
        print("\n======================")
        print("RESULTADOS DE INSPECCIÓN:")
        for i, res in enumerate(inspection_results, start=1):
            print(f" Punto {i}: {res}")


if __name__ == "__main__":
    main()

