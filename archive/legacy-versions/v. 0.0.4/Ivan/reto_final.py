import time, math, io, contextlib; import cv2, numpy as np, keyboard, threading, serial; from djitellopy import Tello
from ultralytics import YOLO; from pywifi import PyWiFi, const, Profile

model = YOLO("D:\carlo\Documents\Python\Concentration\Reto\v. 0.0.4\Ivan\best.pt")   # 0=FIRE, 1=ANOMALY, 2=SHOE           # ================= YOLO & ARDUINO =================
PUERTO = "COM14"
arduino = serial.Serial(PUERTO, 9600, timeout=1)

frame_global = None # ================= VIDEO THREAD =================
def video_thread():
    global frame_global
    cap = cv2.VideoCapture("udp://0.0.0.0:11111", cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[ERROR] Unable to open stream"); return
    while True:
        ok, frame = cap.read()
        if ok: frame_global = cv2.resize(frame, (640, 480))
        else: time.sleep(0.01)

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ================= DETECCIÓN TUBERÍA (ALIGN) =================
def detect_white_angle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 245]); upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = None
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c); box = cv2.boxPoints(rect); box = np.int32(box)
        cv2.drawContours(frame, [box], 0, (255,0,0), 2)
        angle_raw = rect[2]; w,h = rect[1]
        angle_calc = angle_raw
        if w < h: angle_calc += 90
        angle_calc -= 90
        if angle_calc > 75 or angle_calc < -75: angle_calc = 0
        angle = angle_calc
        cv2.putText(frame,f"White angle: {angle:+.1f}",(10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
    return frame, angle

# ================= DETECCIÓN LÍNEA BLANCA (PID) =================
def detect_white_line_angle(frame):
    h,w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,245]); upper_white = np.array([180,50,255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    x1 = w//2 - 200; y1 = h//2 - 280
    x2 = w//2 + 200; y2 = h//2 - 90

    cv2.circle(frame,(w//2,h//2),6,(0,255,0),-1)
    cv2.line(frame,(0,h//2),(w,h//2),(0,255,170),1)
    cv2.line(frame,(w//2,0),(w//2,h),(0,255,170),1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)

    roi_mask = np.zeros_like(mask,dtype=np.uint8)
    cv2.rectangle(roi_mask,(x1,y1),(x2,y2),255,-1)
    mask_roi = cv2.bitwise_and(mask,roi_mask)

    cnts,_ = cv2.findContours(mask_roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cx = cy = 0; angle_deg = 0; detected = False

    if not cnts:
        cv2.putText(frame,"No white line",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        return frame, angle_deg, cx, cy, detected

    cnt = max(cnts,key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1500:
        cv2.putText(frame,"Line too small",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        return frame, angle_deg, cx, cy, detected

    [vx,vy,x0,y0] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
    vx,vy = float(vx),float(vy)
    if vy > 0: vx,vy = -vx,-vy
    angle_rad = math.atan2(vx,-vy); angle_deg = math.degrees(angle_rad)

    rect = cv2.minAreaRect(cnt); box = cv2.boxPoints(rect); box = np.intp(box)
    cv2.polylines(frame,[box],True,(255,0,0),2)
    cx,cy = rect[0]; w_box,h_box = rect[1]
    cx,cy = int(cx),int(cy)
    cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
    if (h_box / w_box) > 0.85: angle_deg = 0
    return frame, angle_deg, cx, cy, True

# ================= YOLO (FIRE / ANOMALY / SHOE) =================
def detect_yolo(frame, conf_thresh=0.55):
    fire_detected = anomaly_detected = shoe_detected = False
    fx = fy = ax = ay = sx = sy = None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = model(frame, imgsz=320, verbose=False)
    if not results: return False,False,False,None,None,None,None,None,None,frame
    r = results[0]
    if r.boxes is None or len(r.boxes)==0:
        return False,False,False,None,None,None,None,None,None,frame

    for b in r.boxes:
        cls = int(b.cls[0]); conf = float(b.conf[0])
        if conf < conf_thresh: continue
        x1,y1,x2,y2 = map(int,b.xyxy[0].tolist())
        cx,cy = (x1+x2)//2,(y1+y2)//2

        if cls==0: # FIRE
            fire_detected = True; fx,fy = cx,cy
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,"FIRE",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

        elif cls==1: # ANOMALY
            anomaly_detected = True; ax,ay = cx,cy
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,"ANOMALY",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.circle(frame,(cx,cy),4,(0,255,0),-1)

        elif cls==2: # SHOE
            shoe_detected = True; sx,sy = cx,cy
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.putText(frame,"SHOE",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2)
            cv2.circle(frame,(cx,cy),4,(255,0,255),-1)

    return fire_detected, anomaly_detected, shoe_detected, fx, fy, ax, ay, sx, sy, frame

# ================= MISIÓN DRON 1 =================
def drone1_mission():
    global frame_global
    COOLDOWN_INSPECTION = 3.7; INSPECTION_DURATION = 4.0; MAX_INSPECTIONS = 8

    t = Tello(); t.connect(); print("[DRON 1] Battery:", t.get_battery())
    t.streamon(); threading.Thread(target=video_thread,daemon=True).start()

    airborne = False; auto = False; phase = "ALIGN"
    max_yaw = 120; max_fb = 50; max_lr = 50

    inspection_index = 0
    inspection_results = ["-"]*MAX_INSPECTIONS
    cooldown_inspection = -999.0
    fire_seen = False; inspect_start = 0.0; current_point = None

    # PID state
    last_angle = error_angle = angle_deg = angle_raw = 0
    integralcx = integralcy = integrala = 0
    derivativocx = derivativocy = derivativoa = 0
    eintbx = eintby = eintba = 0
    incx = incy = inan = 0
    ebx = eby = eba = 0
    cxe = cye = angle_error = 0

    Kpcx = 0.11; Kpcy = -0.065; Kpan = 0.82
    Kicx = 0.15; Kicy = -0.023; Kian = 0.4
    Kdcx = 0.1;  Kdcy = -0.01;  Kda  = 0.15

    last_time = time.perf_counter()

    # memory mode
    last_lr = last_fb = last_yaw = 0
    lost_line_time = 0
    in_memory_mode = False
    MAX_MEMORY_TIME = 4.0

    # segmentos para dron 2
    segments = []
    segment_active = False
    segment_start_time = 0.0; segment_fb_sum = 0.0; segment_fb_count = 0

    # pausa por shoe
    in_shoe_pause = False; shoe_pause_start = 0.0

    # elevación robusta
    elevation_started = False; elevation_cycle_start = 0.0
    pipe_side = None; side_search_start = 0.0
    SEARCH_SIDE_TIME = 3.0
    SIDE_MEMORY_TIME = 4.0

    try:
        print("\n[DRON 1] SPACE=takeoff, t=AUTO, l=land")
        while True:
            frame = frame_global
            if frame is None:
                time.sleep(0.01); continue

            frame = cv2.flip(frame,0)
            f = frame.copy()
            h,w = f.shape[:2]; cx_img,cy_img = w//2,h//2
            cv2.circle(f,(cx_img,cy_img),4,(0,255,0),-1)

            fire_now, anomaly_now, shoe_now, fx,fy,ax,ay,sx,sy,f = detect_yolo(f,conf_thresh=0.45)

            # teclas básicas
            if keyboard.is_pressed("space") and not airborne:
                t.takeoff(); airborne = True; auto = False; phase = "ALIGN"; time.sleep(0.3)
            if keyboard.is_pressed("l") and airborne:
                t.land(); airborne = False; break
            if keyboard.is_pressed("t"):
                auto = not auto; print("[DRON 1] AUTO =", auto); phase = "ALIGN"; time.sleep(0.3)
            if keyboard.is_pressed("esc"): break

            lr = fb = ud = yaw = 0

            # ================= AUTO =================
            if airborne and auto:
                # pausa por shoe (no INSPECT)
                if shoe_now and phase != "INSPECT":
                    if not in_shoe_pause:
                        in_shoe_pause = True; arduino.write(b'2')
                        shoe_pause_start = time.time()
                        print("[DRON 1] SHOE DETECTED → PAUSE")
                    t.send_rc_control(0,0,0,0)
                    cv2.putText(f,"SHOE DETECTED - PAUSE",(10,100),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2)
                    cv2.imshow("Tello – PID + Anomaly + Memory + Shoe",f)
                    if cv2.waitKey(1)&0xFF==27: break
                    time.sleep(0.012); continue

                if in_shoe_pause and (not shoe_now):
                    pause_dt = time.time() - shoe_pause_start
                    cooldown_inspection += pause_dt
                    in_shoe_pause = False; arduino.write(b'0')
                    print(f"[DRON 1] SHOE desapareció, reanudando (pause={pause_dt:.1f}s)")

                # -------- ALIGN --------
                if phase == "ALIGN":
                    f,angle = detect_white_angle(f)
                    cv2.putText(f,"PHASE: ALIGN",(10,25),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                    if angle is None:
                        fb = 10
                    else:
                        yaw = int(clamp(angle*0.8,-25,25))
                        if 0 < yaw <= 5: yaw = 5
                        if -5 <= yaw < 0: yaw = -5
                        if abs(angle) < 3.0:
                            phase = "FOLLOW_LINE"
                            cooldown_inspection = time.time()
                    t.send_rc_control(0,0,0,yaw)

                # -------- FOLLOW_LINE --------
                elif phase == "FOLLOW_LINE":
                    cv2.putText(f,"PHASE: FOLLOW_LINE (PID)",(10,25),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

                    big_x1 = cx_img-200; big_x2 = cx_img+200
                    big_y1 = cy_img-80;  big_y2 = cy_img+190
                    cv2.rectangle(f,(big_x1,big_y1),(big_x2,big_y2),(0,255,0),1)

                    anomaly_in_roi = False
                    if anomaly_now and ax is not None and ay is not None:
                        if big_x1 <= ax <= big_x2 and big_y1 <= ay <= big_y2:
                            anomaly_in_roi = True
                            cv2.circle(f,(ax,ay),6,(0,0,255),2)

                    now = time.perf_counter()
                    dt = now - last_time; last_time = now
                    if dt <= 0: dt = 1e-3

                    error_angle = angle_raw - last_angle
                    if abs(error_angle) > 90:
                        angle_raw = last_angle; angle_deg = angle_raw
                    else:
                        last_angle = angle_raw; angle_deg = angle_raw

                    f,angle_raw,cx,cy,detected_line = detect_white_line_angle(f)

                    if detected_line:
                        in_memory_mode = False
                        ebx,eby,eba = cxe,cye,angle_error
                        cxe = cx - cx_img; cye = cy - cy_img; angle_error = angle_raw

                        if abs(cxe)>1:
                            integralcx = eintbx + cxe*dt; eintbx = cxe*dt
                            derivativocx = (cxe-ebx)/dt
                            integralcx = min(integralcx,25)
                            incx = Kpcx*cxe + Kicx*integralcx + Kdcx*derivativocx
                            incx = int(clamp(incx,-max_lr,max_lr))
                        else: incx = 0

                        if abs(cye)>3:
                            integralcy = eintby + cye*dt; eintby = cye*dt
                            derivativocy = (cye-eby)/dt
                            integralcy = min(integralcy,25)
                            incy = Kpcy*cye + Kicy*integralcy + Kdcy*derivativocy
                            incy = int(clamp(incy,-max_fb,max_fb))
                        else: incy = 0

                        if abs(angle_error)>1:
                            integrala = eintba + angle_error*dt; eintba = angle_error*dt
                            derivativoa = (angle_error-eba)/dt
                            integrala = min(integrala,50)
                            inan = (Kpan*angle_error + Kian*integrala + Kda*derivativoa)
                            inan = int(clamp(inan,-max_yaw,max_yaw))
                        else: inan = 0

                        lr,fb,yaw = int(incx),int(incy),int(inan)
                        t.send_rc_control(lr,fb+12,ud,yaw)
                        last_lr,last_fb,last_yaw = lr,fb,yaw

                        if segment_active:
                            segment_fb_sum += fb; segment_fb_count += 1

                        cv2.putText(f,
                                    f"y_er:{int(cye)} x_er:{int(cxe)} a_er:{int(angle_error)}",
                                    (w-330,h-15),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(50,220,255),2)

                        lost_line_time = time.time()

                    else:
                        cv2.putText(f,"NO WHITE LINE (MEMORY MODE)",(10,60),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                        if not in_memory_mode:
                            in_memory_mode = True
                            lost_line_time = time.time()

                        elapsed_mem = time.time() - lost_line_time

                        # 🔴 FIX: si ya hay 8 inspecciones → sólo memoria y luego LAND (sin elevación)
                        if inspection_index >= MAX_INSPECTIONS and elapsed_mem >= MAX_MEMORY_TIME:
                            print(">>> 8 INSPECTIONS COMPLETE — LOST PIPE 3s — LANDING")
                            t.send_rc_control(0,0,0,0)
                            try: t.land()
                            except Exception as e: print("Error en landing final:",e)
                            airborne = False; break

                        if elapsed_mem <= MAX_MEMORY_TIME:
                            lr = int(last_lr*0.5) - 13
                            fb = int(last_fb*0.5) + 9
                            yaw = int(last_yaw*0.5) - 19
                            t.send_rc_control(lr,fb,ud,yaw)
                            cv2.putText(f,f"MEMORY MOVE ({elapsed_mem:.1f}s)",
                                        (10,85),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                        else:
                            # aquí sí habilitamos elevación robusta
                            print(">>> LOST LINE > 3s → ELEVATION_SEARCH (ROBUST)")
                            phase = "ELEVATION_SEARCH"
                            in_memory_mode = False
                            elevation_started = False
                            elevation_cycle_start = 0.0
                            pipe_side = None
                            side_search_start = 0.0
                            t.send_rc_control(0,0,0,0)
                            continue

                    # trigger de INSPECT
                    if (anomaly_in_roi and
                        inspection_index < MAX_INSPECTIONS and
                        (time.time() - cooldown_inspection) >= COOLDOWN_INSPECTION):

                        if segment_active and segment_fb_count>0:
                            seg_duration = time.time() - segment_start_time
                            fb_avg = segment_fb_sum/segment_fb_count
                            seg_id = len(segments)+1
                            segments.append((seg_id,fb_avg,seg_duration))
                            print(f"[SEGMENT {seg_id}] FB_AVG={fb_avg:.3f}, TIME={seg_duration:.3f}s")
                        segment_active = False

                        print(f">> INSPECTING anomaly at point {inspection_index+1}")
                        phase = "INSPECT"
                        inspect_start = time.time()
                        fire_seen = False
                        current_point = inspection_index
                        t.send_rc_control(0,-3,0,0)
                        continue

                # -------- ELEVATION_SEARCH (robusto) --------
                elif phase == "ELEVATION_SEARCH":
                    cv2.putText(f,"PHASE: ELEVATION_SEARCH",(10,25),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

                    if not elevation_started:
                        try: t.move_up(50)
                        except Exception as e: print("Error en move_up(50):",e)
                        elevation_started = True
                        elevation_cycle_start = time.time()
                        pipe_side = None; side_search_start = 0.0
                        integralcx = integralcy = integrala = 0
                        eintbx = eintby = eintba = 0
                        cxe = cye = angle_error = 0; last_angle = 0
                        print(">>> Elevation search: subida de 50 cm (nuevo ciclo)")

                    f2 = f.copy()
                    f2,angle_raw2,cx2,cy2,detected_line2 = detect_white_line_angle(f2)

                    if pipe_side is None:
                        elapsed_since_elev = time.time() - elevation_cycle_start
                        if detected_line2:
                            pipe_side = "LEFT" if cx2 < cx_img else "RIGHT"
                            side_search_start = time.time()
                            print(f">>> PIPE SIDE DETECTADO: {pipe_side}")
                        elif elapsed_since_elev >= SEARCH_SIDE_TIME:
                            pipe_side = "LEFT"
                            side_search_start = time.time()
                            print(">>> No se detectó tubería arriba, asumiendo LEFT")
                        t.send_rc_control(0,0,0,0)

                    else:
                        f3 = f.copy()
                        f3,angle_raw3,cx3,cy3,detected_line3 = detect_white_line_angle(f3)

                        if detected_line3:
                            print(">>> Línea encontrada después de ELEVATION_SEARCH lateral")
                            try: t.move_down(40)
                            except Exception as e: print("Error en move_down(50):",e)
                            phase = "FOLLOW_LINE"
                            lost_line_time = time.time()
                            in_memory_mode = False
                            t.send_rc_control(0,0,0,0)
                            cxe = cx3 - cx_img; cye = cy3 - cy_img; angle_error = angle_raw3
                            continue

                        elapsed_side = time.time() - side_search_start
                        if pipe_side == "LEFT":
                            lr,fb,yaw = -14,11,-13
                        else:
                            lr,fb,yaw = 14,11,13

                        t.send_rc_control(lr,fb,0,yaw)
                        cv2.putText(f,f"SIDE SEARCH {pipe_side} ({elapsed_side:.1f}s)",
                                    (10,55),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

                        if elapsed_side >= SIDE_MEMORY_TIME:
                            print(">>> Side search sin línea → bajar y reintentar luego")
                            try: t.move_down(40)
                            except Exception as e: print("Error en move_down(50) tras side search:",e)
                            elevation_started = False
                            pipe_side = None; side_search_start = 0.0
                            phase = "FOLLOW_LINE"
                            lost_line_time = time.time()
                            in_memory_mode = False
                            t.send_rc_control(0,0,0,0)

                # -------- INSPECT --------
                elif phase == "INSPECT":
                    cv2.putText(f,f"PHASE: INSPECT (Point {current_point+1})",
                                (10,25),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                    if fire_now: fire_seen = True
                    t.send_rc_control(0,0,0,0)

                    elapsed = time.time() - inspect_start
                    cv2.putText(f,f"t_inspect={elapsed:.1f}s",
                                (10,50),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                    if elapsed >= INSPECTION_DURATION:
                        if fire_seen:
                            arduino.write(b'1')
                            inspection_results[current_point] = "MALO"
                            print(f">> Point {current_point+1}: MALO (FIRE)")
                        else:
                            inspection_results[current_point] = "BUENO"
                            print(f">> Point {current_point+1}: BUENO (NO FIRE)")

                        inspection_index += 1
                        cooldown_inspection = time.time()

                        # 🔴 FIX IMPORTANTE:
                        # Al salir de INSPECT siempre reiniciamos memoria
                        in_memory_mode = False
                        lost_line_time = time.time()
                        elevation_started = False
                        pipe_side = None
                        side_search_start = 0.0

                        if inspection_index < MAX_INSPECTIONS:
                            segment_active = True
                            segment_start_time = time.time()
                            segment_fb_sum = 0.0; segment_fb_count = 0

                        phase = "FOLLOW_LINE"
                        continue

            # ================= MANUAL =================
            elif airborne:
                if keyboard.is_pressed("w"): fb = max_fb
                if keyboard.is_pressed("s"): fb = -max_fb
                if keyboard.is_pressed("a"): lr = -max_lr
                if keyboard.is_pressed("d"): lr = max_lr
                if keyboard.is_pressed("q"): yaw = -max_yaw
                if keyboard.is_pressed("e"): yaw = max_yaw
                if keyboard.is_pressed("r"): ud = max_lr
                if keyboard.is_pressed("f"): ud = -max_lr
                t.send_rc_control(lr,fb,ud,yaw)

            cv2.imshow("Tello – PID + Anomaly + Memory + Shoe",f)
            if cv2.waitKey(1)&0xFF==27: break
            time.sleep(0.012)

    finally:
        try: t.send_rc_control(0,0,0,0)
        except: pass
        try:
            if airborne: t.land()
        except: pass
        cv2.destroyAllWindows()
        try: t.end()
        except: pass

        print("\n===== INSPECTION SUMMARY (DRON 1) =====")
        for i,res in enumerate(inspection_results,start=1):
            print(f"Point {i}: {res}")

        try:
            with open("traj_segments.txt","w") as ftxt:
                ftxt.write("SEGMENTS (between inspections, starting AFTER first inspect):\n")
                for seg_id,fb_avg,duration in segments:
                    ftxt.write(f"SEGMENT {seg_id}\n")
                    ftxt.write(f"FB_AVG={fb_avg:.3f}\n")
                    ftxt.write(f"TIME={duration:.3f}\n")
                    if seg_id in (2,4): ftxt.write("TURN_90_LEFT=1\n")
                    ftxt.write("\n")
                ftxt.write("POINT_RESULTS:\n")
                for i,res in enumerate(inspection_results,start=1):
                    ftxt.write(f"POINT {i}={res}\n")
            print("\n>>> Archivo 'traj_segments.txt' generado correctamente.")
        except Exception as e:
            print("Error writing traj_segments.txt:", e)

# ================= DRON 2 =================
def read_fire_points(filename):
    fire = []; inside = False
    with open(filename,"r") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            if ("POINT_RESULTS" in line) and not inside:
                inside = True; continue
            if inside and line.startswith("POINT") and "=" in line:
                left,result = line.split("=",1)
                parts = left.strip().split()
                if len(parts)<2: continue
                try: idx = int(parts[1])
                except: continue
                fire.append(1 if result.upper()=="MALO" else 0)
    print("\n=== FIRE LISTA GENERADA (DRON 2) ==="); print(fire); print("====================================\n")
    return fire

def second_drone_mission(fire):
    t = Tello(); t.connect(); print("Drone 2 conectado!")
    t.takeoff(); time.sleep(2)
    c = 360; d = 410
    coords = [
        (0,0),
        (0,c/2),
        (0,c),
        ((-d-5)/2,c),
        (-(3*d-10)/4,c),
        (-d-10,c),
        (-d,(c/2)-10),
        (-d,0)
    ]
    fp = [0]
    for i,val in enumerate(fire):
        if val==1: fp.append(i)
    fp.append(0)
    print(f"Puntos a visitar: {fp}")
    last_yaw = 0

    for i in range(len(fp)-1):
        p1,p2 = fp[i],fp[i+1]
        x1,y1 = coords[p1]; x2,y2 = coords[p2]
        dx,dy = x2-x1,y2-y1
        print(f"\nVECTOR: ({dx},{dy})")
        angle_world = -math.degrees(math.atan2(dx,dy))
        rotation = angle_world - last_yaw
        rotation = (rotation+180)%360 - 180
        dist = int(math.sqrt(dx*dx + dy*dy))
        print(f"Rotate = {int(rotation)} deg"); print(f"Move   = {dist} cm")
        last_yaw += rotation; last_yaw = (last_yaw+180)%360 - 180
        print(f"New yaw = {int(last_yaw)}")

        if abs(rotation)>1:
            if rotation>0:
                t.rotate_counter_clockwise(int(abs(rotation)))
            else:
                t.rotate_clockwise(int(abs(rotation)))
            print("rotation",int(abs(rotation))); time.sleep(1.5)

        if dist>0:
            if dist>=500:
                t.move_forward(500); t.move_forward(dist-500); arduino.write(b'1')
            else:
                t.move_forward(dist); arduino.write(b'1')
            print("distance",int(abs(dist))); time.sleep(2)

    t.land(); print("Drone 2 finalizó su misión.")

# ================= WIFI =================
def connect_wifi(ssid,interface_index=1):
    wifi = PyWiFi(); interfaces = wifi.interfaces()
    if interface_index >= len(interfaces):
        print(f"Error: invalid interface index {interface_index}."); return None
    iface = interfaces[interface_index]; print(f"Connecting to {ssid} via {iface.name}")
    profile = Profile(); profile.ssid = ssid
    profile.auth = const.AUTH_ALG_OPEN
    profile.akm.append(const.AKM_TYPE_NONE)
    profile.cipher = const.CIPHER_TYPE_NONE
    iface.remove_all_network_profiles(); iface.add_network_profile(profile); iface.connect(profile)
    time.sleep(5)
    if iface.status()==const.IFACE_CONNECTED:
        print(f"Connected to {ssid}!"); return iface
    else:
        print(f"Failed to connect to {ssid}"); return None

def disconnect_wifi(interface_index=1):
    wifi = PyWiFi(); interfaces = wifi.interfaces()
    if interface_index >= len(interfaces):
        print(f"Error: invalid interface index {interface_index}."); return
    iface = interfaces[interface_index]
    iface.remove_all_network_profiles()
    print(f"Disconnected from WiFi on {iface.name}")

# ================= MAIN =================
def main():
    if connect_wifi("TELLO-9A57E0",interface_index=1):
        print("\n[MAIN] Ejecutando misión DRON 1 (inspecciones + TXT)...")
        drone1_mission()
        disconnect_wifi(interface_index=1)

    if connect_wifi("TELLO-FE193A",interface_index=1):
        print("\n[MAIN] Ejecutando misión DRON 2 (trayectoria con MALO)...")
        fire = read_fire_points("traj_segments.txt")
        second_drone_mission(fire)
        disconnect_wifi(interface_index=1)

if __name__=="__main__":
    main()