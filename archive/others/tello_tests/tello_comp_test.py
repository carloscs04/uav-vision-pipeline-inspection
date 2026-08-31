# Tello low-latency video + smooth control (Windows)
import os, sys, time, threading
import cv2
import keyboard
from djitellopy import Tello

# Quiet FFmpeg logs
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"

# Low-latency FFmpeg URL (keeps buffers tiny)
URL = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=500000&fflags=nobuffer&flags=low_delay"

t = Tello()
t.connect()
print("Battery:", t.get_battery())

# (optional) lower bitrate to reduce jitter; ignore error if unsupported
try: t.set_video_bitrate(t.BITRATE_1M)
except: pass

cap = None
latest_frame = [None]        # mutable container for thread share
running = True
airborne = False

def capture_loop():
    # drain frames as fast as possible; keep only the newest
    while running:
        # grab multiple times to drop stale frames (if backend queues)
        for _ in range(2):   # tune to 2–5 if needed
            cap.grab()
        ok, frame = cap.retrieve()
        if ok:
            latest_frame[0] = frame

try:
    t.streamon()
    time.sleep(1.5)

    # Open capture with FFmpeg backend
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    # Ask OpenCV to keep tiny buffer (some builds honor this)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warm-up (non-blocking): wait up to 2s for first frame
    start = time.time()
    while latest_frame[0] is None and time.time() - start < 2.0:
        ok, f = cap.read()
        if ok: latest_frame[0] = f

    # Start capture thread (keeps freshest frame in latest_frame[0])
    thr = threading.Thread(target=capture_loop, daemon=True)
    thr.start()

    print("Keys: SPACE=Takeoff  L=Land  W/A/S/D=move  R/F=up/down  Q/E=yaw  ESC=exit")
    speed = 30   # [-100..100]

    # ---- Main loop (aim ~30–40 Hz) ----
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        # one-shot actions
        if not airborne and keyboard.is_pressed('space'):
            print("Taking off...")
            t.takeoff()
            airborne = True
            time.sleep(0.2)
        if airborne and keyboard.is_pressed('l'):
            print("Landing...")
            t.land()
            airborne = False
            time.sleep(0.2)

        # smooth RC (only when airborne)
        lr = fb = ud = yaw = 0
        if airborne:
            if keyboard.is_pressed('w'): fb =  speed
            if keyboard.is_pressed('s'): fb = -speed
            if keyboard.is_pressed('a'): lr = -speed
            if keyboard.is_pressed('d'): lr =  speed
            if keyboard.is_pressed('r'): ud =  speed
            if keyboard.is_pressed('f'): ud = -speed
            if keyboard.is_pressed('q'): yaw = -speed
            if keyboard.is_pressed('e'): yaw =  speed
            t.send_rc_control(lr, fb, ud, yaw)
        else:
            t.send_rc_control(0, 0, 0, 0)

        # show the most recent frame (no blocking read here)
        frame = latest_frame[0]
        if frame is not None:
            # frame = cv2.resize(frame, (960, 720))   # optional, comment out for lowest latency
            cv2.imshow("Tello (low-latency)", frame)

        time.sleep(0.025)   # ~40 Hz (reduce latency vs 0.05)

finally:
    running = False
    try: t.send_rc_control(0, 0, 0, 0)
    except: pass
    cv2.destroyAllWindows()
    try:
        if airborne:
            t.land()
            airborne = False
    except: pass
    if cap is not None:
        try: cap.release() 
        except: pass
    try: t.streamoff()
    except: pass
    t.end()
