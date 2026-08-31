from djitellopy import Tello
import cv2, time, os

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"  # hide noisy logs
URL = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000&fflags=nobuffer&flags=low_delay"

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

cap = None
try:
    tello.streamon()
    time.sleep(2.0)

    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

    print("[INFO] Taking off...")
    tello.takeoff()

    while True:
        ok, frame = cap.read()
        if ok:
            cv2.imshow("Tello", cv2.resize(frame, (960, 720)))
        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("[INFO] Landing...")
    tello.land()

finally:
    cv2.destroyAllWindows()
    try:
        if cap: cap.release()
    except: pass
    try: tello.streamoff()
    except: pass
    tello.end()
