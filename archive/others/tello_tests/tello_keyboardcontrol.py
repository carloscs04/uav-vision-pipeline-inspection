from djitellopy import Tello
import keyboard
import time

tello = Tello()
tello.connect()
tello.streamon()

print("Battery:", tello.get_battery())
tello.takeoff()

try:
    while True:
        lr = 0
        fb = 0
        ud = 0
        yaw = 0

        speed = 30  # change this to make movement faster/slower

        if keyboard.is_pressed('w'):
            fb = speed
        if keyboard.is_pressed('s'):
            fb = -speed
        if keyboard.is_pressed('a'):
            lr = -speed
        if keyboard.is_pressed('d'):
            lr = speed
        if keyboard.is_pressed('q'):
            yaw = -speed
        if keyboard.is_pressed('e'):
            yaw = speed
        if keyboard.is_pressed('r'):
            ud = speed
        if keyboard.is_pressed('f'):
            ud = -speed

        tello.send_rc_control(lr, fb, ud, yaw)

        if keyboard.is_pressed('x'):
            break

        time.sleep(0.05)   # 20 Hz control loop

finally:
    tello.send_rc_control(0,0,0,0)
    tello.land()
    tello.end()

