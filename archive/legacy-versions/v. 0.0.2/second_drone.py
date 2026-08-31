import time
import math
import io
import contextlib

import cv2
import numpy as np
import keyboard
import threading
from djitellopy import Tello

from pywifi import PyWiFi, const, Profile

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

        if abs(rotation) > 1:
            if rotation > 0:
                t.rotate_counter_clockwise(int(abs(rotation)))
                print("rotation",int(abs(rotation)))
            else:
                t.rotate_clockwise(int(abs(rotation)))
                print("rotation",int(abs(rotation)))
            time.sleep(1.5)

        if dist > 0:
            if dist >= 500:
                t.move_forward(500)
                t.move_forward(dist- 500)  
            else:
                t.move_forward(dist)
            print("distance", int(abs(dist)))
            time.sleep(2)

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
    fire = [0,1,0,1,0,1,0,1]

    #second_drone_mission(fire)

    # =============================
    #   ✈️ DRON 2 (TELLO-FE193A)
    # =============================
    if connect_wifi("TELLO-FE193A", interface_index=1):
        print("\n[MAIN] Ejecutando misión DRON 2 (trayectoria con MALO)...")
        second_drone_mission(fire)
        disconnect_wifi(interface_index=1)

if __name__ == "__main__":
    main()
