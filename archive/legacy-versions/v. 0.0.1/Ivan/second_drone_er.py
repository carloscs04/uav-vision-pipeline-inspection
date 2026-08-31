import time
from pywifi import PyWiFi, const, Profile
from djitellopy import Tello  # Import Tello SDK
import math

def connect_wifi(ssid, interface_index):
    wifi = PyWiFi()  # Create a PyWiFi object
    interfaces = wifi.interfaces()  # Get all available interfaces

    if interface_index >= len(interfaces):
        print(f"Error: Invalid interface index {interface_index}. Available interfaces: {len(interfaces)}")
        return None

    iface = interfaces[interface_index]  # Select the Wi-Fi interface based on index
    print(f"Connecting to {ssid} using interface {iface.name}")

    profile = Profile()
    profile.ssid = ssid  # Set the SSID to the network you're connecting to
    profile.auth = const.AUTH_ALG_OPEN  # Open network (no password)
    profile.akm.append(const.AKM_TYPE_NONE)  # No encryption
    profile.cipher = const.CIPHER_TYPE_NONE  # No encryption

    iface.remove_all_network_profiles()  # Remove any existing profiles
    iface.add_network_profile(profile)  # Add the new profile
    iface.connect(profile)  # Connect to the network

    time.sleep(5)

    if iface.status() == const.IFACE_CONNECTED:
        print(f"Successfully connected to {ssid} on {iface.name}")
        return iface
    else:
        print(f"Failed to connect to {ssid} on {iface.name}")
        return None

def disconnect_wifi(interface_index):
    wifi = PyWiFi()  # Create a PyWiFi object
    interfaces = wifi.interfaces()  # Get all available interfaces

    if interface_index >= len(interfaces):
        print(f"Error: Invalid interface index {interface_index}. Available interfaces: {len(interfaces)}")
        return

    iface = interfaces[interface_index]  # Select the Wi-Fi interface based on index
    print(f"Disconnecting from Wi-Fi using interface {iface.name}")

    iface.remove_all_network_profiles()  # Remove the active network profile to disconnect
    print(f"Disconnected from Wi-Fi using {iface.name}")

def second_drone(fire):
    #drone.connect()
    #print("Drone connected.")
    #drone.takeoff()  # Take off

    last_angle = 0
    angle_move = 0
    angle_raw = 0
    angle_correction = 0

    c = 370  # centimeters of the square
    d = 440
    coordenates = [(0, 0), (0, c / 2), (0, c), (-d / 2, c), (-(3 * d) / 4, c), (-d, c), (-d, c / 2), (-d, 0)]
    fp = [0]
    for index in range(len(fire)):
        if fire[index] == 1:
            fp.append(index)

    fp.append(0)

    print("FPPPP",fp)

    # Ensure that the list has at least two elements to work with
    if len(fp) < 2:
        print("Error: Not enough points in fire to calculate movement.")
        return

    # Loop through the valid pairs of indices in fp to calculate movement and angle
    for i in range(len(fp)-1):  # Adjust loop to avoid out of bounds when using i+1
        # Access coordinates using the indices in fp
        raw_vector = (coordenates[fp[i+1]][0] - coordenates[fp[i]][0], 
                      coordenates[fp[i+1]][1] - coordenates[fp[i]][1])

        # Calculate the distance (y_move)
        y_move = int(math.sqrt(raw_vector[0]**2 + raw_vector[1]**2))

        last_angle = angle_raw + angle_correction

        if last_angle < -45 and last_angle > -90: 
            last_angle = -90
        
        # Calculate the angle of movement (angle_move) using atan2 for better handling of x=0
        if (raw_vector[0] > 0 or raw_vector[1] > 0) and (raw_vector[0] !=0 and raw_vector[1] !=0):
            angle_raw = int(math.degrees(math.atan(raw_vector[0]/ raw_vector[1])))
            if last_angle < -45: last_angle = 0
            
        else:    
            angle_raw = int(math.degrees(math.atan2(raw_vector[0], raw_vector[1])))
        
        if angle_raw > 0: angle_raw = - angle_raw
        if (angle_raw == -180 and last_angle == -90 and raw_vector[0] > 0): angle_raw = 0
        if angle_raw == 0 and angle_correction != 0 and last_angle != -90: 
            last_angle = 0
            print("HOLA")

        print(raw_vector)
        print("angle raw", angle_raw)

        angle_move = -int(abs(angle_raw - last_angle))  # This ensures correct positive angle
        angle_correction = -int(abs(angle_move + 90))
        if raw_vector[0] < 0 and raw_vector[1] < 0: 
            angle_correction = -int(abs(angle_correction + 90))
            print("HOLA")
            if angle_move == angle_correction:
                angle_correction = -int(abs(angle_correction + 90))
                print("ADIOS")
        if angle_correction <= -90 or angle_correction >= 90: angle_correction = 0

        if len(fp) == 3 and i == 1: 
            angle_move = -int(180 + angle_raw) 
          

        print("last_angle", last_angle)
        print(f"Movement {i}: Distance = {y_move:.2f} cm, angle move = {angle_move:.2f} degrees")
        print("correction_angle", angle_correction)

        #try:
            #print(f"Rotating by {abs(angle_move)} degrees.")
            #drone.rotate_counter_clockwise(abs(angle_move))  # Correctly passing angle
            #time.sleep(2)
            #drone.move_forward(y_move)  # Pass y_move directly as distance
            #time.sleep(3)
            #drone.rotate_counter_clockwise(abs(angle_correction))  # Rotate with correction angle
            #time.sleep(2)
        #except Exception as e:
            #print(f"Error while sending command to Tello: {e}")

    time.sleep(2)  # Simulate drone flying for 5 seconds
    #drone.land()  # Land the drone
    #print("Drone landed.")





def main():
    # Example usage
    # Step 1: Connect to Tello's Wi-Fi (the drone's Wi-Fi network)
    #connect_wifi('TELLO-9A57E0', interface_index=1)

    #drone = Tello()

    #fire = [0,0,1,1,0,0,1,1] # Ya jala

    #fire = [0,0,1,0,0,0,1,1] # Ya jala

    #fire = [0,0,0,1,0,0,1,0] # Ya jala

    #fire = [0,1,0,1,0,1,0,1] # Ya jala   

    #fire = [0,0,0,0,1,0,1,0] # Ya jala

    #fire = [0,0,1,0,0,0,1,1] # Ya jala
    
    #fire = [0,0,1,0,0,0,0,1] # Ya jala

    #fire = [0,0,0,0,0,1,1,0]  # Ya jalo

    #fire = [0,1,1,1,1,1,1,1] # Ya jalo

    #fire = [0,0,1,0,0,0,0,0] # Ya jalo

    fire = [0,0,0,1,0,0,0,0]

    second_drone(fire)

    
    # Step 3: Disconnect from the Tello Wi-Fi after the drone has landed
    #disconnect_wifi(interface_index=1)

    # Step 4: Connect to another Wi-Fi network
    # connect_wifi('TELLO-FE193A', interface_index=1)

    # drone.connect()
    # print("Drone connected.")
    # drone.takeoff()  # Take off
    # time.sleep(5)  # Simulate drone flying for 5 seconds
    # drone.land()  # Land the drone
    # print("Drone landed.")

    # Step 6: Disconnect from the new Wi-Fi
    #disconnect_wifi(interface_index=1)

if __name__ == "__main__":
    main()