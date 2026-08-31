import time
from pywifi import PyWiFi, const, Profile
from djitellopy import Tello  # Import Tello SDK

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

def take_off_drone():
    # Connect to the drone and take off
    drone = Tello()
    drone.connect()
    print("Drone connected.")
    drone.takeoff()  # Take off
    time.sleep(5)  # Simulate drone flying for 5 seconds
    drone.land()  # Land the drone
    print("Drone landed.")

# Example usage
# Step 1: Connect to Tello's Wi-Fi (the drone's Wi-Fi network)
if connect_wifi('TELLO-9A57E0', interface_index=1):
    # Step 2: Control the drone
    take_off_drone()

    # Step 3: Disconnect from the Tello Wi-Fi after the drone has landed
    disconnect_wifi(interface_index=1)

    # Step 4: Connect to another Wi-Fi network
    connect_wifi('TELLO-FE193A', interface_index=1)

    # Step 5: Control the drone again
    take_off_drone()

    # Step 6: Disconnect from the new Wi-Fi
    disconnect_wifi(interface_index=1)
