import cv2
import numpy as np
import torch
import math
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
    
# Global variables for the clicked points
clicked_points = []

# Load the pre-trained MiDaS depth model
model_type = "DPT_Large"  # You can also try "DPT_Hybrid" or "MiDaS_small"
model = DPTDepthModel(model_type=model_type)
transform = Resize(384)
normalize = NormalizeImage()

# Load the pre-trained weights
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to compute distance in real-world coordinates
def compute_real_world_distance(p1, p2, depth_map):
    # Get the depth at each point (assumed depth map is in meters)
    depth1 = depth_map[p1[1], p1[0]]  # Depth of first point
    depth2 = depth_map[p2[1], p2[0]]  # Depth of second point

    # Use the basic 3D distance formula (Euclidean distance)
    distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (depth2 - depth1) ** 2)
    return distance

# Mouse callback to store clicked points
def capture_click(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point clicked: {x}, {y}")
        if len(clicked_points) == 2:  # If two points are clicked
            print(f"Calculating distance between {clicked_points[0]} and {clicked_points[1]}")

def main():
    global clicked_points

    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✔ Webcam opened. Press ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read error")
            break
        
        # Process the frame for depth estimation
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = transform(input_frame)
        input_frame = np.expand_dims(input_frame, axis=0)
        input_frame = torch.tensor(input_frame).to(device)
        
        # Predict depth
        with torch.no_grad():
            depth_map = model(input_frame).cpu().numpy().squeeze()
        
        # Normalize depth map for visualization
        depth_map_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_vis = np.uint8(depth_map_vis)
        
        # Draw the center point on the frame
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        cv2.circle(frame, center, 6, (0, 255, 0), -1)
        
        # Draw the points and calculate the distance
        if len(clicked_points) == 2:
            p1 = clicked_points[0]
            p2 = clicked_points[1]
            distance = compute_real_world_distance(p1, p2, depth_map)
            cv2.putText(frame, f"Distance: {distance:.2f} meters", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Real-world distance: {distance:.2f} meters")
            clicked_points = []  # Reset after calculation

        # Show the frame
        cv2.imshow("Depth Estimation", frame)

        # Exit the program when the ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up the mouse callback
    cv2.namedWindow("Depth Estimation")
    cv2.setMouseCallback("Depth Estimation", capture_click)

    main()
