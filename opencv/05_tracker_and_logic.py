import math

# --- 1. Setup: Simulate Frame Detections ---
# In a real system, these would come from YOLO (like File 3) running on a video.
# Each frame has a list of bounding boxes: [x_min, y_min, x_max, y_max]
# Let's simulate two cars driving.
# Car A drives normally. Car B appears late and drives very fast (tailgating).
frames = [
    # Frame 1: Car A appears
    [[10, 10, 50, 50]],
    # Frame 2: Car A moves
    [[30, 10, 70, 50]],
    # Frame 3: Car A moves, Car B appears behind it
    [[50, 10, 90, 50], [0, 10, 40, 50]],
    # Frame 4: Car A moves normally, Car B speeds up dramatically
    [[70, 10, 110, 50], [45, 10, 85, 50]],
]

# --- 2. Tracking Memory ---
# We need a dictionary to remember where cars were in the PREVIOUS frame.
# Format: { "ID_1": (centroid_x, centroid_y) }
active_tracks = {}
next_id = 1


# --- 3. Helper Functions ---
def get_centroid(box):
    """Calculates the center point (x, y) of a bounding box."""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return x_center, y_center


def calculate_distance(point1, point2):
    """Calculates the straight-line distance between two points."""
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


# --- 4. Frame Processing Loop ---
for frame_num, detections in enumerate(frames):
    print(f"\n--- Processing Frame {frame_num + 1} ---")
    current_frame_tracks = {}

    for box in detections:
        centroid = get_centroid(box)
        assigned_id = None

        # Try to match this detection to an existing track from the previous frame
        for track_id, prev_centroid in active_tracks.items():
            dist = calculate_distance(centroid, prev_centroid)

            # If the centroid is close to where it was last frame, it's the same car
            if dist < 40:  # 40 pixels is our matching threshold
                assigned_id = track_id

                # --- RULE 1: Speeding Anomaly ---
                # If it moved a lot, but still within our threshold, flag it!
                if dist > 30:
                    print(
                        f" ALERT: Car ID {assigned_id} is SPEEDING! (Moved {dist:.1f}px)"
                    )

                # We matched it, so remove it from the pool of available tracks
                # (so two boxes don't claim the same ID)
                del active_tracks[track_id]
                break

        # If we didn't find a match, this is a new car entering the camera view
        if assigned_id is None:
            assigned_id = next_id
            print(f"✨ New object detected! Assigned ID: {assigned_id}")
            next_id += 1

        # Record the updated position for the next frame
        current_frame_tracks[assigned_id] = centroid
        print(f"Car ID {assigned_id} located at {centroid}")

    # --- RULE 2: Tailgating Anomaly (Spatial Logic) ---
    # Now that we know where all cars are in the CURRENT frame, check their distance relative to each other.
    ids = list(current_frame_tracks.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            car1_id = ids[i]
            car2_id = ids[j]
            car1_pos = current_frame_tracks[car1_id]
            car2_pos = current_frame_tracks[car2_id]

            dist_between_cars = calculate_distance(car1_pos, car2_pos)
            if dist_between_cars < 35:  # If centers are less than 35px apart
                print(
                    f" ALERT: Car {car2_id} is TAILGATING Car {car1_id}! (Distance: {dist_between_cars:.1f}px)"
                )

    # Update our memory for the next frame
    active_tracks = current_frame_tracks
