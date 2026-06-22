import math

# Simulate frame-by-frame bounding box detections
# In a production pipeline, these spatial coordinates are output by an object detector (e.g., YOLO).
# Bounding box format: [x_min, y_min, x_max, y_max]
# Simulating two objects moving: Object A moves normally; Object B enters later at high speed.
frames = [
    # Frame 1: Object A enters frame
    [[10, 10, 50, 50]],
    # Frame 2: Object A shifts location
    [[30, 10, 70, 50]],
    # Frame 3: Object A shifts; Object B enters from behind
    [[50, 10, 90, 50], [0, 10, 40, 50]],
    # Frame 4: Object A shifts; Object B shifts rapidly (high velocity)
    [[70, 10, 110, 50], [45, 10, 85, 50]],
]

# Track tracking history state
# Stores centroids from the previous frame: { object_id: (centroid_x, centroid_y) }
active_tracks = {}
next_id = 1


def get_centroid(box):
    """Computes the center point (x, y) of a bounding box."""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return x_center, y_center


def calculate_distance(point1, point2):
    """Computes the Euclidean distance between two points."""
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


# Process detections frame by frame
for frame_num, detections in enumerate(frames):
    print(f"\n--- Processing Frame {frame_num + 1} ---")
    current_frame_tracks = {}

    for box in detections:
        centroid = get_centroid(box)
        assigned_id = None

        # Attempt to match current detection with a track from the previous frame
        for track_id, prev_centroid in active_tracks.items():
            dist = calculate_distance(centroid, prev_centroid)

            # Match criteria: centroid distance falls below a specific threshold (e.g., 40 pixels)
            if dist < 40:
                assigned_id = track_id

                # Velocity Check: Flag excessive frame-to-frame movement (e.g., dist > 30 px)
                if dist > 30:
                    print(
                        f" ALERT: Vehicle ID {assigned_id} is speeding (delta: {dist:.1f}px)"
                    )

                # Remove matched ID from active pool to avoid double-assignment
                del active_tracks[track_id]
                break

        # Register a new track if no matching previous track is found
        if assigned_id is None:
            assigned_id = next_id
            print(f"Registered new track. Assigned ID: {assigned_id}")
            next_id += 1

        # Save current position
        current_frame_tracks[assigned_id] = centroid
        print(f"Vehicle ID {assigned_id} position: {centroid}")

    # Tailgating Detection (Proximity Alert)
    # Checks Euclidean distance between all tracked objects in the current frame
    ids = list(current_frame_tracks.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            car1_id = ids[i]
            car2_id = ids[j]
            car1_pos = current_frame_tracks[car1_id]
            car2_pos = current_frame_tracks[car2_id]

            dist_between_cars = calculate_distance(car1_pos, car2_pos)
            if dist_between_cars < 35:
                print(
                    f" ALERT: Vehicle {car2_id} tailgating vehicle {car1_id} (gap: {dist_between_cars:.1f}px)"
                )

    # Persist current tracks for next frame evaluation
    active_tracks = current_frame_tracks
