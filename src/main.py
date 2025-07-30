import argparse
import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle
import sys
from mediapipe.framework.formats import landmark_pb2
import math
import collections


# Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Mapping joint names to PoseLandmark enum
JOINTS_MAP = {
    # "nose": mp_pose.PoseLandmark.NOSE,
    # "left_eye": mp_pose.PoseLandmark.LEFT_EYE,
    # "right_eye": mp_pose.PoseLandmark.RIGHT_EYE,
    # "left_ear": mp_pose.PoseLandmark.LEFT_EAR,
    # "right_ear": mp_pose.PoseLandmark.RIGHT_EAR,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}

# Parse arguments
parser = argparse.ArgumentParser(description="Bike Fit Pose Estimation Tool")
parser.add_argument(
    "-source",
    type=str,
    required=True,
    choices=["image", "video", "camera"],
    help="Source type",
)
parser.add_argument("-file", type=str, default=None, help="Path to image or video file")
parser.add_argument(
    "-j",
    "--joints",
    nargs="+",
    default=["left_hip", "left_knee", "left_ankle"],
    help="Joints to display. Use 'right' for all right-side joints or 'left' for all left-side joints.",
)
parser.add_argument(
    "-camera",
    type=int,
    default=0,
    help="Camera device index (default: 0). Use 1 for the second camera, etc.",
)

args = parser.parse_args()

# Handle 'right' or 'left' shortcuts for joints
if "right" in args.joints:
    args.joints = [joint for joint in JOINTS_MAP.keys() if joint.startswith("right_")]
elif "left" in args.joints:
    args.joints = [joint for joint in JOINTS_MAP.keys() if joint.startswith("left_")]

# Print selected joints
print(f"Selected joints: {args.joints}")


# Function to draw selected joints
def draw_selected_joints(image, landmarks, selected_joints, depth_threshold=-0.5):
    """
    Draw only the selected joints on the image.
    """
    h, w, _ = image.shape
    for joint_name in selected_joints:
        if joint_name in JOINTS_MAP:
            lm = landmarks[JOINTS_MAP[joint_name].value]
            cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
            if cz < depth_threshold:  # Optional depth filtering
                cv2.circle(image, (cx, cy), 8, (0, 255, 0), cv2.FILLED)  # Draw joint
                cv2.putText(
                    image,
                    joint_name,
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )


# Function to compute head tilt angle (nose relative to shoulder line)
def compute_head_tilt(landmarks, w, h):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Shoulder center
    shoulder_cx = (left_shoulder.x + right_shoulder.x) / 2 * w
    shoulder_cy = (left_shoulder.y + right_shoulder.y) / 2 * h

    nose_cx = nose.x * w
    nose_cy = nose.y * h

    # Vector Shoulder_center → Nose
    dx = nose_cx - shoulder_cx
    dy = nose_cy - shoulder_cy

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_bikefit_angles(landmarks, w, h, args):
    angles = {}
    joint_coords = {}
    side = "LEFT" if "left" in args.joints[0] else "RIGHT"
    if len(args.joints) == 0:
        sys.exit()

    def get_coords(joint_name):
        lm = landmarks[mp_pose.PoseLandmark[joint_name].value]
        return np.array([lm.x * w, lm.y * h])

    # Knee Angle
    a_knee = get_coords("{}_HIP".format(side))
    b_knee = get_coords("{}_KNEE".format(side))
    c_knee = get_coords("{}_ANKLE".format(side))
    knee_angle = calculate_angle(a_knee, b_knee, c_knee)
    angles["Knee Angle"] = knee_angle  # Compute the opposite angle
    joint_coords["Knee Angle"] = (a_knee, b_knee, c_knee)

    # Hip Angle (shoulder, hip, and vertical line below hip)
    a_hip = get_coords("{}_SHOULDER".format(side))
    b_hip = get_coords("{}_HIP".format(side))
    # c_hip: vertical point below hip (same x, y+offset)
    vertical_offset = 100  # pixels downward; adjust as needed
    c_hip = np.array([b_hip[0], b_hip[1] + vertical_offset])
    angles["Hip Angle"] = 180 - calculate_angle(a_hip, b_hip, c_hip)
    joint_coords["Hip Angle"] = (a_hip, b_hip, c_hip)

    # Shoulder Angle
    a_shoulder = get_coords("{}_ELBOW".format(side))
    b_shoulder = get_coords("{}_SHOULDER".format(side))
    c_shoulder = get_coords("{}_HIP".format(side))
    angles["Shoulder Angle"] = calculate_angle(a_shoulder, b_shoulder, c_shoulder)
    joint_coords["Shoulder Angle"] = (a_shoulder, b_shoulder, c_shoulder)

    # Elbow Angle
    a_elbow = get_coords("{}_SHOULDER".format(side))
    b_elbow = get_coords("{}_ELBOW".format(side))
    c_elbow = get_coords("{}_WRIST".format(side))
    angles["Elbow Angle"] = calculate_angle(a_elbow, b_elbow, c_elbow)
    joint_coords["Elbow Angle"] = (a_elbow, b_elbow, c_elbow)

    # Ankle Angle (optional)
    a_ankle = get_coords("{}_KNEE".format(side))
    b_ankle = get_coords("{}_ANKLE".format(side))
    c_ankle = (
        get_coords("{}_FOOT_INDEX".format(side))
        if hasattr(mp_pose.PoseLandmark, "{}_FOOT_INDEX".format(side))
        else b_ankle
    )
    # angles["Ankle Angle"] = calculate_angle(a_ankle, b_ankle, c_ankle)
    # joint_coords["Ankle Angle"] = (a_ankle, b_ankle, c_ankle)

    # Head tilt angle (no joint_coords needed — it's computed separately)
    # head_tilt = compute_head_tilt(landmarks, w, h)
    # angles["Head Tilt"] = head_tilt
    # Do not add head tilt to joint_coords (not drawn as arc)

    return angles, joint_coords


def draw_angles_table(
    image,
    angles,
    angle_stats,  # NEW: persistent min/max per angle
    margin_ratio=0.03,
    max_font_scale=0.5,
    min_font_scale=0.3,
    box_alpha=0.6,
):
    """
    Draw an angles table with semi-transparent background.
    For each angle, shows: current (min / max)
    (Portuguese-BR labels)
    """
    h, w, _ = image.shape

    num_angles = len(angles)
    if num_angles == 0:
        return  # nothing to draw

    # Margin from edges
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    # Define desired max line height in pixels
    max_line_height_px = int(h * 0.035)  # about 3.5% of image height
    min_line_height_px = int(h * 0.025)  # about 2.5%

    # Dynamic font scale based on image height
    line_height_px = max(min_line_height_px, min(max_line_height_px, int(h / 25)))
    font_scale = line_height_px / 30.0  # empirically works well
    font_scale = max(min_font_scale, min(max_font_scale, font_scale))
    font_thickness = 1 if h < 600 else 2

    # Portuguese translation for angle names
    angle_name_pt = {
        "Knee Angle": "Angulo do Joelho",
        "Hip Angle": "Angulo do Quadril",
        "Shoulder Angle": "Angulo do Ombro",
        "Elbow Angle": "Angulo do Cotovelo",  # ,
        # "Head Tilt": "Inclinacao da Cabeca",
    }

    # Estimate text width (rough, based on sample text)
    text_sample = "X. Ângulo do Joelho: 000.0 (000.0 / 000.0)"
    ((text_width, text_height), _) = cv2.getTextSize(
        text_sample, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )

    # Box size
    box_width = text_width + 15
    box_height = (num_angles + 1) * line_height_px + 15

    # Bottom-right corner of the image
    x0 = w - margin_x - box_width
    y0 = h - margin_y - box_height

    # ---- Create semi-transparent overlay ----
    overlay = image.copy()

    # Draw solid rectangle on overlay
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + box_width, y0 + box_height),
        (50, 50, 50),  # dark gray background
        -1,
    )

    # Blend overlay with original image
    cv2.addWeighted(overlay, box_alpha, image, 1 - box_alpha, 0, image)

    # ---- Draw text on top of blended background ----

    # Draw header "Angles" in Portuguese
    header_text = "Ângulos"
    cv2.putText(
        image,
        header_text,
        (int(x0 + 10), int(y0 + line_height_px // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (200, 200, 200),
        font_thickness,
    )

    # Draw angles lines in Portuguese
    for i, (angle_name, angle_value) in enumerate(angles.items(), start=1):
        # Get min/max from angle_stats
        min_val = angle_stats[angle_name]["min"]
        max_val = angle_stats[angle_name]["max"]

        # Use Portuguese label if available
        label = angle_name_pt.get(angle_name, angle_name)
        text = f"{i}. {label}: {angle_value:.1f} ({min_val:.1f} / {max_val:.1f})"

        cv2.putText(
            image,
            text,
            (int(x0 + 10), int(y0 + (i + 0.5) * line_height_px)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10, gap_length=10):
    """Draw a dashed line between pt1 and pt2."""
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    dashes = int(dist // (dash_length + gap_length))
    if dashes == 0:
        cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        return
    for i in range(dashes + 1):
        start_frac = i * (dash_length + gap_length) / dist
        end_frac = min((i * (dash_length + gap_length) + dash_length) / dist, 1.0)
        start = (
            int(pt1[0] + (pt2[0] - pt1[0]) * start_frac),
            int(pt1[1] + (pt2[1] - pt1[1]) * start_frac),
        )
        end = (
            int(pt1[0] + (pt2[0] - pt1[0]) * end_frac),
            int(pt1[1] + (pt2[1] - pt1[1]) * end_frac),
        )
        cv2.line(img, start, end, color, thickness, lineType=cv2.LINE_AA)


def draw_angle_arc(image, a, b, c, angle_deg, color=(255, 255, 255), radius=40):
    """
    Draw an arc to visualize the *smallest angle* at point 'b' between points a and c.
    For the hip angle, draw a dashed vertical line above the hip joint and draw the arc between the shoulder-hip and vertical-up line.
    For all, align the angle value at the middle of the arc (arc bisector), always on the same side as the arc.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Draw dashed line ONLY for hip angle: b is a hip joint and c is directly above or below b (vertical)
    # We'll check if b is left_hip or right_hip, and c is vertically aligned with b (x nearly equal), and c is above or below b by a significant offset
    hip_joint_indices = [
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
    ]
    is_hip_joint = False
    try:
        # Find which joint b is (by comparing to landmark positions)
        for joint_name, enum in JOINTS_MAP.items():
            if enum.value in hip_joint_indices:
                # If b matches the landmark for this hip
                if np.allclose(
                    b,
                    [
                        landmark_pb2.NormalizedLandmark().x,
                        landmark_pb2.NormalizedLandmark().y,
                    ],
                    atol=1e-3,
                ):
                    is_hip_joint = True
    except Exception:
        # Fallback: check if b is likely a hip by y position (not robust, but avoids crash)
        is_hip_joint = False
    # Instead, let's use a more robust geometric check:
    # If c is vertically aligned with b (x nearly equal), and c is above or below b by at least 50 pixels
    is_vertical = np.abs(c[0] - b[0]) < 2 and np.abs(c[1] - b[1]) > 30
    is_hip_angle = is_vertical and (b[1] < c[1] or b[1] > c[1])

    if is_hip_angle:
        # Draw dashed vertical line above the hip joint (always up)
        vertical_length = np.abs(c[1] - b[1])
        top = b - np.array([0, vertical_length])
        overlay = image.copy()
        draw_dashed_line(
            overlay, b, top, (0, 255, 255), thickness=2, dash_length=12, gap_length=10
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        # Draw arc between shoulder-hip and vertical-up
        ba = a - b
        bt = top - b
        angle_a = np.degrees(np.arctan2(ba[1], ba[0])) % 360
        angle_t = np.degrees(np.arctan2(bt[1], bt[0])) % 360
        diff_cw = (angle_t - angle_a) % 360
        diff_ccw = (angle_a - angle_t) % 360
        if diff_cw <= diff_ccw:
            start_angle = angle_a
            end_angle = angle_t
        else:
            start_angle = angle_t
            end_angle = angle_a
        if end_angle < start_angle:
            end_angle += 360
        center = tuple(b.astype(int))
        axes = (radius, radius)
        cv2.ellipse(image, center, axes, 0, start_angle, end_angle, color, 2)
        # Place angle value at the middle of the arc (arc bisector, same side as arc)
        mid_angle = (start_angle + end_angle) / 2
        text_offset = radius + 20
        text_x = int(b[0] + text_offset * np.cos(np.radians(mid_angle)))
        text_y = int(b[1] + text_offset * np.sin(np.radians(mid_angle)))
        cv2.putText(
            image,
            f"{int(angle_deg)}",
            (int(text_x), int(text_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        return

    # Calculate angles in degrees for vectors BA and BC
    ba = a - b
    bc = c - b
    angle_a = np.degrees(np.arctan2(ba[1], ba[0])) % 360
    angle_c = np.degrees(np.arctan2(bc[1], bc[0])) % 360
    diff_cw = (angle_c - angle_a) % 360
    diff_ccw = (angle_a - angle_c) % 360
    if diff_cw <= diff_ccw:
        start_angle = angle_a
        end_angle = angle_c
    else:
        start_angle = angle_c
        end_angle = angle_a
    if end_angle < start_angle:
        end_angle += 360
    center = tuple(b.astype(int))
    axes = (radius, radius)
    cv2.ellipse(image, center, axes, 0, start_angle, end_angle, color, 2)
    # Place angle value at the middle of the arc (arc bisector, same side as arc)
    mid_angle = (start_angle + end_angle) / 2
    text_offset = radius + 20
    text_x = int(b[0] + text_offset * np.cos(np.radians(mid_angle)))
    text_y = int(b[1] + text_offset * np.sin(np.radians(mid_angle)))
    cv2.putText(
        image,
        f"{int(angle_deg)}",
        (int(text_x), int(text_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def create_filtered_landmarks(original_landmarks, selected_joints):
    fake_landmarks = []

    for i, lm in enumerate(original_landmarks):
        # Get joint name for this index
        try:
            joint_name = next(
                name for name, enum in JOINTS_MAP.items() if enum.value == i
            )
        except StopIteration:
            joint_name = None

        if joint_name in selected_joints:
            # Keep original landmark
            new_lm = landmark_pb2.NormalizedLandmark(
                x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
            )
        else:
            # Move offscreen and make invisible
            new_lm = landmark_pb2.NormalizedLandmark(
                x=-10.0, y=-10.0, z=lm.z, visibility=0.0
            )

        fake_landmarks.append(new_lm)

    # Wrap into a NormalizedLandmarkList
    filtered_landmark_list = landmark_pb2.NormalizedLandmarkList(
        landmark=fake_landmarks
    )
    return filtered_landmark_list


def get_crank_angle(landmarks, w, h, side):
    """
    Returns the angle (in degrees, 0-360) of the ankle relative to the hip, with respect to vertical down.
    Used for crank rotation detection.
    """
    hip = landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value]
    ankle = landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value]
    x_hip, y_hip = hip.x * w, hip.y * h
    x_ankle, y_ankle = ankle.x * w, ankle.y * h
    dx = x_ankle - x_hip
    dy = y_ankle - y_hip
    # Angle from vertical down (0 at 6 o'clock, increases CCW)
    angle = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    return angle


# Helper: detect if cyclist is seated (hip y below threshold and stable)
def is_cyclist_seated(
    landmarks, h, side, seated_y_ratio=0.65, stable_seconds=5, fps=30
):
    """
    Returns True if the hip y position is below a threshold (seated) and stable for N frames (N = stable_seconds * fps).
    """
    stable_frames = int(stable_seconds * fps)
    if not hasattr(is_cyclist_seated, "history"):
        is_cyclist_seated.history = []
    hip = landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value]
    y_hip = hip.y * h
    is_cyclist_seated.history.append(y_hip)
    if len(is_cyclist_seated.history) > stable_frames:
        is_cyclist_seated.history.pop(0)
    threshold = h * seated_y_ratio
    return len(is_cyclist_seated.history) == stable_frames and all(
        y < threshold for y in is_cyclist_seated.history
    )


# Helper: detect wheel centers using HoughCircles (fallback to manual click if not found)
def detect_wheel_centers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=40,
        maxRadius=200,
    )
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            centers.append((i[0], i[1]))
        # If more than 2, pick the two farthest apart
        if len(centers) > 2:
            max_dist = 0
            best_pair = (centers[0], centers[1])
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                    if d > max_dist:
                        max_dist = d
                        best_pair = (centers[i], centers[j])
            centers = list(best_pair)
    return centers


def detect_wheel_centers_with_ankle_and_hip(image, landmarks, h, w):
    """
    Improved wheel center detection: Use ankle and hip joints to restrict search area.
    Only consider circles (wheels) that are below the hip joint and near the ankle joints.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=40,
        maxRadius=200,
    )
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Get hip and ankle y positions
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h
        left_ankle = (
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h,
        )
        right_ankle = (
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h,
        )
        for i in circles[0, :]:
            cx, cy = i[0], i[1]
            # Only consider circles below both hips
            if cy > left_hip_y and cy > right_hip_y:
                # Must be within 200px of at least one ankle
                if (
                    np.linalg.norm(np.array([cx, cy]) - np.array(left_ankle)) < 200
                    or np.linalg.norm(np.array([cx, cy]) - np.array(right_ankle)) < 200
                ):
                    centers.append((cx, cy))
        # If more than 2, pick the two farthest apart
        if len(centers) > 2:
            max_dist = 0
            best_pair = (centers[0], centers[1])
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                    if d > max_dist:
                        max_dist = d
                        best_pair = (centers[i], centers[j])
            centers = list(best_pair)
    return centers


# Helper: get crank center as midpoint between two wheel centers
def get_crank_center(wheel_centers):
    if len(wheel_centers) == 2:
        # Force crank center to be horizontally between the wheels (average x, average y of the two wheels)
        x0, y0 = wheel_centers[0]
        x1, y1 = wheel_centers[1]
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        # Set cy to be exactly between the two wheels (horizontal line)
        cy = (y0 + y1) // 2
        return (cx, cy)
    return None


# Process sources
if args.source == "image":
    if args.file is None:
        raise ValueError("You must provide -file for image source.")

    image = cv2.imread(args.file)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        draw_selected_joints(image, results.pose_landmarks.landmark, args.joints)

        # Compute and display angles table
        h, w, _ = image.shape
        angles, joint_coords = compute_bikefit_angles(
            results.pose_landmarks.landmark, w, h, args
        )

        draw_angles_table(image, angles)

        for angle_name, angle_value in angles.items():
            # Skip Head Tilt for arc drawing
            if angle_name == "Head Tilt":
                continue
            a, b, c = joint_coords[angle_name]
            draw_angle_arc(image, a, b, c, angle_value)

    cv2.imshow("Pose on Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif args.source == "video":
    if args.file is None:
        raise ValueError("You must provide -file for video source.")

    angle_stats = {}  # Initialize empty at first
    cap = cv2.VideoCapture(args.file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    reset_interval = int(fps * 10)  # Reset every 10 seconds

    # Crank rotation detection state
    crank_angles = collections.deque(maxlen=30)  # store last N angles
    crank_rotations = 0
    overlays_enabled = False
    last_crank_angle = None
    side = "LEFT" if "left" in args.joints[0] else "RIGHT"

    # Seated detection state
    seated_detected = False
    # Wheel/crank detection state
    wheel_centers = None
    crank_center = None
    ankle_passes = {"left": 0, "right": 0}
    last_ankle_side = None
    ankle_close = False
    pass_radius = 60  # pixels, adjust as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % reset_interval == 0:
            angle_stats = {}
            print("Reset min/max stats.")

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            h, w, _ = frame.shape
            # --- Seated detection ---
            if not seated_detected:
                if is_cyclist_seated(
                    results.pose_landmarks.landmark, h, side, stable_seconds=5, fps=fps
                ):
                    seated_detected = True
            if not seated_detected:
                msg = "Aguardando ciclista sentar..."
                cv2.putText(
                    frame,
                    msg,
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3,
                )
                cv2.imshow("Pose on Video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            # --- End seated detection ---

            # --- Wheel/crank detection ---
            if wheel_centers is None:
                wheel_centers = detect_wheel_centers_with_ankle_and_hip(
                    frame, results.pose_landmarks.landmark, h, w
                )
                if len(wheel_centers) != 2:
                    msg = (
                        "Detectando rodas... (Ajuste iluminação ou clique manualmente)"
                    )
                    cv2.putText(
                        frame,
                        msg,
                        (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("Pose on Video", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                crank_center = get_crank_center(wheel_centers)
            # Draw wheel centers and crank center
            for c in wheel_centers:
                cv2.circle(frame, c, 12, (0, 0, 255), 3)
            if crank_center:
                cv2.circle(frame, crank_center, 10, (0, 255, 255), 3)
            # --- End wheel/crank detection ---

            # --- Ankle pass detection ---
            # Use left and right ankle
            left_ankle = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_ANKLE.value
            ]
            right_ankle = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ]
            left_ankle_pt = (int(left_ankle.x * w), int(left_ankle.y * h))
            right_ankle_pt = (int(right_ankle.x * w), int(right_ankle.y * h))
            # Check if either ankle is close to crank center
            for side_name, pt in zip(
                ["left", "right"], [left_ankle_pt, right_ankle_pt]
            ):
                dist = np.linalg.norm(np.array(pt) - np.array(crank_center))
                if dist < pass_radius:
                    if last_ankle_side != side_name:
                        ankle_passes[side_name] += 1
                        last_ankle_side = side_name
                        print(f"Ankle {side_name} pass: {ankle_passes[side_name]}")
            # Only enable overlays after 3 passes per side
            overlays_enabled = all(v >= 3 for v in ankle_passes.values())
            # --- End ankle pass detection ---

            overlay = frame.copy()
            cv2.rectangle(
                overlay, (0, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1
            )
            global_alpha = 0.6
            cv2.addWeighted(overlay, global_alpha, frame, 1 - global_alpha, 0, frame)
            filtered_landmarks = create_filtered_landmarks(
                results.pose_landmarks.landmark, args.joints
            )
            mp_draw.draw_landmarks(frame, filtered_landmarks, mp_pose.POSE_CONNECTIONS)

            if overlays_enabled:
                angles, joint_coords = compute_bikefit_angles(
                    filtered_landmarks.landmark, w, h, args
                )
                for angle_name, angle_value in angles.items():
                    if angle_name not in angle_stats:
                        angle_stats[angle_name] = {
                            "min": angle_value,
                            "max": angle_value,
                        }
                    else:
                        angle_stats[angle_name]["min"] = min(
                            angle_stats[angle_name]["min"], angle_value
                        )
                        angle_stats[angle_name]["max"] = max(
                            angle_stats[angle_name]["max"], angle_value
                        )
                    if angle_name == "Head Tilt":
                        continue
                    a, b, c = joint_coords[angle_name]
                    draw_angle_arc(frame, a, b, c, angle_value)
                draw_angles_table(frame, angles, angle_stats)
            else:
                # Show waiting message
                msg = f"Pedale {3 - crank_rotations} volta(s) completa(s) para iniciar a análise..."
                cv2.putText(
                    frame,
                    msg,
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3,
                )

        cv2.imshow("Pose on Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

elif args.source == "camera":
    import time
    import ctypes
    import collections

    # Get screen resolution (Windows only)
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    cap = cv2.VideoCapture(0)
    angle_stats = {}
    last_reset_time = time.time()
    reset_interval = 30  # seconds

    # Crank rotation detection state
    crank_angles = collections.deque(maxlen=30)
    crank_rotations = 0
    overlays_enabled = False
    last_crank_angle = None
    side = "LEFT" if "left" in args.joints[0] else "RIGHT"

    # Seated detection state
    seated_detected = False
    # Wheel/crank detection state
    wheel_centers = None
    crank_center = None
    ankle_passes = {"left": 0, "right": 0}
    last_ankle_side = None
    ankle_close = False
    pass_radius = 60  # pixels, adjust as needed

    fps = 30  # Camera FPS (approximate, or you can measure)

    # Get camera frame size
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        sys.exit(1)
    cam_h, cam_w = frame.shape[:2]
    cam_aspect = cam_w / cam_h
    screen_aspect = screen_width / screen_height
    window_name = "Bike Fit Pose"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        if not ret:
            break

        # Calculate new size to fit screen while keeping aspect ratio
        if cam_aspect > screen_aspect:
            # Fit width
            new_w = screen_width
            new_h = int(screen_width / cam_aspect)
        else:
            # Fit height
            new_h = screen_height
            new_w = int(screen_height * cam_aspect)
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Create black background and center the frame
        display_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        y_offset = (screen_height - new_h) // 2
        x_offset = (screen_width - new_w) // 2
        display_frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            resized_frame
        )

        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            h, w, _ = display_frame.shape
            # --- Seated detection ---
            if not seated_detected:
                if is_cyclist_seated(
                    results.pose_landmarks.landmark, h, side, stable_seconds=5, fps=fps
                ):
                    seated_detected = True
            # --- End seated detection ---
            if not seated_detected:
                msg = "Aguardando ciclista sentar..."
                cv2.putText(
                    display_frame,
                    msg,
                    (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    4,
                )
                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                ret, frame = cap.read()
                continue

            # --- Crank rotation detection ---
            angle = get_crank_angle(results.pose_landmarks.landmark, w, h, side)
            if last_crank_angle is not None:
                if last_crank_angle > 300 and angle < 60:
                    crank_rotations += 1
            last_crank_angle = angle
            crank_angles.append(angle)
            if crank_rotations >= 3:
                overlays_enabled = True
            # --- End crank rotation detection ---

            overlay = display_frame.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (display_frame.shape[1], display_frame.shape[0]),
                (50, 50, 50),
                -1,
            )
            global_alpha = 0.6
            cv2.addWeighted(
                overlay, global_alpha, display_frame, 1 - global_alpha, 0, display_frame
            )
            filtered_landmarks = create_filtered_landmarks(
                results.pose_landmarks.landmark, args.joints
            )
            mp_draw.draw_landmarks(
                display_frame, filtered_landmarks, mp_pose.POSE_CONNECTIONS
            )
            if overlays_enabled:
                angles, joint_coords = compute_bikefit_angles(
                    filtered_landmarks.landmark, w, h, args
                )
                for angle_name, angle_value in angles.items():
                    if angle_name not in angle_stats:
                        angle_stats[angle_name] = {
                            "min": angle_value,
                            "max": angle_value,
                        }
                    else:
                        angle_stats[angle_name]["min"] = min(
                            angle_stats[angle_name]["min"], angle_value
                        )
                        angle_stats[angle_name]["max"] = max(
                            angle_stats[angle_name]["max"], angle_value
                        )
                    if angle_name == "Head Tilt":
                        continue
                    a, b, c = joint_coords[angle_name]
                    if not math.isnan(angle_value):
                        draw_angle_arc(display_frame, a, b, c, angle_value)
                draw_angles_table(display_frame, angles, angle_stats)
            else:
                msg = f"Pedale {3 - crank_rotations} volta(s) completa(s) para iniciar a análise..."
                cv2.putText(
                    display_frame,
                    msg,
                    (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    4,
                )
        # Reset min/max every 30 seconds
        if time.time() - last_reset_time > reset_interval:
            angle_stats = {}
            last_reset_time = time.time()

        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
