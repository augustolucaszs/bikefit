
import argparse
import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle
import sys
from mediapipe.framework.formats import landmark_pb2
import math


# Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Mapping joint names to PoseLandmark enum
JOINTS_MAP = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_eye": mp_pose.PoseLandmark.LEFT_EYE,
    "right_eye": mp_pose.PoseLandmark.RIGHT_EYE,
    "left_ear": mp_pose.PoseLandmark.LEFT_EAR,
    "right_ear": mp_pose.PoseLandmark.RIGHT_EAR,
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
    head_tilt = compute_head_tilt(landmarks, w, h)
    angles["Head Tilt"] = head_tilt
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

    Args:
        image: input image (modified in-place)
        angles: dict {angle_name: value}
        angle_stats: dict {angle_name: {"min": vmin, "max": vmax}}
        margin_ratio: margin from image border
        max_font_scale: max font scale allowed
        min_font_scale: min font scale allowed
        box_alpha: transparency of background box (0.0 = fully transparent, 1.0 = solid)
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

    # Estimate text width (rough, based on sample text)
    text_sample = "X. Knee Angle: 000.0 (000.0 / 000.0)"
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

    # Draw header "Angles"
    header_text = "Angles"
    cv2.putText(
        image,
        header_text,
        (int(x0 + 10), int(y0 + line_height_px // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (200, 200, 200),
        font_thickness,
    )

    # Draw angles lines
    for i, (angle_name, angle_value) in enumerate(angles.items(), start=1):
        # Get min/max from angle_stats
        min_val = angle_stats[angle_name]["min"]
        max_val = angle_stats[angle_name]["max"]

        text = f"{i}. {angle_name}: {angle_value:.1f} ({min_val:.1f} / {max_val:.1f})"

        cv2.putText(
            image,
            text,
            (int(x0 + 10), int(y0 + (i + 0.5) * line_height_px)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )


def draw_angle_arc(image, a, b, c, angle_deg, color=(255, 255, 255), radius=40):
    """
    Draw an arc to visualize the *smallest angle* at point 'b' between points a and c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate angles in degrees for vectors BA and BC
    ba = a - b
    bc = c - b

    angle_a = np.degrees(np.arctan2(ba[1], ba[0])) % 360
    angle_c = np.degrees(np.arctan2(bc[1], bc[0])) % 360

    # Compute difference between angles (both directions)
    diff_cw = (angle_c - angle_a) % 360
    diff_ccw = (angle_a - angle_c) % 360

    # Choose direction that corresponds to the smallest angle
    if diff_cw <= diff_ccw:
        start_angle = angle_a
        end_angle = angle_c
    else:
        start_angle = angle_c
        end_angle = angle_a

    # Ensure start < end (for OpenCV ellipse)
    if end_angle < start_angle:
        end_angle += 360

    # Draw arc
    center = tuple(b.astype(int))
    axes = (radius, radius)
    cv2.ellipse(image, center, axes, 0, start_angle, end_angle, color, 2)

    # Optional: put angle text outside the arc
    mid_angle = (start_angle + end_angle) / 2
    # Dynamically adjust text position based on angle and radius
    text_offset = radius + 20  # Increase offset for better visibility
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
    while True:
        cap = cv2.VideoCapture(args.file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            if results.pose_landmarks:
                overlay = frame.copy()

                # Draw solid rectangle over full image
                cv2.rectangle(
                    overlay, (0, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1
                )

                # Blend full frame
                global_alpha = 0.6

                cv2.addWeighted(
                    overlay, global_alpha, frame, 1 - global_alpha, 0, frame
                )
                # Draw landmarks and selected joints
                filtered_landmarks = create_filtered_landmarks(
                    results.pose_landmarks.landmark, args.joints
                )
                mp_draw.draw_landmarks(
                    frame, filtered_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # Compute and display angles table
                h, w, _ = frame.shape
                angles, joint_coords = compute_bikefit_angles(
                    filtered_landmarks.landmark, w, h, args
                )

                for angle_name, angle_value in angles.items():
                    # Skip Head Tilt for arc drawing
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

            cv2.imshow("Pose on Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

elif args.source == "camera":
    import time
    cap = cv2.VideoCapture(0)
    angle_stats = {}
    last_reset_time = time.time()
    reset_interval = 30  # seconds
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            overlay = frame.copy()
            # Draw solid rectangle over full image
            cv2.rectangle(
                overlay, (0, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1
            )
            # Blend full frame
            global_alpha = 0.6
            cv2.addWeighted(
                overlay, global_alpha, frame, 1 - global_alpha, 0, frame
            )
            # Draw landmarks and selected joints
            filtered_landmarks = create_filtered_landmarks(
                results.pose_landmarks.landmark, args.joints
            )
            mp_draw.draw_landmarks(
                frame, filtered_landmarks, mp_pose.POSE_CONNECTIONS
            )
            # Compute and display angles table
            h, w, _ = frame.shape
            angles, joint_coords = compute_bikefit_angles(
                filtered_landmarks.landmark, w, h, args
            )
            for angle_name, angle_value in angles.items():
                # Skip Head Tilt for arc drawing
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
                    draw_angle_arc(frame, a, b, c, angle_value)
            draw_angles_table(frame, angles, angle_stats)

        # Reset min/max every 30 seconds
        if time.time() - last_reset_time > reset_interval:
            angle_stats = {}
            last_reset_time = time.time()

        cv2.imshow("Bike Fit Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
