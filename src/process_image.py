import cv2
import mediapipe as mp
from utils import calculate_angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Load image
image = cv2.imread(
    r"C:\Users\u60696\Downloads\Bikefit\istockphoto-497021621-612x612.jpg"
)
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)

# Draw landmarks
if results.pose_landmarks:
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

cv2.imshow("Pose on Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
