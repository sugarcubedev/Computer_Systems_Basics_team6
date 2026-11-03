import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

image = cv2.imread("hand.png")
if image is None:
    raise FileNotFoundError("hand.jpg 파일을 확인하세요.")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)

with HandLandmarker.create_from_options(options) as detector:
    result = detector.detect(mp_image)

    print("검출된 손:", len(result.hand_landmarks))

    for hand in result.hand_landmarks:
        h, w = image.shape[:2]
        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

cv2.imwrite("hand_result.jpg", image)
print("결과 저장: hand_result.jpg")
