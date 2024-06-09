import math
import mediapipe as mp
import cv2
import pyautogui as pg

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
camera = cv2.VideoCapture(0)
prev_pos = None
mouse_down = False

def get_distance(a, b):
    return math.dist(a, b)

def is_pinch(landmarks, image_shape):
    h, w, _ = image_shape
    index_tip = (landmarks.landmark[8].x * w, landmarks.landmark[8].y * h)
    thumb_tip = (landmarks.landmark[4].x * w, landmarks.landmark[4].y * h)
    distance = get_distance(index_tip, thumb_tip)
    return distance < 30  # Điều chỉnh ngưỡng này nếu cần thiết

def get_index_pos(landmarks, image_shape):
    h, w, _ = image_shape
    index_tip = (landmarks.landmark[8].x * w, landmarks.landmark[8].y * h)
    return index_tip

def move_mouse_pointer(curr_pos):
    try:
        screen_width, screen_height = pg.size()
        cam_width, cam_height = 640, 480  # Kích thước khung hình từ webcam
        screen_x = int(curr_pos[0] * screen_width / cam_width)
        screen_y = int(curr_pos[1] * screen_height / cam_height)
        pg.moveTo(screen_x, screen_y)
        print(f"Mouse moved to: ({screen_x}, {screen_y})")  # In trạng thái di chuyển chuột
    except Exception as e:
        print(f"Failed to move mouse: {e}")

while True:
    ret, image = camera.read()
    if not ret or image is None:
        print("Failed to capture image")
        break

    # Lật hình ảnh trước khi xử lý
    image = cv2.flip(image, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
            curr_pos = get_index_pos(landmarks, image.shape)

            move_mouse_pointer(curr_pos)  # Di chuyển chuột đến vị trí của ngón trỏ

            if is_pinch(landmarks, image.shape):
                if not mouse_down:
                    pg.mouseDown()
                    mouse_down = True
                    print("Mouse down")  # In trạng thái giữ chuột
            else:
                if mouse_down:
                    pg.mouseUp()
                    mouse_down = False
                    print("Mouse up")  # In trạng thái thả chuột

            prev_pos = curr_pos

    cv2.imshow("Frame", image)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
