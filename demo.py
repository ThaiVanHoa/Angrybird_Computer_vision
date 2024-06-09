import math
import mediapipe as mp
import cv2
import pyautogui as pg
import time

# Khởi tạo các module cần thiết từ Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Khởi tạo webcam
camera = cv2.VideoCapture(0)
prev_pos = None
mouse_down = False

# Hàm tính khoảng cách giữa hai điểm
def get_distance(a, b):
    return math.dist(a, b)

# Hàm kiểm tra xem ngón cái và ngón trỏ có chụm lại không
def is_pinch(landmarks, image_shape):
    h, w, _ = image_shape
    index_tip = (landmarks.landmark[8].x * w, landmarks.landmark[8].y * h)  # Vị trí ngón trỏ
    thumb_tip = (landmarks.landmark[4].x * w, landmarks.landmark[4].y * h)  # Vị trí ngón cái
    distance = get_distance(index_tip, thumb_tip)
    return distance < 30  # Điều chỉnh ngưỡng này nếu cần thiết

# Hàm lấy vị trí của ngón trỏ
def get_index_pos(landmarks, image_shape):
    h, w, _ = image_shape
    index_tip = (landmarks.landmark[8].x * w, landmarks.landmark[8].y * h)
    return index_tip

# Hàm di chuyển con trỏ chuột đến vị trí mới
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

# Biến để tính FPS
prev_time = 0

while True:
    ret, image = camera.read()
    if not ret or image is None:
        print("Failed to capture image")
        break

    # Lật hình ảnh trước khi xử lý để khớp với thực tế
    image = cv2.flip(image, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Vẽ các điểm và kết nối trên bàn tay
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
            curr_pos = get_index_pos(landmarks, image.shape)

            # Di chuyển con trỏ chuột đến vị trí ngón trỏ
            move_mouse_pointer(curr_pos)

            # Kiểm tra xem ngón cái và ngón trỏ có chụm lại không
            if is_pinch(landmarks, image.shape):
                if not mouse_down:
                    pg.mouseDown()  # Giữ chuột khi chụm lại
                    mouse_down = True
                    print("Mouse down")  
            else:
                if mouse_down:
                    pg.mouseUp()  # Thả chuột khi tách ra
                    mouse_down = False
                    print("Mouse up")  

            prev_pos = curr_pos

    # Tính toán FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Hiển thị FPS lên màn hình
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Frame", image)
    if cv2.waitKey(1) == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
