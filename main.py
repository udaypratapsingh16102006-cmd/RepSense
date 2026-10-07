import math
import cv2
import numpy as np
import mediapipe as mp
import pyautogui

import platform

pyautogui.PAUSE = 0
set_volume = None
backend = "none"

if platform.system() == "Windows":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        speakers = AudioUtilities.GetSpeakers()
        if hasattr(speakers, "EndpointVolume"):
            volume_ctrl = speakers.EndpointVolume
        else:
            interface = speakers.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

        volume_ctrl.SetMasterVolumeLevelScalar(0.5, None)

        def set_volume(level):
            volume_ctrl.SetMasterVolumeLevelScalar(float(level), None)

        backend = "pycaw"
    except Exception as e:
        print("pycaw failed:", repr(e))
        set_volume = None

if set_volume is None:
    STEPS = 50
    for _ in range(STEPS):
        pyautogui.press("volumedown")
    current_step = 0

    def set_volume(level):
        global current_step
        target = int(round(level * STEPS))
        diff = target - current_step
        if diff != 0:
            pyautogui.press("volumeup" if diff > 0 else "volumedown",
                            presses=abs(diff), interval=0)
            current_step = target

    backend = "keys"

print("Volume backend in use:", backend)

pyautogui.FAILSAFE = False
cap = cv2.VideoCapture(0)
hands_module = mp.solutions.hands
hand_detector = hands_module.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

smoothed_volume = 0.5
vol_percent = 50

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outputs = hand_detector.process(rgb_frame)

    if outputs.multi_hand_landmarks:
        for hand, handedness in zip(outputs.multi_hand_landmarks,
                                    outputs.multi_handedness):

            label = handedness.classification[0].label
            drawing_utils.draw_landmarks(frame, hand, hands_module.HAND_CONNECTIONS)
            lm = hand.landmark

            def px(i):
                return int(lm[i].x * frame_width), int(lm[i].y * frame_height)

            thumb = px(4)
            index = px(8)

            if label == "Right":

                cv2.circle(frame, thumb, 10, (255, 0, 255), -1)
                cv2.circle(frame, index, 10, (255, 0, 255), -1)
                cv2.line(frame, thumb, index, (255, 0, 255), 3)

                dist = math.hypot(index[0] - thumb[0], index[1] - thumb[1])

                wrist, mid_base = px(0), px(9)
                hand_size = math.hypot(mid_base[0] - wrist[0], mid_base[1] - wrist[1])
                ratio = dist / max(hand_size, 1)

                target = float(np.interp(ratio, [0.15, 1.2], [0.0, 1.0]))
                smoothed_volume += (target - smoothed_volume) * 0.3
                set_volume(smoothed_volume)
                vol_percent = int(smoothed_volume * 100)

            else:

                cv2.circle(frame, index, 30, (0, 255, 255), 2)
                cv2.circle(frame, thumb, 30, (0, 255, 255), 2)
                pyautogui.moveTo(screen_width / frame_width * index[0],
                                 screen_height / frame_height * index[1])

                if math.hypot(index[0] - thumb[0], index[1] - thumb[1]) < 30:
                    pyautogui.click()
                    pyautogui.sleep(1)

    bar_y = int(np.interp(vol_percent, [0, 100], [400, 150]))
    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
    cv2.rectangle(frame, (50, bar_y), (85, 400), (0, 255, 0), -1)
    cv2.putText(frame, f"{vol_percent}%", (40, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Virtualmouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()