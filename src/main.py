import cv2
import mediapipe as mp
import numpy as np
import math
import warnings
import screen_brightness_control as sbc

warnings.filterwarnings("ignore")


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

hands = mp_hands.Hands(max_num_hands=1)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

mp_draw = mp.solutions.drawing_utils


from pycaw.pycaw import AudioUtilities

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]


cap = cv2.VideoCapture(0)

is_muted = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   
    face_results = face_detection.process(imgRGB)


    if not face_results.detections:
        volume.SetMute(1, None)

        cv2.putText(img, "NO FACE - MUTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

    else:
        cv2.putText(img, "Face Detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

     
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                h, w, c = img.shape

                thumb = handLms.landmark[4]
                index = handLms.landmark[8]
                middle = handLms.landmark[12]
                pinky = handLms.landmark[20]

                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)
                x3, y3 = int(middle.x * w), int(middle.y * h)
                x4, y4 = int(pinky.x * w), int(pinky.y * h)

                cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x4, y4), 8, (255, 0, 255), cv2.FILLED)

                vol_length = math.hypot(x2 - x1, y2 - y1)
                bright_length = math.hypot(x3 - x2, y3 - y2)
                mute_length = math.hypot(x4 - x1, y4 - y1)

                cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.line(img, (x2,y2), (x3,y3), (0,255,255), 2)
                cv2.line(img, (x1,y1), (x4,y4), (0,0,255), 2)

                if mute_length < 30:
                    volume.SetMute(1, None)
                    is_muted = True

                    cv2.putText(img, "MUTED", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 3)

                elif mute_length > 40 and is_muted:
                    volume.SetMute(0, None)
                    is_muted = False

                elif bright_length < 200 and vol_length < 50:

                    brightness = np.interp(bright_length, [30, 200], [0, 100])
                    sbc.set_brightness(int(brightness))

                    cv2.putText(img, "Brightness Mode", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                else:
                    vol = np.interp(vol_length, [30, 200], [min_vol, max_vol])
                    volume.SetMasterVolumeLevel(vol, None)

                    vol_percent = int(np.interp(vol_length, [30, 200], [0, 100]))

                    cv2.putText(img, f"Volume: {vol_percent}%", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("IntelliGest - AI Control System", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()