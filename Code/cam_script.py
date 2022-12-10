import cv2
import numpy as np
import face_recognition


def eye_extracted(frame):

    facial_feature_list = face_recognition.face_landmarks(frame)

    try:
        eye_feature_left = facial_feature_list[0]['left_eye']
        eye_feature_right = facial_feature_list[0]['right_eye']
    except Exception as e:
        print("Exception: "+str(e))
        return

    x_max = max([coordinate[0] for coordinate in eye_feature_left])
    x_min = min([coordinate[0] for coordinate in eye_feature_left])
    y_max = max([coordinate[1] for coordinate in eye_feature_left])
    y_min = min([coordinate[1] for coordinate in eye_feature_left])

    x_range = x_max - x_min
    y_range = y_max - y_min

    # adding 50% padding to axis with larger ranger

    if x_range > y_range:
        right = round(.5*x_range) + x_max
        left = x_min - round(.5*x_range)
        bottom = round((((right-left) - y_range))/2) + y_max
        top = y_min - round((((right-left) - y_range))/2)
    else:
        bottom = round(.5*y_range) + y_max
        top = y_min - round(.5*y_range)
        right = round((((bottom-top) - x_range))/2) + x_max
        left = x_min - round((((bottom-top) - x_range))/2)

    cropped = frame[top:(bottom + 1), left:(right + 1)]
    cropped = cv2.resize(cropped, (80, 80))
    return cropped.reshape(-1, 80, 80, 3)


def main_function():
    vid_frame = cv2.VideoCapture(0)
    print(vid_frame.get(cv2.CAP_PROP_FPS))

    if not vid_frame.isOpened():
        print("Error opening Webcam")

    while True:
        ret, frame = vid_frame.read()
        img_predict = eye_extracted(frame)

        cv2.imshow('Drowsiness Detection', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    vid_frame.release()
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main_function()


vid = cv2.VideoCapture(0)

while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
