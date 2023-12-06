import csv
import cv2
import mediapipe as mp
import math
import numpy as np
import time


def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point 
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    return (((p1[:2] - p2[:2])**2).sum())**0.5


def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width. 
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)


def eye_feature(landmarks):
    ''' Calculate the eye feature as the average of the eye aspect ratio for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Eye feature value
    '''
    return (eye_aspect_ratio(landmarks, left_eye) + \
    eye_aspect_ratio(landmarks, right_eye))/2


def pupil_circularity(landmarks, eye):
    ''' Calculate pupil circularity feature.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Pupil circularity for the eye coordinates
    '''
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
            distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
            distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
            distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
            distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
            distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
            distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
            distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)


def pupil_feature(landmarks):
    ''' Calculate the pupil feature as the average of the pupil circularity for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Pupil feature value
    '''
    return (pupil_circularity(landmarks, left_eye) + \
        pupil_circularity(landmarks, right_eye))/2


def run_face_mp(image):
    ''' Get face landmarks using the FaceMesh MediaPipe model.
    Calculate facial features using the landmarks.
    :param image: Image for which to get the face landmarks
    :return: Feature 1 (Eye), Feature 2 (Mouth), Feature 3 (Pupil), \
        Feature 4 (Combined eye and mouth feature), image with mesh drawings
    '''
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # draw face mesh over image
        for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        ear = eye_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
    else:
        ear = -1000
        puc = -1000

    return ear, puc, image


def calibrate(calib_frame_count=50):
    ''' Perform clibration. Get features for the neutral position.
    :param calib_frame_count: Image frames for which calibration is performed. Default Vale of 25.
    :return: Normalization Values for feature 1, Normalization Values for feature 2, \
        Normalization Values for feature 3, Normalization Values for feature 4
    '''
    ears = []
    pucs = []

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        ear, puc, image = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            pucs.append(puc)

        cv2.putText(image, "Calibration", (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if len(ears) >= calib_frame_count:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    ears = np.array(ears)
    pucs = np.array(pucs)

    return [ears.mean(), ears.std()], \
           [pucs.mean(), pucs.std()],


def infer(ears_norm, pucs_norm):
    ''' Perform inference.
    :param ears_norm: Normalization values for eye feature
    :param pucs_norm: Normalization values for pupil feature
    '''
    ear_main = 0
    puc_main = 0
    decay = 0.9  # use decay to smoothen the noise in feature values

    ear_arr_per_minute = []
    puc_arr_per_minute = []

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        seconds = ticker()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        ear, puc, image = run_face_mp(image)
        if ear != -1000:
            ear = (ear - ears_norm[0])/ears_norm[1]
            puc = (puc - pucs_norm[0])/pucs_norm[1]
            if ear_main == -1000:
                ear_main = ear
                puc_main = puc
            else:
                ear_main = ear_main*decay + (1-decay)*ear
                puc_main = puc_main*decay + (1-decay)*puc
        else:
            ear_main = -1000
            puc_main = -1000

        cv2.putText(image, "EAR: %.2f" %(ear_main), (int(0.02*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "PUC: %.2f" %(puc_main), (int(0.52*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if seconds >= 60.00:
            ear_average = cal_averages(ear_arr_per_minute)
            puc_average = cal_averages(puc_arr_per_minute)
            add_averages_to_csv(ear_average, puc_average)
            ear_arr_per_minute = []
            puc_arr_per_minute = []
        else:
            ear_arr_per_minute.append(ear_main)
            puc_arr_per_minute.append(puc_main)

        # add_to_csv(ear_main, puc_main)

        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty('MediaPipe FaceMesh', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    cap.release()


# calculate averages
def cal_averages(array):
    average = sum(array) / len(array)
    print(average)
    return average


# with time
def add_to_csv(ear, puc):
    fieldnames = ['EAR', 'PUC', 'TIME']
    filename = 'data.csv'

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        seconds = ticker()

        writer.writerow({'EAR': ear, 'PUC': puc, 'TIME': seconds})

        if seconds >= 60.00:
            writer.writerow({'EAR': '', 'PUC': '', 'TIME': ''})


# with average values
def add_averages_to_csv(a_ear, a_puc):
    fieldnames = ['EAR', 'PUC', 'STATUS']
    filename = 'data.csv'
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow({'EAR': a_ear, 'PUC': a_puc, 'STATUS': 'Active'})


def ticker():
    global start_time

    if start_time == 0:
        start_time = time.time()

    time_stop = time.time()
    delta_time = round(time_stop - start_time, 2)

    if delta_time >= 60.00:  # Если прошло 60 секунд, обнуляем секунды
        start_time = 0

    print(f"Прошло {delta_time} сек.")
    return delta_time


# constants and variables
start_time = 0  # Время начала работы
curr_sec = 0  # Текущее значение секунд

right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark positions
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark positions

# Declaring FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# start calibrate
print('Starting calibration. Please be in neutral state')
time.sleep(1)
ears_norm, pucs_norm = calibrate()

# start app
print('Starting main application')
time.sleep(1)
infer(ears_norm, pucs_norm)

# close if break
face_mesh.close()

