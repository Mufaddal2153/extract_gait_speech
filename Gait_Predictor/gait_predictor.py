import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd
import joblib, os

# empty dataframe with columns
# main_df = pd.DataFrame(columns=['patientType','right_shoulder_x','right_shoulder_y','right_elbow_x','right_elbow_y','right_wrist_x',
#                     'right_wrist_y','right_hip_x','right_hip_y','right_knee_x','right_knee_y','right_ankle_x','right_ankle_y',
#                     'right_heel_x','right_heel_y','right_foot_index_x','right_foot_index_y'])

main_df = pd.read_csv("Gait_Predictor/dataset_avg.csv")

def calc_distance(p11, p22):
    p1 = np.array(p11)
    p2 = np.array(p22)

    return round(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2),2) # Pythagorean theorem

#Helper functions
def quickPrint(li):
    for d in li:
        print(d)

def getX(list):
    x = []
    for d in list:
        x.append(d[0])
    return x
def getY(list):
    x = []
    for d in list:
        x.append(d[1])
    return x

def getClassList(li,patientType):
    for d in li:
        classList.append(int(patientType))

#Calculate AVG
def calculateAvg(list1):
    sum = 0
    # if len(list1) == 0:
    #     return 0
    for d in list1:
        sum += d
    return sum/len(list1)

classList = []

def extract_landmarks_features(landmarks, type):
    pass


'''
    videoPath -> .mp4 video file path
    patientType -> This is to be defined because for Unhealthy and Healthy patient
                   the arm swing distance from hip to wrist is different 
                   so in order to calculate a proper arm swing we need to define
                   initially if a patient is healthy or unhealthy
                   [1] -> Healthy
                   [0] -> Unhealthy

    showVideo -> Whether to show video frame or not
'''
def predict(videoPath,paitentType,showVideo):

    global main_df

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(videoPath)
    isFront = False
    SWING_HANDS = 0

    _F = 0
    _B = 0

    if paitentType == "1":
        _F = 0.14
        _B = 0.06
    else:
        _F = 0.06
        _B = 0.03

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontLocation = (10,60)
    fontScale              = 1
    fontColor              = (255,140,0)
    thickness              = 2
    lineType               = 2

    #joints 

    right_shoulder = []
    right_elbow = []
    right_wrist = []
    right_hip = []
    right_knee = []
    right_ankle = []
    right_heel = []
    right_foot_index = []

    left_shoulder = []
    left_elbow = []
    left_wrist = []
    left_hip = []
    left_knee = []
    left_ankle = []
    left_heel = []
    left_foot_index = []

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():    
            ret,frame = cap.read()
            print(ret, frame)
            if ret is False:
                break

            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            #Detections
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            
            #Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
                dist = calc_distance(wrist, hip)
                if dist >= _F and not isFront:
                    isFront = True
                if dist <= _B and isFront:

                    print("Extracting Features...")

                    isFront = False
                    SWING_HANDS += 1

                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                    l_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                    foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                    right_shoulder.append(shoulder)
                    right_elbow.append(elbow)
                    right_wrist.append(wrist)
                    right_hip.append(hip)
                    right_knee.append(knee)
                    right_ankle.append(ankle)
                    right_heel.append(heel)
                    right_foot_index.append(foot_index)

                    left_shoulder.append(l_shoulder)
                    left_elbow.append(l_elbow)
                    left_wrist.append(l_wrist)
                    left_hip.append(l_hip)
                    left_knee.append(l_knee)
                    left_ankle.append(l_ankle)
                    left_heel.append(l_heel)
                    left_foot_index.append(l_foot_index)


                cv2.putText(image,"Hand Swinged:"+str(SWING_HANDS), fontLocation, font, fontScale,fontColor,thickness,lineType)
            
            except:
                pass

            #Apply Joints Rendering
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS ,
                                    mp_drawing.DrawingSpec(color=(255,100,22),thickness=3,circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(255,189 ,22),thickness=3,circle_radius=3),)

            if showVideo:
                cv2.imshow("Gait Extractor" , image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # print(right_shoulder, right_elbow, right_wrist, right_hip, right_knee, right_ankle, right_heel, right_foot_index, sep="\n")
    #Extract x and y from the respective joints
    right_shoulder_x = getX(right_shoulder)
    right_shoulder_y = getY(right_shoulder)
    left_shoulder_x = getX(left_shoulder)
    left_shoulder_y = getY(left_shoulder)

    right_elbow_x = getX(right_elbow)
    right_elbow_y = getY(right_elbow)
    left_elbow_x = getX(left_elbow)
    left_elbow_y = getY(left_elbow)

    right_wrist_x = getX(right_wrist)
    right_wrist_y = getY(right_wrist)
    left_wrist_x = getX(left_wrist)
    left_wrist_y = getY(left_wrist)

    right_hip_x = getX(right_hip)
    right_hip_y = getY(right_hip)
    left_hip_x = getX(left_hip)
    left_hip_y = getY(left_hip)

    right_knee_x = getX(right_knee)
    right_knee_y = getY(right_knee)
    left_knee_x = getX(left_knee)
    left_knee_y = getY(left_knee)

    right_ankle_x = getX(right_ankle)
    right_ankle_y = getY(right_ankle)
    left_ankle_x = getX(left_ankle)
    left_ankle_y = getY(left_ankle)

    right_heel_x = getX(right_heel)
    right_heel_y = getY(right_heel)
    left_heel_x = getX(left_heel)
    left_heel_y = getY(left_heel)

    right_foot_index_x = getX(right_foot_index)
    right_foot_index_y = getY(right_foot_index)
    left_foot_index_x = getX(left_foot_index)
    left_foot_index_y = getY(left_foot_index)

    getClassList(right_shoulder,paitentType)

    #CALCULATING AVG
    shoulder_x_avg = calculateAvg(right_shoulder_x)
    shoulder_y_avg = calculateAvg(right_shoulder_y)
    left_shoulder_x_avg = calculateAvg(left_shoulder_x)
    left_shoulder_y_avg = calculateAvg(left_shoulder_y)

    elbow_x_avg = calculateAvg(right_elbow_x)
    elbow_y_avg = calculateAvg(right_elbow_y)
    left_elbow_x_avg = calculateAvg(left_elbow_x)
    left_elbow_y_avg = calculateAvg(left_elbow_y)

    wrist_x_avg = calculateAvg(right_wrist_x)
    wrist_y_avg = calculateAvg(right_wrist_y)
    left_wrist_x_avg = calculateAvg(left_wrist_x)
    left_wrist_y_avg = calculateAvg(left_wrist_y)

    hip_x_avg = calculateAvg(right_hip_x)
    hip_y_avg = calculateAvg(right_hip_y)
    left_hip_x_avg = calculateAvg(left_hip_x)
    left_hip_y_avg = calculateAvg(left_hip_y)

    knee_x_avg = calculateAvg(right_knee_x)
    knee_y_avg = calculateAvg(right_knee_y)
    left_knee_x_avg = calculateAvg(left_knee_x)
    left_knee_y_avg = calculateAvg(left_knee_y)

    ankle_x_avg = calculateAvg(right_ankle_x)
    ankle_y_avg = calculateAvg(right_ankle_y)
    left_ankle_x_avg = calculateAvg(left_ankle_x)
    left_ankle_y_avg = calculateAvg(left_ankle_y)

    heel_x_avg = calculateAvg(right_heel_x)
    heel_y_avg = calculateAvg(right_heel_y)
    left_heel_x_avg = calculateAvg(left_heel_x)
    left_heel_y_avg = calculateAvg(left_heel_y)

    foot_index_x_avg = calculateAvg(right_foot_index_x)
    foot_index_y_avg = calculateAvg(right_foot_index_y)
    left_foot_index_x_avg = calculateAvg(left_foot_index_x)
    left_foot_index_y_avg = calculateAvg(left_foot_index_y)

    #Clear every list
    right_shoulder_x.clear()
    right_shoulder_y.clear()
    left_shoulder_x.clear()
    left_shoulder_y.clear()

    right_elbow_x.clear()
    right_elbow_y.clear()
    left_elbow_x.clear()
    left_elbow_y.clear()

    right_wrist_x.clear()
    right_wrist_y.clear()
    left_wrist_x.clear()
    left_wrist_y.clear()

    right_hip_x.clear()
    right_hip_y.clear()
    left_hip_x.clear()
    left_hip_y.clear()

    right_knee_x.clear()
    right_knee_y.clear()
    left_knee_x.clear()
    left_knee_y.clear()

    right_ankle_x.clear()
    right_ankle_y.clear()
    left_ankle_x.clear()
    left_ankle_y.clear()

    right_heel_x.clear()
    right_heel_y.clear()
    left_heel_x.clear()
    left_heel_y.clear()

    right_foot_index_x.clear()
    right_foot_index_y.clear()
    left_foot_index_x.clear()
    left_foot_index_y.clear()

    #Append Averages to list
    right_shoulder_x.append(shoulder_x_avg)
    right_shoulder_y.append(shoulder_y_avg)
    left_shoulder_x.append(left_shoulder_x_avg)
    left_shoulder_y.append(left_shoulder_y_avg)

    right_elbow_x.append(elbow_x_avg)
    right_elbow_y.append(elbow_y_avg)
    left_elbow_x.append(left_elbow_x_avg)
    left_elbow_y.append(left_elbow_y_avg)

    right_wrist_x.append(wrist_x_avg)
    right_wrist_y.append(wrist_y_avg)
    left_wrist_x.append(left_wrist_x_avg)
    left_wrist_y.append(left_wrist_y_avg)

    right_hip_x.append(hip_x_avg)
    right_hip_y.append(hip_y_avg)
    left_hip_x.append(left_hip_x_avg)
    left_hip_y.append(left_hip_y_avg)

    right_knee_x.append(knee_x_avg)
    right_knee_y.append(knee_y_avg)
    left_knee_x.append(left_knee_x_avg)
    left_knee_y.append(left_knee_y_avg)

    right_ankle_x.append(ankle_x_avg)
    right_ankle_y.append(ankle_y_avg)
    left_ankle_x.append(left_ankle_x_avg)
    left_ankle_y.append(left_ankle_y_avg)

    right_heel_x.append(heel_x_avg)
    right_heel_y.append(heel_y_avg)
    left_heel_x.append(left_heel_x_avg)
    left_heel_y.append(left_heel_y_avg)

    right_foot_index_x.append(foot_index_x_avg)
    right_foot_index_y.append(foot_index_y_avg)
    left_foot_index_x.append(left_foot_index_x_avg)
    left_foot_index_y.append(left_foot_index_y_avg)

    df = pd.DataFrame(np.column_stack([paitentType,right_shoulder_x,right_shoulder_y,right_elbow_x,right_elbow_y,
                                    right_wrist_x,right_wrist_y,right_hip_x,right_hip_y,right_knee_x,right_knee_y,right_ankle_x,
                                    right_ankle_y,right_heel_x,right_heel_y,right_foot_index_x,right_foot_index_y,
                                    left_shoulder_x,left_shoulder_y,left_elbow_x,left_elbow_y,left_wrist_x,left_wrist_y,
                                    left_hip_x,left_hip_y,left_knee_x,left_knee_y,left_ankle_x,left_ankle_y,left_heel_x,
                                    left_heel_y,left_foot_index_x,left_foot_index_y]),
                    columns=['patientType','right_shoulder_x','right_shoulder_y','right_elbow_x','right_elbow_y','right_wrist_x',
                    'right_wrist_y','right_hip_x','right_hip_y','right_knee_x','right_knee_y','right_ankle_x','right_ankle_y',
                    'right_heel_x','right_heel_y','right_foot_index_x','right_foot_index_y', 'left_shoulder_x',
                    'left_shoulder_y','left_elbow_x','left_elbow_y','left_wrist_x','left_wrist_y','left_hip_x','left_hip_y',
                    'left_knee_x','left_knee_y','left_ankle_x','left_ankle_y','left_heel_x','left_heel_y','left_foot_index_x',
                    'left_foot_index_y'])

    main_df = pd.DataFrame(np.concatenate([main_df.values, df.values]), columns=main_df.columns)
    # model = joblib.load('Gait_Predictor/models/KNNModel.sav')
    # print("Predicting...")
    # res = model.predict(df)
    # res = str(res)

    # if(res == "[1.]"):
    #     return "Healthy Gait"
    # else:
    #     return "Unhealthy Gait"

####



hp_videos = os.listdir("Gait_Predictor/videos/hp")
uhp_videos = os.listdir("Gait_Predictor/videos/uhp")

# for video in hp_videos:
#     print("Processing: " + "HP "+video + "...")
#     predict("Gait_Predictor/videos/hp/"+video,"1",True)

# for video in uhp_videos:
#     print("Processing: " + "HP "+video + "...")
#     predict("Gait_Predictor/videos/uhp/"+video,"0",True)

# main_df.to_csv("Gait_Predictor/dataset_avg.csv",index=False)

predict("Gait_Predictor/videos/uhp/gait_6.mp4","0",True)
# print(len(main_df))
# print("Result: " + result)