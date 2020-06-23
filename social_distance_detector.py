#import packages
import cx_Oracle
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
from datetime import date,timedelta,datetime
from yoloface_master.detect_mask_image import detect_mask
#db connect
con = cx_Oracle.connect('SYSTEM/1234@127.0.0.1/XE')
cur = con.cursor()
#cur.execute("""TRUNCATE TABLE social_distance""")

#Load coco class
labelsPath = os.path.sep.join([config.MODEL_PATH,'coco.names'])
LABELS = open(labelsPath).read().strip().split('\n')


#derive path for yolo weight and model config
weightsPath = os.path.sep.join([config.MODEL_PATH,'yolov3.weights'])
configPath = os.path.sep.join([config.MODEL_PATH,'yolov3.cfg'])

#load yolo object detector
#print('loading YOLO from disk...')
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    #print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#determine only the output layer names that need from yolo
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
#print("accessing video stream...")
names = ['Chaepauk.mp4','Egmore.mp4','Parryscorner.mp4','Pondy Bazaar.mp4','Triplicane.mp4']
titles = ['Chaepauk','Egmore','Parryscorner','Pondy Bazaar','Triplicane']
#names = ['Chaepauk.mp4']
#titles = ['Chaepauk']
vs = [cv2.VideoCapture('videos/'+i) for i in names]
#loop over frames
data = []
#total_frames = 0
frame = [None]*len(names)
ret = [None]*len(names)
total_frames = [0]*len(names)
start = datetime.now()
x = date.today()
day = (x.strftime("%d")+'-'+x.strftime("%b").upper()+'-'+x.strftime("%y"))
flag = 0
initial = [0]*len(names)
current = [0]*len(names)
while True:
    for index,video in enumerate(vs):
        Mask_with_MSD = 0
        Mask_without_MSD = 0
        Nomask_with_MSD = 0
        Nomask_without_MSD = 0
        total_frames[index] += 1
        ret[index],frame[index] = video.read()
        if total_frames[index]%10:
                continue
        if not ret[index]:
            flag = 1
            break

        #frame[index] = imutils.resize(frame[index],width = 700)

        if frame[index] is None:
            print('[i] ==> Done.......!!!')
        mask_detection = detect_mask(frame[index])

        results = detect_people(frame[index],net,ln,personIdx = LABELS.index('person'))

        #Considering each person in the frame.
        """for (i, (prob, bbox, centroid)) in enumerate(results):
            res = detect_mask(ROI)
            mask_detection.append(res)"""

        accuracy = [0]*len(results)
        label = ['None'] * len(results)
        new_label = []

        person_centroids = np.array([r[2] for r in results])
        face_centroids = np.array([r[2] for r in mask_detection])
        #print(mask_detection)
        for (acc,box,(fcX,fcY),lab) in mask_detection:
            ind = 0
            i = 0
            min = 10000
            for (pcX,pcY) in person_centroids:
                if(abs(pcX-fcX) < min):
                    ind = i
                    min = abs(pcX-fcX)
                i = i+1
            label[ind] = lab
            accuracy[ind] = acc



        for i in range(0,len(label)):
            for j in range(i+1,len(label)):
                if(label[i] == 'Mask' and label[j] == 'Mask'):
                    new_label.append(2)
                elif((label[i] == 'Mask' and label[j] != 'Mask') or (label[i] != 'Mask' and label[j] == 'Mask')):
                    new_label.append(1)
                elif(label[i] == 'No Mask' and label[j] == 'No Mask'):
                    new_label.append(0)
                else:
                    new_label.append(-1)



        violate = set()
        index1 = 0
        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids,metric = 'euclidean')
            for i in range(0,D.shape[0]):
                for j in range(i+1,D.shape[1]):
                    if(new_label[index1] == 2):
                        min_distance = config.MIN_DISTANCE2
                    elif(new_label[index1] == 1):
                        min_distance = config.MIN_DISTANCE1
                    elif(new_label[index1] == 0):
                        min_distance = config.MIN_DISTANCE0
                    else:
                        min_distance = config.MIN_DISTANCE
                    if D[i,j] < min_distance:
                        violate.add(i)
                        violate.add(j)
                    index1 = index1+1
        #Loop over results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            acc = accuracy[i]
            lab = label[i]
            #color = (0,255,0)
            if i in violate:
                if(label[i] == 'Mask'):
                    Mask_without_MSD += 1
                    color = (0,255,255)
                else:
                    Nomask_without_MSD += 1
                    color = (0, 0, 255)
            else:
                if(label[i] == 'Mask'):
                    Mask_with_MSD += 1
                    color = (0,255,0)
                else:
                    Nomask_with_MSD += 1
                    color = (0,255,255)

            cv2.putText(frame[index],lab,(startX, startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
            cv2.rectangle(frame[index], (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame[index], (cX, cY), 5, color, 1)
        end = datetime.now()
        fps = 0
        total_time =  abs(start.second - end.second)
        if total_time:
                fps = total_frames[index]/total_time
        cv2.imshow(titles[index],frame[index])
        current[index] = datetime.now().second
        if int(current[index]) < 15:
                time_st = '5AM-12PM'
        elif int(current[index]) < 30:
                time_st = '12PM-4PM'
        elif int(current[index]) < 45:
                time_st = '4PM-8PM'
        else:
                time_st = '8PM-5AM'
        #timeset = str(current.hour)+":"+str(current.minute)+":"+str(current.second)
        title = titles[index]
        total = len(results)
        #print(Mask_without_MSD,Nomask_without_MSD,Mask_with_MSD,Nomask_with_MSD)
        add = ("INSERT into SOCIAL_DISTANCE (TIME, DAY,PLACE,MASK_WITH_MSD,MASK_WITHOUT_MSD,NOMASK_WITH_MSD,NOMASK_WITHOUT_MSD,TOTAL_PEOPLE) values(:1,:2,:3,:4,:5,:6,:7,:8)")
        data = [time_st,day,title,Mask_with_MSD,Mask_without_MSD,Nomask_with_MSD,Nomask_without_MSD,total]
        cur.execute(add,data)
        con.commit()
    if flag:
        break
    key = cv2.waitKey(1)
    if key == ord('q'):
        break    
cv2.destroyAllWindows()
con.close()
