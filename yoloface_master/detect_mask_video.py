import sys
import os
import cv2
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


from yoloface_master.utils import *


#####################################################################

model_cfg_path = "./cfg/yolov3-face.cfg"
model_weights_path = "./model-weights/yolov3-wider_16000.weights"
output_dir = "outputs/"

print("[INFO] loading face mask detector model...")
maskNet = load_model("model_detector.model")

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', model_cfg_path)
print('[i] The weights of model file: ', model_weights_path)
print('###########################################################\n')


# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def _main():
    wind_name = 'face detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture('samples/trimmed-000-LONDON WALK _ Oxford Street to Carnaby Street _ En(480P).mp4')

    while True:

        has_frame, frame = cap.read()
        frame = imutils.resize(frame, width=700)
        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            #print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces1 = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces1)))
        print('#' * 60)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces1)))
        ]
        for box in faces1:
            (startX, startY, endX, endY) = box
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(IMG_WIDTH - 1, endX), min(IMG_HEIGHT - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            faces = []
            locs = []
            preds = []

            face = frame[startY:endY, startX:endX]
            if face.size != 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

            # only make a predictions if at least one face was detected
            if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                preds = maskNet.predict(face)

            label = ""


            if(len(preds)>0):
                (mask, withoutMask) = preds[0]

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                acc = "{:.2f}%".format(max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output
            # frame
                cv2.putText(frame, acc, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)




        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break
        cv2.imshow(wind_name, frame)

    #cap.release()
    #cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
