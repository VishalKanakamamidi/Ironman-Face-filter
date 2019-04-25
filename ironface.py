# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
from imutils.video import VideoStream
from PIL import Image
import imutils


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = 'frozen_inference_graph_face.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.2
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    #vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)


    while True:
        count = 0
        while(True):
            
            img = vs.read()                             # for skipping frames
            count = count + 1
            
            if(count == 5):
                break
        img = vs.read()
        img = imutils.resize(img, width=600)

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        if(len(boxes)>0):
            background = Image.fromarray(img)
            for i in range(len(boxes)):
                
                if  scores[i] > threshold:
                    box = boxes[i]
                    # cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                    # cv2.putText(img, str("Face"), (box[1]-10,box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    s_img = cv2.imread("iron.jpg")
                # frame[y:y+278, x:x+172] = s_img
                # cv2.imshow("Frame1", s_img)
                    
                    foreground = Image.open("iron.jpg")
                    x = box[1]
                    y = box[0]
                    w = box[3]-box[1]
                    h = box[2]-box[0]
                    size = w+int(w/1.3),h+int(h/1.3)
                    foreground.thumbnail(size, Image.ANTIALIAS)
                    x1 = (2*x+w)/2
                    y1 = (2*y+h)/2
                    background.paste(foreground, (int(x), int(y-y*0.2)), foreground)
                    try:
                        background.save("ironman.jpg")
                    except:
                        continue
                    img = cv2.imread("ironman.jpg")
            
            cv2.imshow("Frame1", img)
        if(len(boxes) == 0):
            cv2.imshow("Frame1", frame)


        # cv2.imshow("preview", img)
        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break