import cv2
import numpy as np

net = cv2.dnn.readNet('doc/yolov3.weights', 'doc/yolov3.cfg')

classes = []
with open("doc/classes.txt", "r") as f:
    classes = f.read().splitlines()
print(classes)
camera = int(input("Index of Camera : "))

cap = cv2.VideoCapture(camera)
font = cv2.FONT_HERSHEY_COMPLEX
colors = np.random.uniform(0, 255, size=(100, 3))
object_present = 0

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            cv2.circle(img,(x+w//2,y+h//2),10,(0,255,0),-1)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(img, label, (x, y), font, 1, (255,0,0), 2)
            cv2.putText(img, str(x) + "," + str(y), (x, y+100), font, 1, (0, 0, 0), 2)


    if object_present != len(indexes):
        print(len(indexes))
        object_present = len(indexes)

    cv2.imshow('Capsule', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()