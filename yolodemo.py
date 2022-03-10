# %%
import cv2
import numpy as np
import keyboard
import requests
import os

path = r'F:\Users\admin\Downloads\99.mp4'#TODO #1 :изменить на путь к датасету.

cap = cv2.VideoCapture(0)#для вывода видео

whT = 320#idk what is that
confThreshold = 0.5
nmsThreshold = 0.3

links=[['https://pjreddie.com/media/files/yolov3-spp.weights', 'yolov3-spp.weights'], ['https://github.com/pjreddie/darknet/raw/master/data/coco.names', 'coco.names'], ['https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-spp.cfg', 'yolov3-spp.cfg']]
#Проверка на наличие нужных файлов
def downloadFiles(link, name):
    r = requests.get(link, allow_redirects=True)
    with open(name, 'wb') as f:
        f.write(r.content)

for i in links:
    if(not os.path.exists(f'./{i[1]}')):
        downloadFiles(i[0],i[1])
        

classesFile = 'coco.names'#для классов YOLO штука (чтобы искать не только котов, а и машин, например)
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3-spp.cfg'#подгрузка YOLO3-spp
modelWeights = 'yolov3-spp.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#функция, для нахождения котиков UwU :з
def findObjects(outputs,img):
    hT, wT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int (det[2] * wT), int (det[3] * hT)
                x, y = int (det[0] * wT) - w / 2, int (det[1] * hT) - h / 2
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (int(x),int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        cv2.putText(img, f'WOLF {int(confs[i]*100)}%',(int(x),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

while True:
    #TODO #2 изменить на while ролик не кончился *idk how btw*
    success, img = cap.read()
    
    blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObjects(outputs,img)

    cv2.imshow('Image',img)
    cv2.waitKey(1)
    
    if keyboard.is_pressed("q"):##если не сделать TODO #2, то при нажатии q уберутся все окна и выполнение программы завершится.
        cv2.destroyAllWindows()
        break