import cv2
import os

path = 'Documents'
images = []
documentTypes = ['No document detected']
files = os.listdir(path)
orb = cv2.ORB_create(nfeatures=1000)
print('Total Documents Detected:', len(files))

# Import images and document types
for file in files:
    currentImg = cv2.imread(f'{path}/{file}', 0)
    images.append(currentImg)
    documentTypes.append(os.path.splitext(file)[0])


def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
        imgKp = cv2.drawKeypoints(img, kp, None)
        cv2.imshow('test', imgKp)
        cv2.waitKey(0)
    return desList


def findID(img, desList, threshold):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalID = 0
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass

    print(matchList)

    if len(matchList) != 0:
        if max(matchList) > threshold:
            finalID = matchList.index(max(matchList)) + 1
    return finalID


desList = findDes(images)
print(desList)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = findID(img2, desList, 10)
    cv2.putText(imgOriginal, documentTypes[id], (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)
