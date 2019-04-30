import numpy as np
import cv2
import face_recognition
import os
from time import sleep
import requests
import sys
import pickle

cap = cv2.VideoCapture(1)

width = cap.get(3)  # float
height = cap.get(4) # float

face_cascade = cv2.CascadeClassifier('G:\\Home\\anaconda3\\envs\\hackriddle\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

IP = "34.73.48.217"
PORT = 80

def trainPersonModel(classifier):
    person = []
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            enc = face_recognition.face_encodings(frame)
            person.append(enc)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return np.asarray(person)


def compareEncoding(baseEnc, library, SCALE=10, THRESHOLD=75):
    bestMatch = None
    for person in library:
        try:
            accuracys = face_recognition.face_distance(baseEnc, person["encoding"])
        except:
            print("Error comparing faces")
            continue
        
        acc = []
        for a in accuracys:
            acc.append(np.average(a))

        finalAccuracy = 100 - ((sum(acc) / len(acc)) * SCALE)
        print(finalAccuracy)
        if bestMatch == None and finalAccuracy >= THRESHOLD:
            bestMatch = (finalAccuracy, person)
        elif bestMatch and finalAccuracy >= THRESHOLD:
            bestMatch = (finalAccuracy, person) if finalAccuracy > bestMatch[0] else bestMatch
    
    return bestMatch


def averagePerson(person):
    return np.mean( np.array(person), axis=0 )


def trainGroup():
    lib = []

    while True:
        firstTrain = trainPersonModel(face_cascade)

        print("Who was that?")
        name = input()
        # TODO: Try to clean array before it gets averaged together
        try:
            lib.append({"name": name, "encoding": averagePerson(firstTrain)})
        except:
            print("Failed to train model")
        print("Are you done?[y/n]")
        name = input()
        if name == "y":
            break
    return lib
        

def detectPeople(lib):
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance = set()
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        encs = face_recognition.face_encodings(frame)
        
        if len(encs) > 0:
            best = compareEncoding(encs[0], lib, THRESHOLD=96)
            if best:
                cv2.putText(frame, best[1]["name"] + ": " + str(round(best[0])),(10,50), font, 2,(255,255,255))
                print("Best match: ", best[0], best[1]["name"])
                name = best[1]["name"].split()
                attendance.add((name[0], name[1]))
            else:
                print("No Match")

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return attendance


def postLib(lib):
    for person in lib:
        data = dict()
        data["fname"], data["lname"] = person["name"].split()
        print(len(person["encoding"][0]))
        string = ""
        for val in person["encoding"][0]:
            string += str(val) + ","
        print(string)
        data["data"] = string[:-1]
        r = requests.post("http://" + IP + ":" + str(PORT), data=data)
        print(r)

def pullData():
    lib = []
    r = requests.delete("http://" + IP + ":" + str(PORT))
    # print(r.text)
    for p in r.json():
        print(p)
        encoding = p[3]
        encoding = np.asarray([float(landmark) for landmark in encoding.split(",")])
        person = dict()
        person["name"] = p[1] + " " + p[2]
        person["encoding"] = [encoding]
        lib.append(person)
    return lib

def postAttendance(attendList):
    if len(attendList) == 0:
        return

    data = [{"fname": tup[0], "lname": tup[1]} for tup in attendList]

    print(data)
    r = requests.put("http://" + IP + ":" + str(PORT), json={"data": data})
    if r.status_code != 200:
        print("Error submitting attendance request")
        

if len(sys.argv) < 2:
    lib = trainGroup()
    postLib(lib)
else:
    if sys.argv[1] == "train":
        lib = trainGroup()
        postLib(lib)
        # f = open("local", 'w')
        # pickle.dump(lib, f)
        # # f.write(lib)
        # f.close()
        
    elif sys.argv[1] == "attendance":
        lib = pullData()
        # f = open("local", 'w')
        # lib = f.read()
        attend = detectPeople(lib)
        postAttendance(attend)


# lib = trainGroup()
# detectPeople(lib)