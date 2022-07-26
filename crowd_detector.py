# import necessary libraries
import cv2
import imutils
import numpy as np
import argparse
from pygame import mixer

# initialize alarm sound
mixer.init()
sound = mixer.Sound('alarm.WAV')

global threshold

class Crowd_Detection:

    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame, threshold):
        bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
        
        person = 1
        for x,y,w,h in bounding_box_cordinates:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
            person += 1
        
        if (person -1) > threshold:
            sound.play()
            cv2.putText(frame, 'CROWD DETECTED', (350,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3)
            print('CROWD DETECTED')
        else:
            cv2.putText(frame, 'CROWD NOT DETECTED', (350,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3)
            print('CROWD NOT DETECTED')

        cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        cv2.imshow('Output', frame)

        return frame

    def detectByCamera(self, writer, threshold):   
        video = cv2.VideoCapture(0)
        print('Detecting people...')
        cd = Crowd_Detection()
        while True:
            check, frame = video.read()

            frame = cd.detect(frame, threshold)
            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    def detectByPathVideo(self, path, writer, threshold):
        video = cv2.VideoCapture(path)
        cd = Crowd_Detection()
        check, frame = video.read()
        if check == False:
            print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
            return

        print('Detecting people...')
        while video.isOpened():
            #check is True if reading was successful 
            check, frame =  video.read()

            if check:
                frame = imutils.resize(frame , width=min(1200,frame.shape[1]))
                frame = cd.detect(frame, threshold)
                
                if writer is not None:
                    writer.write(frame)
                
                key = cv2.waitKey(1)
                if key== ord('q'):
                    break
            else:
                break

        video.release()
        cv2.destroyAllWindows()

    def detectByPathRTSP(self, path, writer,threshold):
        rtsp = cv2.VideoCapture(path)
        cd = Crowd_Detection()

        check, frame = rtsp.read()
        if check == False:
            print('RTSP Broken')
            return

        print('Detecting people...')
        while rtsp.isOpened():
            #check is True if reading was successful 
            check, frame =  rtsp.read()

            if check:
                frame = imutils.resize(frame , width=min(800,frame.shape[1]))
                frame = cd.detect(frame, threshold)
                
                if writer is not None:
                    writer.write(frame)
                
                key = cv2.waitKey(1)
                if key== ord('q'):
                    break
            else:
                break

        rtsp.release()
        cv2.destroyAllWindows() 
        
    def argsParser(self):
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
        arg_parse.add_argument("-r", "--rtsp", default=None, help="Enter RTSP link")
        arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
        arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
        args = vars(arg_parse.parse_args())

        return args

if __name__ == "__main__":

    while True:

        HOGCV = cv2.HOGDescriptor()
        HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        cd = Crowd_Detection()
        args = cd.argsParser()

        video_path = "Videos/store.mp4"
        rtsp = "rtsp://164.52.204.9:8554/har"
        output_path = "OutputVideos/output.avi" # Change this to mp4 or mov maybe 
        # writer = None
        # if args['output'] is not None and rtsp is None:
        writer = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'XVID'), 10, (600,600))

        print("DETECT CROWD VIA:\n")
        print("1.VIDEO")
        print("2.RTSP")
        print("3.CAMERA IN REAL-TIME")    
        print("4.EXIT")
        print("\n")

        choice=int(input("Enter your choice: "))
        print("\n")
        
        threshold = int(input("Enter your Threshold value: "))
        print("\n")


        if choice==1:
            print('[INFO] Opening Video from path.')
            cd.detectByPathVideo(video_path, writer,threshold)
        
        elif choice==2:
            print('[INFO] Opening RTSP link.')
            cd.detectByPathRTSP(rtsp, writer, threshold)
        
        elif choice==3:
            print('[INFO] Opening Web Cam.')
            cd.detectByCamera(writer, threshold)

        elif choice==4:
            break

        else:
            print("Wrong Choice")
