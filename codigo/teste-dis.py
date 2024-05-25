import copy
import math
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import cv2
import pandas
import time
import requests

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch
import pyttsx3
from gtts import gTTS
import pygame
import os


import stereo_image_utils
from stereo_image_utils import get_cost, draw_detections, annotate_class2 
from stereo_image_utils import get_horiz_dist_corner_tl, get_horiz_dist_corner_br, get_dist_to_centre_tl, get_dist_to_centre_br, get_dist_to_centre_cntr

#n, s, m, l, x
# see https://github.com/ultralytics/ultralytics for more information
model = YOLO("best.pt")
#class names
names =  model.model.names

cnt = 0

#focal length. Pre-calibrated in stereo_image_v6 notebook

  
fl = -3.474215865945297
tantheta = 0.3918921068381497
DEBUG = 1


def speak(text):
    voice = pyttsx3.init()
    voice.setProperty('voice', 'brazil')
    voice.setProperty('age', 15)
    voice.setProperty('gender', 'female')
    voice.setProperty('rate', 160)
    voice.say(text)
    voice.runAndWait()

def speak1(text, lang='pt', speed=1.0):
    tts = gTTS(text=text, lang=lang, slow=False)  # Configura a velocidade da fala
    tts.save("temp.mp3")  # Salvando temporariamente o áudio em um arquivo mp3
    os.system("play " +"temp.mp3"+" tempo 1.4")


if __name__ == '__main__':

    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(4)
    while True:
        ### capture the images
        # Registrar o tempo inicial
        inicio = time.time()
  
        
        if cap_left.isOpened():
            ret_l, frame_l = cap_left.read()
            #release the capture to stop a queu building up. I'm sure there are more efficient ways to do this.
            cap_left.release()
            
            if ret_l:
                pass
            else:
                cap_left.release()
                print("left eye not opened")

        if cap_right.isOpened():
            ret_r, frame_r = cap_right.read()
            #release the capture to stop a queu building up. I'm sure there are more efficient ways to do this.
            cap_right.release()

            
            if ret_r: 
                pass
            else:
                cap_right.release()
                print("right eye not opened")

        if ret_r and ret_l :
            imgs = [cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB),cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)]
            out_l = []
            out_r =[]
            #do stereo matching
            if cnt == 0:  #this condition added mostly for debugging
                out_l = (model.predict(source =cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB), save=False, conf = 0.4, save_txt=False, show = False ))[0]
                out_r = (model.predict(source =cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB), save=False, conf = 0.4, save_txt=False, show = False ))[0]
                
            #do stereo pair matching. See file below for details.
            # https://github.com/jonathanrandall/esp32_stereo_camera/blob/main/python_notebooks/stereo_image_v6.ipynb

            
            if cnt == 0 and (out_l.boxes.shape[0]>0 and out_r.boxes.shape[0]>0): #cnt is just a control for debugging
                #boxes are the coordinates of the boudning boxes.
                cnt = 0 #1
                
                #find the image centre
                sz1 = frame_r.shape[1]
                centre = sz1/2

                #dets are bounding boxes and lbls are labels.
                det = []
                lbls = []
                
                #det[0] are the bounding boxes for the left image
                #det[1] are the bounding boxes for the right image

                if(out_l.boxes.shape[0]>0 and out_r.boxes.shape[0]>0):
                    det.append(np.array(out_l.boxes.xyxy))
                    det.append(np.array(out_r.boxes.xyxy))
                    lbls.append(out_l.boxes.cls)
                    lbls.append(out_r.boxes.cls)
                
                print(det)
                
                #get the cost of matching each object in the left image
                #to each object in the right image
                cost = get_cost(det, lbls = lbls,sz1 = centre)
                
                #choose optimal matches based on the cost.
                tracks = scipy.optimize.linear_sum_assignment(cost)                
                
                #find top left and bottom right corner distance to centre (horizonatlly)
                dists_tl =  get_horiz_dist_corner_tl(det)
                dists_br =  get_horiz_dist_corner_br(det)

                final_dists = []
                dctl = get_dist_to_centre_tl(det[0],cntr = centre)
                dcbr = get_dist_to_centre_br(det[0], cntr = centre)
                
                #measure distance of object from the centre so I can see how far I need to turn.
                d0centre = get_dist_to_centre_cntr(det[0], cntr = centre)
                d1centre = get_dist_to_centre_cntr(det[1], cntr = centre)
                
                #classes for left and right images. nm0 is left, nm1 is right
                q = [i.item() for i in lbls[0]]
                nm0 = [names[i] for i in q]
                q = [i.item() for i in lbls[1]]
                nm1 = [names[i] for i in q]
                
                #check if bottle is upright. height greater than width and move car certain angle.

                for i, j in zip(*tracks):
                    if dctl[i] < dcbr[i]:
                        final_dists.append((dists_tl[i][j],nm0[i]))

                    else:
                        final_dists.append((dists_br[i][j],nm0[i]))
                
                #final distances as list
                fd = [i for (i,j) in final_dists]
                #find distance away
                dists_away = (10/2)*sz1*(1/tantheta)/np.array((fd))+fl
                cat_dist = []
                for i in range(len(dists_away)):
                    cat_dist.append(f'{nm0[(tracks[0][i])]} {dists_away[i]:.1f}cm')
                    print(f'{nm0[(tracks[0][i])]} is {dists_away[i]:.1f}cm away')
                t1 = [list(tracks[1]), list(tracks[0])]
                frames_ret = []
                if DEBUG:
                    for i, imgi in enumerate(imgs):
                        img = imgi.copy()
                        deti = det[i].astype(np.int32)
                        draw_detections(img,deti[list(tracks[i])], obj_order=list(t1[1]))
                        annotate_class2(img,deti[list(tracks[i])],lbls[i][list(tracks[i])],cat_dist)
                        frames_ret.append(img)
                    #cv2.imshow("left_eye", cv2.cvtColor(frames_ret[0],cv2.COLOR_RGB2BGR))
                    #cv2.imshow("right_eye", cv2.cvtColor(frames_ret[1],cv2.COLOR_RGB2BGR))
                dist_round = [round(i) for i in dists_away]

                print("A função levou {:.5f} segundos para ser executada.".format(time.time()-inicio))
                text = "Caneca a " +str(dist_round[0])+ " centímetros a esquerda"
                speak1(text)
                time.sleep(10)

            key = cv2.waitKey(1)

                # Verificar se o usuário pressionou a tecla 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap_left.release()
    cap_right.release()
