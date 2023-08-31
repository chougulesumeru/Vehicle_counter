#vehicle counter in python 

#importing some libraries
import cv2
from cv2 import dilate  
import numpy as np

#import the video path 
vid= cv2.VideoCapture('vehicle_counter.mp4')


count_line_position=550
min_width_rec= 80   #minimum width rectangle
min_height_rec=80   #minimum height rectangle

#algorithm set
algo= cv2.createBackgroundSubtractorMOG2() 

def center_handle(x,y,w,h):
    x1=int(w/2) 
    y1=int(h/2)
    cx= x+x1
    cy=y+y1
    return cx,cy

detect= []     #null list
offset=6       #allow error in pixel
counter=0      #initilize first

while True:
    ret,frame1= vid.read()
    grey= cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5) 
    
    #apply on all vaehicle
    img_sub = algo.apply(blur)
    dilate= cv2.dilate(img_sub,np.ones((5,5))) 
    kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
    dilatedata= cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    dilatedata= cv2.morphologyEx(dilatedata,cv2.MORPH_CLOSE,kernel)
    countershape,h= cv2.findContours(dilatedata,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow('grey image',dilatedata)
    
    #draw a line 
    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
    
    #draw rectangle to all vehicles
    for (i,c) in enumerate(countershape):
        (x,y,w,h)= cv2.boundingRect(c) 
        validate_counter= (w>=min_width_rec) and (h>=min_height_rec)
        if not validate_counter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(124,252,0),2)
        cv2.putText(frame1,"vehicle"+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2) 
        
        #draw a circle on vehicles
        center=center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1) 
        
        for(x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1 
            
            cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
            detect.remove((x,y))
            print(" Vehicle Counter :- " + str(counter))
            
    cv2.putText(frame1,"Vehicle Counter : "+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)     
    
    cv2.imshow('vehicle',frame1)
    
    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows() 
vid.release()  
  