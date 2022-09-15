import cv2
import numpy as np
import pytesseract as ocr
import dlib
from imutils import face_utils
import time
from datetime import datetime
from utils import for_point_warp

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/mohcenaouadj/Téléchargements/shape_predictor_68_face_landmarks.dat')


ratio = 1


cap = cv2.VideoCapture(0)
while 1:
    _, image = cap.read()
    inputImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([80,0,139])
    upper_blue = np.array([111,118,220])
    blue  = cv2.inRange(inputImg, lower_blue, upper_blue)
    showBlue = cv2.bitwise_and(image,image, mask= blue)
    #cv2.imshow('showBlue',showBlue)
    
    colorLow = [[80,0,139], [6,79,112], [105,68,90], [0,44,100]]
    colorUp = [[111,118,220], [29,255,220], [180,239,170], [7,200,200]]
    value = [0, 0, 0, 0]
    for x in range(0,4):
        lower_color = np.array([colorLow[x][0],colorLow[x][1],colorLow[x][2]])
        upper_color = np.array([colorUp[x][0],colorUp[x][1],colorUp[x][2]])
        mask  = cv2.inRange(inputImg, lower_color, upper_color)
        rows,cols = mask.shape
        for i in range(rows):
            for j in range(cols):
                pixel = mask[i,j]
                if pixel == 255:
                    value[x] = value[x] + 1
        #print "value ",x," = ",value[x]

    result = value[0]
    for i in range(1,4):
        result = result - value[i]
    #print "result = ",result

    font = cv2.FONT_HERSHEY_SIMPLEX
    if result >= 3000:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(gray_image, (3,3), 0)
        retval,imgThreshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        edge = cv2.Canny(gray_image, 50, 150)
        #cv2.imshow('edge', edge)
        img_r = cv2.inRange(gray_image,np.array(0,dtype='uint8'),np.array(140,dtype='uint8'))
        imgtest = img_r + edge
        #cv2.imshow('imgtest', imgtest)

        kernel = np.ones((5,5),np.uint8)

        dilation = cv2.dilate(imgtest,kernel)
        #cv2.imshow('dilation', dilation)

        


        cnts = cv2.findContours(dilation.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dilation,cnts[0],-1,(255,0,0),-1)

        indexAreMax = 0
        maxx = 0
        i = 0
        for c in cnts[0]:
            if cv2.contourArea(c)>maxx:
                maxx = cv2.contourArea(c)
                rect = cv2.minAreaRect(c)
                indexAreMax = i
                (x3,y3,w3,h3)= cv2.boundingRect(c)
          
            i = i + 1
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,0,255),2)
        #print(box)
        warped = for_point_warp(box/ratio, image)   

      
        #detect warped
        warped_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        retval,warpedThreshold = cv2.threshold(warped_image, 127, 255, cv2.THRESH_BINARY)

        kernelW = np.ones((2,2))
        dilationW = cv2.dilate(warpedThreshold,kernelW)
        #cv2.imshow('warpedThreshold',warpedThreshold)
        #cv2.imshow('dilationW',dilationW)
        h,w,cr = warped.shape
        roiID = dilationW[int((h*10)/100):int((h*18)/100), int((w*44)/100):int(w)]
        cv2.rectangle(warped,(int((w*44)/100),int((h*10)/100)),(int(w),int((h*18)/100)),(0,255,0),2)

        roiNameTH = dilationW[int((h*18)/100):int((h*30)/100), int((w*29)/100):int(w)]
        cv2.rectangle(warped,(int((w*29)/100),int((h*18)/100)),(int(w),int((h*30)/100)),(0,255,0),2)

        roiNameEN = dilationW[int((h*29)/100):int((h*37)/100), int((w*39)/100):int(w)] 
        cv2.rectangle(warped,(int((w*39)/100),int((h*29)/100)),(int(w),int((h*37)/100)),(0,255,0),2)

        roiLastNameEN = dilationW[int((h*37)/100):int((h*45)/100), int((w*44)/100):int(w)]
        cv2.rectangle(warped,(int((w*44)/100),int((h*37)/100)),(int(w),int((h*45)/100)),(0,255,0),2)

        roiImg = warped[int((h*48)/100):int((h*91)/100), int((w*75)/100): int((w*98)/100)]
        cv2.rectangle(warped,(int((w*75)/100),int((h*48)/100)),(int((w*98)/100),int((h*91)/100)),(0,255,0),2)
        """
        print('ID')
        print(pytesseract.image_to_string(roiID, lang = 'eng',config=tessdata_dir_config))
        print('NameTH')
        print(pytesseract.image_to_string(roiNameTH, lang = 'tha',config=tessdata_dir_config))
        print('NameEN')
        print(pytesseract.image_to_string(roiNameEN, lang = 'eng',config=tessdata_dir_config))
        print('LastNameEN')
        print(pytesseract.image_to_string(roiLastNameEN, lang = 'eng',config=tessdata_dir_config))

        id_card = pytesseract.image_to_string(roiID, lang = 'eng',config=tessdata_dir_config)
        nameTH = pytesseract.image_to_string(roiNameTH, lang = 'tha',config=tessdata_dir_config)
        nameEng = pytesseract.image_to_string(roiNameEN, lang = 'eng',config=tessdata_dir_config)
        lastNameEng = pytesseract.image_to_string(roiLastNameEN, lang = 'eng',config=tessdata_dir_config)
        timestr = time.strftime("%d_%m_%Y-%H_%M_%S")
        """
        #detect face
        """
        rects = detector(gray_image,1)
       
        if len(rects) == 0:
            print ("Don't have any face")
        else:
            for(i, rect) in enumerate(rects):
                (xf, yf, wf, hf) = face_utils.rect_to_bb(rect)
                cv2.rectangle(roiImg,(xf,yf),(xf+wf,yf+hf),(0,255,0),2)
                roiface = image[yf:yf+hf,xf:xf+wf]
                f = open('data.txt','a')
                f.write("\nID : "+id_card.encode('utf8')+"\n")
                f.write("Thai Name : "+nameTH.encode('utf8')+"\n")
                f.write("Name : "+nameEng.encode('utf8')+"\n")
                f.write("Last Name : "+lastNameEng.encode('utf8')+"\n")
                f.write ("Image : "+timestr+".jpg"+"\n")
                f.write("\n")
                cv2.imwrite(timestr+".jpg", roiImg)
                f.close()
                cv2.imshow('roiface',roiface)
        
        """                                      

        #cv2.imshow('warped',warped)
        #cv2.imshow('dilation',dilation)
        cv2.putText(image,'Find ID Card',(30,50), font, 0.8,(105,210,0),2,cv2.LINE_AA)

        
    else:
        cv2.putText(image,'This card not a ID Card',(30,50), font, 0.8,(100,100,255),2,cv2.LINE_AA)
            


    cv2.imwrite('roiID.png',roiID)
    #cv2.imshow('roiNameTH',roiNameTH)
    #cv2.imshow('roiNameEN',roiNameEN)
    #cv2.imshow('roiLastNameEN',roiLastNameEN)
    #cv2.imshow('roiImg',roiImg)
    
    #cv2.imshow('gray_image',gray_image)
    #cv2.imshow('image',image)
    #cv2.imshow('face',roiface)
    #cv2.imshow('erosion',erosion)
    
    cv2.waitKey(1)
