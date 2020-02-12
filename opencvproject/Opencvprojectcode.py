from PyQt5 import QtCore, QtGui, QtWidgets, uic
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import random
import math
from PIL import Image, ImageQt
import sys

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('gui.ui', self)
        self.show()
    #button dealing with images
        self.loadButton.clicked.connect(self.setImage)
        self.pushButtonC.clicked.connect(self.ColorSet)
        self.LogoAdder.clicked.connect(self.Logodupere)
        self.Saltybutton.clicked.connect(self.thesaltyfunction)
        self.eqhisto.clicked.connect(self.equalizehisto)
        self.gethisto.clicked.connect(self.histogram)
        self.EdgeB.clicked.connect(self.edgedetectionf)
        self.houghtB.clicked.connect(self.houghf)
        self.contourB.clicked.connect(self.contoursf)
        self.ResetB.clicked.connect(self.resetducul)
        self.HarisB.clicked.connect(self.harrisf)
        self.ShapeB.clicked.connect(self.Shapelachevre)
        self.ExesimpleB.clicked.connect(self.harrypotersontpere)
        self.featuredrawB.clicked.connect(self.jedessinedespointmulticolor)
        self.bluryblury.clicked.connect(self.blurypop)
        self.cascadeimg.clicked.connect(self.cascadesurimage)
        
    #button dealing with camera 
        self.loadCamb.clicked.connect(self.lacameraesdouce)
        self.ColorCamB_2.clicked.connect(self.lescouleurchangemaispaslescon)
        self.Saltybutton_2.clicked.connect(self.thesaltyfunctioncameraedition)
        self.Colo_track.clicked.connect(self.lebleurcestshtroumpfs)
        self.eqhisto_2.clicked.connect(self.etlajegaliseecbo)
        self.EdgeB_2.clicked.connect(self.lesboredsontbeau)
        self.contourB_2.clicked.connect(self.lescontoursontimportant)
        self.HarisB_2.clicked.connect(self.harrisf2)
        self.bluryblury_2.clicked.connect(self.blurypopcam)
        self.featuredrawB_2.clicked.connect(self.featurincroyable2)
        self.cascade.clicked.connect(self.lachutedo)
        self.LogoAdder_2.clicked.connect(self.logosurmacamera)

# -----------------------------------------------------------------------------------------------------------------------------------------#
# image part
# -----------------------------------------------------------------------------------------------------------------------------------------#        
    def setImage(self):
        global filename
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filename:
            global I
            I = cv2.imread(filename)
            x,y,z = I.shape
            x = int(x * 0.70)
            y = int(y * 0.70)
            print (x,y)
            I = cv2.resize(I,(y,x))
            cv2.imshow('image', I)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    def resetducul(self):
        resetfilename = filename
        if resetfilename:
            global J
            im = cv2.imread(filename)
            im = cv2.resize(I,(640,480))
            cv2.imshow('image', im)
            return J
            cv2.waitKey(0)
            cv2.destroyAllWindows()            
            
    def ColorSet(self):
        combotext = self.colorBox.currentText()
        global J 
        
        if combotext == "BGR":
            J = I
            #image = np.hstack((I,J))
            cv2.imshow('Bgr color range',J)  
            return J                  
        elif combotext == 'HSv':
            J = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
            #image = np.hstack((I,J))
            cv2.imshow('hsv color range',J) 
            return J
        elif combotext == 'GrayScale':   
            J = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
            #image = np.hstack((I,J))
            cv2.imshow('Gray color range',J)
            return J
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
            
    def Logodupere(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select a logo image", "", "Image Files (*.png *.jpg *.bmp)")
        if filename:
            img = J
            logo = cv2.imread(filename)
            cv2.imshow('logo to be added',logo)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)    
            watermark = cv2.resize(logo, (int(logo.shape[0]/3), int(logo.shape[1]/3)))
            watermark = np.fliplr(np.flipud((watermark)))
            watermark = cv2.cvtColor(watermark,cv2.COLOR_BGR2BGRA)
            frame_h, frame_w, frame_c = img.shape
            overlay = np.zeros((frame_h, frame_w, 4), dtype = 'uint8')    
            watermark_h, watermark_w, watermark_c = watermark.shape
            x, y, w, h = cv2.selectROI(img)
            for i in range(0,watermark_h):
                for j in range(0,watermark_w):
                    if watermark[i,j][3] != 0:
                        offsety = y
                        offsetx = x
                        h_offset = frame_h - watermark_h - offsety
                        w_offset = frame_w - watermark_w - offsetx
                        overlay[h_offset + i, w_offset + j] = watermark[i,j]
            overlay = np.fliplr(np.flipud((overlay)))          
            cv2.addWeighted(overlay, 0.25, img, 1.0 ,0, img)
            img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
            cv2.imshow('logocbo',img)                    
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
                
    def thesaltyfunction(self):
        global J
        global I
        image = J 
        noise_type = self.noiseBox.currentText()
        row,col,ch= image.shape
        
        if noise_type == "gaussian":       
            mean = 0.0
            var = 0.01
            sigma = var**0.5
            gauss = np.array(image.shape)
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            J = noisy.astype('uint8')
            cv2.imshow('gaussian noise',J)
        elif noise_type == "salt&peper":
            s_vs_p = 0.5
            amount = 0.004
            out = image
            # Generate Salt '1' noise
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, k - 1, int(num_salt))
                  for k in image.shape]
            out[coords] = 255
            # Generate Pepper '0' noise
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, k - 1, int(num_pepper))
                  for k in image.shape]
            out[coords] = 0
            J = out
            cv2.imshow('Salt and peper noise', J)        
        elif noise_type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            J = noisy
            cv2.imshow('poisson noise', J)
        elif noise_type == "Sobel&Laplacian":
        # converting to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
        # remove noise
            img = cv2.GaussianBlur(gray,(3,3),0)        
        # convolute with proper kernels
            laplacian = cv2.Laplacian(img,cv2.CV_64F)
            sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
            sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y        
            J = cv2.hconcat([laplacian, sobelx, sobely])         
        # Display the concatenated image
            cv2.imshow('sabol and laplacian',J)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def histogram(self):
        global J
        img = J 
        hist = plt.hist(img.ravel(),256,[0,256]);
        plt.plot(hist)
        plt.show()            
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def equalizehisto(self):
        global J 
        img = J 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        equ = cv2.equalizeHist(gray)
        res = np.hstack((gray,equ)) #stacking images side-by-side
        cv2.imshow('equalized and original image side by side', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
        
    def edgedetectionf(self):
        input_img = J
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,100,200)
        cv2.imshow("image", edges)
 
    def houghf(self):
        input_img = J
        gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        hough = self.houghBox.currentText()
        if hough == "Lines":
            minLineLength = 100
            maxLineGap = 10
            lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
            for x1,y1,x2,y2 in lines[:,0]:
                cv2.line(input_img,(x1,y1),(x2,y2),(0,255,0),2)
            
            cv2.imshow("image", input_img)
        if hough == "Circle":
            cimg = cv2.medianBlur(input_img,5)
            cv2.imshow('test',cimg)
            circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)            
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),2)
            
            cv2.imshow('detected circles',cimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    def contoursf(self):
        input_img = J
        imgray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
        cv2.drawContours(input_img, contours, -1, (0,255,0), 1)
        cv2.imshow("image", input_img)
        
    def harrisf(self):
        inputimg = J
        gray = cv2.cvtColor(inputimg,cv2.COLOR_BGR2GRAY)    
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)        
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)        
        # Threshold for an optimal value, it may vary depending on the image.
        inputimg[dst>0.01*dst.max()]=[0,0,255]        
        cv2.imshow('dst',inputimg)   
        
    def Shapelachevre(self):
        global J
        out = J
        x, y, w, h = cv2.selectROI(out)
        mask = np.zeros(J.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        rect = (x, y, w, h)
        cv2.grabCut(out, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        out = out*mask2[:,:,np.newaxis]
        J = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        cv2.imshow('cuted image',J)
        return J
     
    def harrypotersontpere(self):
        img1 = cv2.imread('hp.jpg',0)
        hpinc = cv2.imread('hp_insc.jpg',0)        
        img2 = cv2.resize(hpinc,(480,640))        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])        
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)        
        cv2.imshow('uneimageincroyable',img3)        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def jedessinedespointmulticolor(self):
        feature_type = self.FeatureB.currentText()
        input_img = J
        if feature_type == "fast":       
            img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)        
            # Initiate FAST object with default values
            fast = cv2.FastFeatureDetector_create()        
            # find and draw the keypoints
            kp = fast.detect(img,None)
            img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))        
            # Print all default params
            print( "Threshold: {}".format(fast.getThreshold()) )
            print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
            print( "neighborhood: {}".format(fast.getType()) )
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
            cv2.imshow('fast_true.png',img2)           
        elif feature_type == "surf":
            img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)        
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
            surf = cv2.xfeatures2d.SURF_create(400)    
        # Find keypoints and descriptors directly
            kp, des = surf.detectAndCompute(img,None)    
            img2 = cv2.drawKeypoints(img,kp,None)   
            cv2.imshow('surf feature',img2)           
        elif feature_type == "sift":
            img = input_img
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(gray,None)        
            img=cv2.drawKeypoints(gray,kp, None)        
            cv2.imshow('sift_keypoints.jpg',img)

    def blurypop(self):
        global J
        input_img = J
        blur_type = self.blurbox.currentText()
        if blur_type == "blur":
            blurred_img = cv2.blur(input_img,(5,5))
            J = blurred_img
            cv2.imshow('blurred image', J)
        elif blur_type == "gaussianblur":
            blurred_img = cv2.GaussianBlur(input_img,(5,5),0)
            J = blurred_img
            cv2.imshow('blurred image', J)
            
    def cascadesurimage(self):        
        Output_ext = J
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
        gray = cv2.cvtColor(Output_ext, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3)
        for rect in faces:
            (x, y, w, h) = rect
            Output_ext = cv2.rectangle(Output_ext, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('image detected',Output_ext)
            
# -----------------------------------------------------------------------------------------------------------------------------------------#
# camera part
# -----------------------------------------------------------------------------------------------------------------------------------------#        
    def lacameraesdouce(self):
        cap = cv2.VideoCapture(0)        
        while True:
            ret, frame = cap.read()            
            cv2.imshow("frame", frame)            
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def lescouleurchangemaispaslescon(self):
        color = self.colorBox_2.currentText()
        global frame
        if color == "BGR":
            cap = cv2.VideoCapture(0)        
            while True:
                ret, frame = cap.read()
                #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                cv2.imshow("frame", frame) 
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()             
        elif color == 'HSv':
            cap = cv2.VideoCapture(0)        
            while True:
                ret, frame = cap.read()   
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                cv2.imshow("frame", frame)  
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()   
        elif color == 'GrayScale':   
            cap = cv2.VideoCapture(0)        
            while True:
                ret, frame = cap.read()  
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                cv2.imshow("frame", frame) 
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()   
            
    def thesaltyfunctioncameraedition(self):
        noise_type = self.noiseBox_2.currentText()
        if noise_type == "salt&peper":
            cap = cv2.VideoCapture(0)        
            while True:
                ret, image = cap.read()
                row,col,ch= image.shape
                s_vs_p = 0.5
                amount = 0.004
                out = image
                # Generate Salt '1' noise
                num_salt = np.ceil(amount * image.size * s_vs_p)
                coords = [np.random.randint(0, k - 1, int(num_salt))
                      for k in image.shape]
                out[coords] = 255
                # Generate Pepper '0' noise
                num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
                coords = [np.random.randint(0, k - 1, int(num_pepper))
                      for k in image.shape]
                out[coords] = 0
                cv2.imshow('Salt and peper noise', out)  
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
        elif noise_type == "salt&peperGray":
            cap = cv2.VideoCapture(0)        
            while True:
                ret, image = cap.read()
                row,col,ch= image.shape
                s_vs_p = 0.5
                amount = 0.004
                out = image
                # Generate Salt '1' noise
                num_salt = np.ceil(amount * image.size * s_vs_p)
                coords = [np.random.randint(0, k - 1, int(num_salt))
                      for k in image.shape]
                out[coords] = 255
                # Generate Pepper '0' noise
                num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
                coords = [np.random.randint(0, k - 1, int(num_pepper))
                      for k in image.shape]
                out[coords] = 0
                out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
                cv2.imshow('Salt and peper noise', out)  
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows() 
        elif noise_type == "Sobel&Laplacian":
            cap = cv2.VideoCapture(0)        
            while True:
                ret, frame = cap.read()  
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                img = cv2.GaussianBlur(frame,(3,3),0)        
            # convolute with proper kernels
                laplacian = cv2.Laplacian(img,cv2.CV_64F)
                sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
                sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y        
                J = cv2.hconcat([laplacian, sobelx, sobely])         
            # Display the concatenated image
                cv2.imshow('sabol and laplacian',J)
                cv2.imshow("frame", frame) 
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows() 
            
    def lebleurcestshtroumpfs(self):
        def nothing(x):
            pass        
        cap = cv2.VideoCapture(0)        
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
        cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)        
        while True:
            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")
            lower_blue = np.array([l_h, l_s, l_v])
            upper_blue = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            result = cv2.bitwise_and(frame, frame, mask=mask)            
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   
            for contour in contours:
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)    
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))         
                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                    cv2.circle(frame, center, 3, (0, 0, 255), -1)
                    cv2.putText(frame,"centroid", (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
                    cv2.putText(frame,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)        
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def etlajegaliseecbo(self):
        cap = cv2.VideoCapture(0)        
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            equ = cv2.equalizeHist(gray)
            res = np.hstack((gray,equ)) #stacking images side-by-side
            cv2.imshow('equalized and original image side by side', res)                       
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows() 
        
    def lesboredsontbeau(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,100,200)
            cv2.imshow("image", edges)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows() 

    def lachutedo(self):        
        def nothing(x):
            pass       
        cap = cv2.VideoCapture(0)       
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")        
        cv2.namedWindow("Frame")
        cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)        
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
            neighbours = cv2.getTrackbarPos("Neighbours", "Frame")        
            faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
            for rect in faces:
                (x, y, w, h) = rect
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)                
            cv2.imshow("Frame", frame)        
            key = cv2.waitKey(1)
            if key == 27:
                break        
        cap.release()
        cv2.destroyAllWindows()        
  
    def lescontoursontimportant(self):
        def nothing(x):
            pass
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            input_img = frame
            imgray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
            cv2.drawContours(input_img, contours, -1, (0,255,0), 1)
            cv2.imshow("image", input_img)
            key = cv2.waitKey(1)
            if key == 27:
                break        
        cap.release()
        cv2.destroyAllWindows()            
                    
    def harrisf2(self):        
        def nothing(x):
            pass
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()        
            inputimg = frame
            gray = cv2.cvtColor(inputimg,cv2.COLOR_BGR2GRAY)    
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray,2,3,0.04)        
            #result is dilated for marking the corners, not important
            dst = cv2.dilate(dst,None)        
            # Threshold for an optimal value, it may vary depending on the image.
            inputimg[dst>0.01*dst.max()]=[0,0,255] 
            cv2.imshow('dst',inputimg) 
            key = cv2.waitKey(1)
            if key == 27:
                break        
        cap.release()
        cv2.destroyAllWindows()

    def featurincroyable2(self):
        feature_type = self.FeatureB_2.currentText()
        if feature_type == "fast": 
            def nothing(x):
                pass
            cap = cv2.VideoCapture(0)
            while True:
                _, frame = cap.read()        
                input_img = frame
                img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)        
                # Initiate FAST object with default values
                fast = cv2.FastFeatureDetector_create()        
                # find and draw the keypoints
                kp = fast.detect(img,None)
                img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))        
                # Print all default params
                print( "Threshold: {}".format(fast.getThreshold()) )
                print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
                print( "neighborhood: {}".format(fast.getType()) )
                print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
                cv2.imshow('fast_true.png',img2) 
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows() 
        elif feature_type == "surf":
            def nothing(x):
                pass
            cap = cv2.VideoCapture(0)
            while True:
                _, frame = cap.read()        
                input_img = frame            
                img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)        
                # Create SURF object. You can specify params here or later.
                # Here I set Hessian Threshold to 400
                surf = cv2.xfeatures2d.SURF_create(400)    
                # Find keypoints and descriptors directly
                kp, des = surf.detectAndCompute(img,None)    
                img2 = cv2.drawKeypoints(img,kp,None)   
                cv2.imshow('surf feature',img2)  
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows() 
        elif feature_type == "sift":
            def nothing(x):
                pass
            cap = cv2.VideoCapture(0)
            while True:
                _, frame = cap.read()        
                input_img = frame            
                img = input_img
                gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
                sift = cv2.xfeatures2d.SIFT_create()
                kp = sift.detect(gray,None)        
                img=cv2.drawKeypoints(gray,kp, None)        
                cv2.imshow('sift_keypoints.jpg',img)  
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows() 

    def blurypopcam(self):
        blur_type = self.blurbox.currentText()
        if blur_type == "blur":
            def nothing(x):
                pass
            cap = cv2.VideoCapture(0)
            while True:
                _, frame = cap.read()        
                input_img = frame
                blurred_img = cv2.blur(input_img,(5,5))
                J = blurred_img
                cv2.imshow('blurred image', J)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows() 
        elif blur_type == "gaussianblur":
            def nothing(x):
                pass
            cap = cv2.VideoCapture(0)
            while True:
                _, frame = cap.read()        
                input_img = frame
                blurred_img = cv2.GaussianBlur(input_img,(5,5),0)
                J = blurred_img
                cv2.imshow('blurred image', J)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows() 
    
    def logosurmacamera(self):
        img_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select a logo image", "", "Image Files (*.png *.jpg *.bmp)")
        logo = cv2.imread(img_path)
        watermark = cv2.resize(logo, (int(logo.shape[0]/3), int(logo.shape[1]/3)))
        watermark = np.fliplr(np.flipud((watermark)))
        watermark = cv2.cvtColor(watermark,cv2.COLOR_BGR2BGRA)
        putlogoimg = np.zeros((480, 640))
        x, y, w, h = cv2.selectROI(putlogoimg)
        
        def nothing(x):
            pass
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            frame_h, frame_w, frame_c = frame.shape
            overlay = np.zeros((frame_h, frame_w, 4), dtype = 'uint8')    
            watermark_h, watermark_w, watermark_c = watermark.shape
            for i in range(0,watermark_h):
                for j in range(0,watermark_w):
                    if watermark[i,j][3] != 0:
                        offsety = y
                        offsetx = x
                        h_offset = frame_h - watermark_h - offsety
                        w_offset = frame_w - watermark_w - offsetx
                        overlay[h_offset + i, w_offset + j] = watermark[i,j]
            overlay = (np.flipud((overlay)))          
            cv2.addWeighted(overlay, 0.25, frame, 1.0 ,0, frame)
            img = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
            img = np.fliplr(img)
            cv2.imshow('logocbo sur une camera',img)                    
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows() 

                 
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()