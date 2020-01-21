import cv2
import numpy as np
import math
from numpy import linalg
import time

def img_process(image):
    global w,h,gap,low_threshold,high_threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    
    #저역통과 필터를 통한 노이즈의 제거가 필요한데 이 과정에서 상세 정보도 손상될 뿐만 아니라, 검출되는 에지 위치의 정확성도 영향을 받는다.
    #잡음 제거뿐만 아니라 경계선 또한 뚜렷해진 영상으로 변환
    #cv2.bilateralFilter(gray,2,3,8)
    bilateral_img=cv2.bilateralFilter(gray,3,20,20)
    cv2.imshow("bilateralFilter", bilateral_img)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4,4))
    CLAHE_img = clahe.apply(bilateral_img)
    #cv2.imshow("CLAHE", CLAHE_img)
    
    CannyAccThresh = cv2.threshold(CLAHE_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    CannyAccThresh=(CannyAccThresh+20)
    #print(CannyAccThresh)
    CannyThresh = CannyAccThresh/4*3
    edges = cv2.Canny(CLAHE_img, CannyThresh,CannyAccThresh)
    
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    vertices = np.array([[(int(w/2)-gap,h),(int(w/2)-gap, 0), (int(w/2)+gap,0),(int(w/2)+gap,h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    #cv2.rectangle(image,(int(w/2)-gap, 0), (int(w/2)+gap, h),(0,255,0),3)

    cv2.imshow("edges", edges)
    
    return masked_edges

# 기울기가 유사한 직선들의 x,y값들을 평균내서 반환하는 함수
def average_lane(lane_data):
    x1s = []
    y1s = []
    x2s = []
    y2s = []   
    for data in lane_data:
        x1s.append(data[2][0])
        y1s.append(data[2][1])
        x2s.append(data[2][2])
        y2s.append(data[2][3])
    return int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))

def draw_lines(img,lines):
    global w,h,MAX_line
    if True:
        # finds the maximum y value for a lane marker
        # (since we cannot assume the horizon will always be at the same point)
        ys = []
        for ii in lines:
            ys += [ii[1], ii[3]]
        min_y = min(ys)
        max_y = max(ys)
        new_lines = []
        line_dict = {}
 
        for idx, xyxy in enumerate(lines):
            
            # These four lines:
            # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points 
            # Used to calculate the definition of a line, given two sets of coords.
            x_coords = (xyxy[0], xyxy[2])
            y_coords = (xyxy[1], xyxy[3])
 
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, b = linalg.lstsq(A, y_coords)[0]
 
            # calculating our new, and improved, xs
            x1 = (min_y - b) / m
            x2 = (max_y - b) / m
 
            # 기울기, 절편, 실제좌표값을 담는 line_dict[] 리스트를 만듭니다
            line_dict[idx] = [m, b,[int(x1), min_y, int(x2), max_y]]
 
            new_lines.append([int(x1), min_y, int(x2), max_y])
 
        final_lanes = {} 
 
        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
 
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
 
            else:
                found_copy = False
 
                for other_ms in final_lanes_copy:
                    if not found_copy:
                        if abs(other_ms*1.05) > abs(m) > abs(other_ms*0.95):
                            if abs(final_lanes_copy[other_ms][0][1]*1.05) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.95):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [[m,b,line]]
        
        line_counter = {}
 
        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes  = sorted(line_counter.items(), key=lambda item: item[1])[::-1]

        line_list=[]
        for i,line in enumerate(top_lanes):
            
            if i>MAX_line-1:
                break
            
            lane_id = top_lanes[i][0]
            xy=average_lane(final_lanes[lane_id])        
            line_list.append(xy)

        return line_list
 
def Hough(img,masked_edges):
    global gap,MAX_angle,angle_arr
    lines = cv2.HoughLines(masked_edges,1,np.pi/180,135)
    line_list=[]

    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 5000*(-b))
        y2 = int(y0 - 5000*(a))

        angle =theta * 180 / 3.14 -90
        if (angle <MAX_angle)and(angle>-MAX_angle):
            line_list.append([x1,y1,x2,y2])
            
    line_list = draw_lines(img, np.array(line_list))
    
    for line in line_list:
        x1,y1,x2,y2=line   
        x3=int(w/2)-gap
        x4=int(w/2)+gap
        A=(y2-y1)/(x2-x1)
        B=y2-A*x2
        angle_arr.append(np.arctan(A)/np.pi*180)        
        print(np.arctan(A)/np.pi*180, file=f) 
        y3=A*x3+B
        y4=A*x4+B
        if(y3<0):
            y3=1
            x3=(y3-B)/A
        elif(y3>h):
            y3=h-1
            x3=(y3-B)/A
        if(y4>h):
            y4=h-1
            x4=(y4-B)/A
        elif(y4<0):
            y4=1
            x4=(y4-B)/A
        cv2.circle(img, (int(x3), int(y3)), 8, (255, 255, 0), -1)
        cv2.circle(img, (int(x4), int(y4)), 8, (0, 0, 255), -1)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
    
    return img

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

#cap = cv2.VideoCapture(0)
filepath='rev_stop'

f = open(filepath+'.txt','w')
cap = cv2.VideoCapture(filepath+'.mp4')
MAX_line=1
MAX_angle=60

Time=[]
angle_arr=[]
while(True):

    ret, image = cap.read()    
    if ret == False:
        break;
    
    last_time = time.time()
    image=rotate_image(image,-90)
    h,w, channel = image.shape
    gap=int(w/5)
    masked_edges=img_process(image)

    try:
        result=Hough(image,masked_edges)
        cv2.imshow("result", result)

    except Exception as ex:
        #print (ex)
        cv2.imshow("result", image)
    T=time.time()-last_time
    Time.append(T)
 
    if cv2.waitKey(1) & 0xFF == 27:
        break   
print("Time per fame",np.mean(np.array(Time)))
print("Time",np.sum(np.array(Time)))
f.close()
cap.release()
cv2.destroyAllWindows()
