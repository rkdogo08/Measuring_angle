import cv2

#cap = cv2.VideoCapture(0)
filepath='hho.mp4'

cap = cv2.VideoCapture(filepath)


while(True):

    ret, image = cap.read()    
    if ret == False:
        break;
    
    cv2.imshow("result", image)
    
    T=time.time()-last_time
    Time.append(T)
    last_time = time.time()
    print('- {} seconds'.format(T))
 
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
print("Time per fame",np.mean(np.array(Time)))
cap.release()
cv2.destroyAllWindows()
