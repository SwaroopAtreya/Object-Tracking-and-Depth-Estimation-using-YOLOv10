import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

num = 0

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()

    k = cv2.waitKey(5)


    if k==27:
        break
    elif k==ord('5'):
        cv2.imwrite('images/stereoLeft/imageL'+str(num)+'.png',img)
        cv2.imwrite('images/stereoright/imageR'+str(num)+'.png',img)
        print("images saved!")
        num +=1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)


cap.release()
cap2.release()


cv2.destroyAllWindows()
