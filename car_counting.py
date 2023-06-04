import cv2
import numpy as np

cap = cv2.VideoCapture("traffic.mp4")
backsub = cv2.createBackgroundSubtractorMOG2()
car_count = 0

while True:
    ret, frame = cap.read()

    if ret:
        mask = backsub.apply(frame)

        cnts, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hier = hier[0]

        except:
            hier = []

        for contour, hierarchy in zip(cnts, hier):
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 40 and h > 40:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                if x > 50 and x < 70:
                    car_count += 1

        cv2.putText(frame, "Car " + str(car_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Car", frame)

    if cv2.waitKey(40) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
