import cv2  
import dlib  
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  
cap = cv2.VideoCapture(0)  
while True:  
    _, frame = cap.read()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = detector(gray)  
    for face in faces:  
        landmarks = predictor(gray, face)  
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)  
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)  
        cv2.circle(frame, left_eye, 3, (0, 255, 0), -1)  
        cv2.circle(frame, right_eye, 3, (0, 255, 0), -1)  
    cv2.imshow("Eye Tracking", frame)  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
cap.release()  
cv2.destroyAllWindows()
