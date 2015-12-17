import cv2

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    


    try:
        cascPath = argv[1]
    except Exception:
        cascPath = "haarcascade_frontalface_default.xml"

    def get_frame(self):
        #  success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        

        
        faceCascade = cv2.CascadeClassifier(VideoCamera.cascPath)

        # Capture frame-by-frame
        ret, frame = self.video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags =  cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        ret, jpeg = cv2.imencode('.jpg', frame)
        #todo just cast jpeg to
        # file = open("binary.file","wb")
        # file.write(jpeg)
        # file.close()
        # jpeg = open("binary.file","rb").read()
        return jpeg.tobytes()