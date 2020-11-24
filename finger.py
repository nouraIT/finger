# FLASK PART
from flask import Flask
from flask_restful import reqparse, Api, Resource
from flask import request
app = Flask(__name__)

api = Api(app)
if request.method == "POST":
	file = request.files['file']
if file:
	img = Image.open(file)
	print("Image successfully loaded.")

###########
import cv2
import mediapipe as mp

# extra stuff for flask:
@app.route('/')
def helloworld():

	mp_drawing = mp.solutions.drawing_utils
	mp_hands = mp.solutions.hands
	file_list=img
	# For static images:

	hands = mp_hands.Hands(
	    static_image_mode=True,
	    max_num_hands=2,
	    min_detection_confidence=0.7)
	for idx, file in enumerate(file_list):
	  # Read an image, flip it around y-axis for correct handedness output (see
	  # above).
	  image = cv2.flip(cv2.imread(file), 1)
	  # Convert the BGR image to RGB before processing.
	  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	  # Print handedness and draw hand landmarks on the image.
	  print('handedness:', results.multi_handedness)
	  if not results.multi_hand_landmarks:
	    continue
	  annotated_image = image.copy()
	  for hand_landmarks in results.multi_hand_landmarks:
	    print('hand_landmarks:', hand_landmarks)
	    #print("THE",hand_landmarks)
	    file1 = open("myfile.txt","w")#append mode 
	    file1.write(str(hand_landmarks)) 
	    file1.close() 
	    mp_drawing.draw_landmarks(
		annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
	  print(cv2.imwrite(
	      '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(image, 1)))

	  cv2.imshow('annotated_image', annotated_image) 

		#waits for user to press any key  
		#(this is necessary to avoid Python kernel form crashing) 
	  cv2.waitKey(0)  

		#closing all open windows  
	  cv2.destroyAllWindows()  
	hands.close()

	#x_px = min(math.floor(normalized_x * image_width), image_width - 1)
	#y_px = min(math.floor(normalized_y * image_height), image_height - 1)

	'''
	# For webcam input:
	hands = mp_hands.Hands(
	    min_detection_confidence=0.7, min_tracking_confidence=0.5)
	cap = cv2.VideoCapture(-1)
	while cap.isOpened():
	  success, image = cap.read()
	  if not success:
	    break

	  # Flip the image horizontally for a later selfie-view display, and convert
	  # the BGR image to RGB.
	  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
	  # To improve performance, optionally mark the image as not writeable to
	  # pass by reference.
	  image.flags.writeable = False
	  results = hands.process(image)

	  # Draw the hand annotations on the image.
	  image.flags.writeable = True
	  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	  if results.multi_hand_landmarks:
	    print(results.multi_hand_landmarks)
	    for hand_landmarks in results.multi_hand_landmarks:
	      mp_drawing.draw_landmarks(
		  image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
	  cv2.imshow('MediaPipe Hands', image)
	  if cv2.waitKey(5) & 0xFF == 27:
	    break
	hands.close()
	cap.release()'''


	# READ TXT FILE
	landmarks_x=[]
	landmarks_y=[]
	with open('myfile.txt', 'r+') as f: #r+ does the work of rw
		lines = f.readlines()
		for i, line in enumerate(lines):
			if line.startswith('  x'):
			    lines[i] = lines[i].strip()
			    x= lines[i]
			    x= x[3:]
			    landmarks_x.append(x)
			    print(lines[i])
			    print(x)
			if line.startswith('  y'):
			    lines[i] = lines[i].strip()
			    y= lines[i]
			    y= y[3:]
			    landmarks_y.append(y)
			    print(lines[i])
			    print(y)
		f.seek(0)


	# NORMALIZING COORDINATES MATH
	import math
	import cv2
	image = cv2.imread("/home/lamia/Downloads/Webp.net-resizeimage (1).jpg")
	print(type(image))
	# <class 'numpy.ndarray'>
	print(image.shape)
	height, width, col = image.shape
	print('height: ', height)
	print('width: ',width)
	print(type(image.shape))
	# 1067, 1599
	#image_width =1599
	#image_height = 1067
	x_px = []
	y_px = []
	#convert normalized coordinated to their pixle form 
	#store each pixle form in its own list either x or y
	for i in range(21):

	    normalized_x= float(landmarks_x[i])   
	    normalized_y= float(landmarks_y[i])
	    xx_px = min(math.floor(normalized_x * width), width - 1)
	    x_px.append(xx_px)
	    yy_px = min(math.floor(normalized_y * height), height - 1)
	    y_px.append(yy_px)
	    print('x: ',int(xx_px))
	    print('y: ',int(yy_px ))


	#flip image horizantaly
	image = cv2.flip(image, 1)
	for i in range(21):
	    cv2.circle(image, (x_px[i], y_px[i]), 10, (0,0,255), -1)
	cv2.imshow('drawing pixle landmarks', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()     



	# CROP FINGERPRINT
	img = image.copy()
	rop_img = img[y_px[8]-40:y_px[7]+10, x_px[8]-40:x_px[7]+50]
	#cv2.rectangle(img,(x_px[8]+5,y_px[8]+5),(x_px[8]+width,y_px[8]+height),(255,0,0),2)
	# 3-4-12-
	crop_img = cv2.rectangle(img,(x_px[8]-40,y_px[8]-40),(x_px[7]+50,y_px[7]+2),(255,0,0),2)
	cv2.imshow("cropped", rop_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return 'Hello, World!'





