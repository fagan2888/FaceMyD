import cv2
import sys

def capture_user(name):

	cap = cv2.VideoCapture(0)

	N = 10
	for i in range(N):
		ret, frame = cap.read()
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
		cv2.imshow('frame', rgb)
		filename = "./Data/" + name + "/" + name + str(i) + ".jpg"
		print("Saving image" + filename)
		out = cv2.imwrite(filename, frame)

	cap.release()
	cv2.destroyAllWindows()



if __name__ == "__main__":
	
	name = sys.argv[1]
	
	capture_user(name)	

