import os
import sys

data_dir = "./Data/"

def capture_new_user(name):
	N = 10
	
	for i in range(N):
		outfile = data_dir + name + "/" + name + str(i) + ".bmp"
#		#res = os.system("raspistill -o " + outfile)
		print("captured image " + outfile)

if __name__ == "__main__":
	name = sys.argv[1]
	capture_new_user(name)


