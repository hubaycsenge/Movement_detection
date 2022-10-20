

# import the necessary packages

#import argparse

import cv2

import os

import csv


'''parser = argparse.ArgumentParser()

parser.add_argument("folder", help="Path to the images")

args = parser.parse_args()'''

image_folder = 'correld_imgs_try_03_10/do' #args.folder​

topleft_x = []

topleft_y = []

bottomright_x = []

bottomright_y = []

cropping = False


def click_and_crop(event, x, y, flags, param):

    global topleft_x

    global topleft_y

    global bottomright_x

    global bottomright_y

    global cropping

 

    # if the left mouse button was clicked, record the starting

    # (x, y) coordinates and indicate that cropping is being performed

    if event == cv2.EVENT_LBUTTONDOWN:

        topleft_x.append(x)

        topleft_y.append(y) 

        cropping = True

 

    # check to see if the left mouse button was released

    elif event == cv2.EVENT_LBUTTONUP:

        bottomright_x.append(x)

        bottomright_y.append(y)

        cropping = False
        zipped = zip(topleft_x,bottomright_x,topleft_y,bottomright_y)
        for tlx,brx,tly,bry in zipped:
        	cv2.rectangle(image, (tlx, tly), (brx, bry), (0, 255, 0), 2)

        cv2.imshow("image", image)

def save_data(final):
	with open('people_detection_csvs/error_correction.csv','w') as f:
                
		for item in final:

			print('writing {}'.format(item))

			f.write(','.join(item))

			f.write('\n')

#[["PATIENTID", "TOPLEFT_X", "TOPLEFT_Y", "BOTTOMRIGHT_X", "BOTTOMRIGHT_Y", "DIRECTION"]]

final = [["FNAME", "TOPLEFT_X", "TOPLEFT_Y", "BOTTOMRIGHT_X", "BOTTOMRIGHT_Y"]]

for i, image in enumerate(os.listdir(image_folder)):

	topleft_x = []
	topleft_y = []
	bottomright_x = []
	bottomright_y = []
	data_for_image = []
	data_for_image.append(image)
	path_to_image = os.path.join(image_folder, image)
	image = cv2.imread(path_to_image)

	cv2.namedWindow("image")

	cv2.setMouseCallback("image", click_and_crop)

	while True:
		copy = image
		cv2.imshow("image", image)

		key = cv2.waitKey(1) & 0xFF

		if key == ord("n"): #NEXT
			break
          
		elif key == ord("u"): #UNDO
			image = copy
			topleft_x = []
			topleft_y = []
			bottomright_x = []
			bottomright_y = []
			cv2.imshow("image",image)
		
		elif key == ord("p"): #POP
			topleft_x.pop()
			topleft_y.pop()
			bottomright_x.pop()
			bottomright_y.pop()
			zipped = zip(topleft_x,bottomright_x,topleft_y,bottomright_y)
			image = copy
			for tlx,brx,tly,bry in zipped:
				cv2.rectangle(image, (tlx, tly), (brx, bry), (0, 255, 0), 2)
			cv2.imshow("image",image)
		
            
		elif key == ord("r"): #ERROR
			print(f'ERROR with image {image}')	
			break
            
		elif key == ord("s"): #SAVE
			save_data(final)
        

	data_for_image.append(' '.join(str(i) for i in topleft_x))

	data_for_image.append(' '.join(str(i) for i in topleft_y))

	data_for_image.append(' '.join(str(i) for i in bottomright_x))

	data_for_image.append(' '.join(str(i) for i in bottomright_y))

	final.append(data_for_image)


	cv2.destroyAllWindows()


with open('people_detection_csvs/error_correction.csv','w') as f:

	for item in final:

		print('writing {}'.format(item))

		f.write(','.join(item))

		f.write('\n')


