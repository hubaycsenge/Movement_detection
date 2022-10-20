

# import the necessary packages

#import argparse

import cv2

import os

import csv


'''parser = argparse.ArgumentParser()

parser.add_argument("folder", help="Path to the images")

args = parser.parse_args()'''

image_folder = 'correld_imgs_try_03_10/do' #args.folder​

topleft_x = 0

topleft_y = 0

bottomright_x = 0

bottomright_y = 0

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

        topleft_x = x

        topleft_y = y 

        cropping = True

 

    # check to see if the left mouse button was released

    elif event == cv2.EVENT_LBUTTONUP:

        bottomright_x = x

        bottomright_y = y

        cropping = False


        cv2.rectangle(image, (topleft_x, topleft_y), (bottomright_x, bottomright_y), (0, 255, 0), 2)

        cv2.imshow("image", image)

def save_data(final):
	with open('image_data_20_batch.csv','w') as f:
                
		for item in final:

			print('writing {}'.format(item))

			f.write(','.join(item))

			f.write('\n')

#[["PATIENTID", "TOPLEFT_X", "TOPLEFT_Y", "BOTTOMRIGHT_X", "BOTTOMRIGHT_Y", "DIRECTION"]]

final = [["FNAME", "TOPLEFT_X", "TOPLEFT_Y", "BOTTOMRIGHT_X", "BOTTOMRIGHT_Y","VIS"]]

for i, image in enumerate(os.listdir(image_folder)):

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

        if key == ord("l"):

            char = "l"

            break
        elif key == ord("n"):

            char = "n"

            break
        elif key == ord("u"):
            cv2.imshow("image",copy)
        elif key == ord("r"):

            char = "rossz"
			
            break
        elif key == ord("s"):
        	save_data(final)
        

    data_for_image.append(str(topleft_x))

    data_for_image.append(str(topleft_y))

    data_for_image.append(str(bottomright_x))

    data_for_image.append(str(bottomright_y))

    data_for_image.append(str(char))

    final.append(data_for_image)


    cv2.destroyAllWindows()


with open('image_data_20_batch.csv','w') as f:

    for item in final:

        print('writing {}'.format(item))

        f.write(','.join(item))

        f.write('\n')


