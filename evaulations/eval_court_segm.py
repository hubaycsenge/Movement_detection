
csv_fname = 'test_annot_batch_2'
csv_fpath = 'annot_people/csv'

# import the necessary packages

#import argparse

import cv2

import os

import csv


'''parser = argparse.ArgumentParser()

parser.add_argument("folder", help="Path to the images")

args = parser.parse_args()'''

image_folder = 'annot_people/do'

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


        cv2.rectangle(image, (topleft_x[-1], topleft_y[-1]), (bottomright_x[-1], bottomright_y[-1]), (0, 255, 0), 2)

        cv2.imshow("image", image)

def save_data(final,fpath,fname):
    if fname + '.csv' in os.listdir(fpath):
        fname = fname + '_v2'
        save_data(final,fpath,fname)
        print('WARNING, U almost oversaved, dumb bitch')

    else:
        with open(os.path.join(fpath,fname+'.csv'),'w') as f:
                    
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

        if key == ord("n"):
            print(f"Done with image #{i}")
            break

        elif key == ord("u"):
            topleft_x = []
            topleft_y = []
            bottomright_x = []
            bottomright_y = []

        elif key == ord("p"):
            topleft_x.pop()
            topleft_y.pop()
            bottomright_x.pop()
            bottomright_y.pop()

        elif key == ord("s"):
        	save_data(final,csv_fpath,csv_fname)
        

    data_for_image.append(' '.join(str(topleft_x)))

    data_for_image.append(' '.join(str(topleft_y)))

    data_for_image.append(' '.join(str(bottomright_x)))

    data_for_image.append(' '.join(str(bottomright_y)))

    final.append(data_for_image)


    cv2.destroyAllWindows()


save_data(final,csv_fpath,csv_fname)


