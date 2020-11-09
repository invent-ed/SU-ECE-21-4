import numpy as np
import cv2
import classfunction
import glob

def init_Recognition(image_source, template_source,paths):
    'Used to initalize a recongition object for each template/image pair'
    # # TODO: create a function that verifies the image and template names match

    rec_list = []
    count = 0

    # add images and templates in a parallel for-loop
    for i in glob.iglob(str(image_source)):

        # add new Recognition object to list
        rec_list.append(classfunction.Recognition())

        # add image title and image to object
        image = cv2.imread(i)
        
        rec_list[count].add_image(i, image)
        #rec_list[count].add_template(rec_list, template_source)

        filter_images(image,i,str(paths['edited_photos']))

        rec_list[count].add_title_chars(i)

        # increment count
        count = count + 1

    # return the list of recognition objects
    return rec_list

################################################################################
def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)
################################################################################
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the variance
    # of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
################################################################################
def histogram_equalization(image):
    gray =  cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hist_image = cv2.equalizeHist(gray)

    print("done with equalization")
    return hist_image
################################################################################  
def edge_sharpening(image):
    kernel = np.array([[-1,-1,-1,],[-1,9,-1],[-1,-1,-1]])
    sharp_image = cv2.filter2D(np.asarray(image), -1, kernel)
    
    return sharp_image
################################################################################


def filter_images(primary_image,image_source,edited_source):
  
    img_yuv = cv2.cvtColor(primary_image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)


    lut_u, lut_v = make_lut_u(), make_lut_v()
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    
    
    min = 125
    max = 130
    i = 530
    #Only checking mid range of each photo for variance from YUV range
    while(i < 550):
        if(u[i][1][1] < 128 or u[i][1][1]>128):

            #Blur score 
            gray = cv2.cvtColor(primary_image, cv2.COLOR_BGR2GRAY)
            threshold = variance_of_laplacian(gray)

            if(threshold < 1500):
                print("not blurry")
            else:
                print("blurry")
        else:
            flag = 1
            print("night")

            save_image = np.copy(primary_image)
            image_path = image_source
            image_path = image_path.split("\\")

            num = image_path.index('images')
            # Writing original image to folder and editing copy
            edited_source = edited_source + "\\" + image_path[num+1]
            cv2.imwrite(edited_source,primary_image)
            
            sharp_image = edge_sharpening(save_image)
            hist_image = histogram_equalization(sharp_image)
            cv2.imwrite(image_source,hist_image)
            
            #sharp_image = edge_sharpening(normal_img)
            
            # cv2.imshow("final image", np.asarray(hist_image))
            # cv2.waitKey(0)
            break
        i += 1

################################################################################

def crop(event, x, y, flags, param):

    global ref_points, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:

        ref_points.append((x, y))
        cropping = False

        cv2.rectangle(param, ref_points[0], ref_points[1], (0, 255, 0), 2)
        cv2.imshow("image", param)

def add_cat_ID(rec_list, cluster_path):

    # create the list
    import pandas as pd
    csv_file = pd.read_csv(cluster_path)
    image_names = list(csv_file['Image Name'])
    cat_ID_list = list(csv_file['Cat ID'])

    for count in range(len(rec_list)):
        image = os.path.basename(rec_list[count].image_title)
        try:
            image_index = image_names.index(image)
        except ValueError:
            print('\tSomething is wrong with cluster_table file. Image name is not present.')

        cat_ID = cat_ID_list[image_index]
        rec_list[count].add_cat_ID(cat_ID)

    return rec_list