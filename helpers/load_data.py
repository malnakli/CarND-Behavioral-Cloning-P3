from sklearn.utils import shuffle
import cv2
import numpy as np


def resize_img_square(img,dim):
    """resize img to have square shape e.g.(200,200).
    Arguments:
        img: np array image
        dim: int 
    Returns:
       resize_image
    """
    res = cv2.resize(img,(dim,dim), interpolation = cv2.INTER_CUBIC)

    return res

# Taken from the first project
def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    imshape = img.shape
    vertices = np.array(
    [[
        (0,imshape[0]),
        (0, imshape[0]*.75),
        (imshape[1]*.45,imshape[0]*.5),
        (imshape[1]*.55,imshape[0]*.5),
        (imshape[1],imshape[0]*.75), 
        (imshape[1],imshape[0])
        
    ]], dtype=np.int32)
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def preprocess_image(image_path, FLAGS, gray=False):
    """pre processes image for training.
    Arguments:
        image_path: the full or relative path of the image 
        flip: List of folders names
        gray: Before enable this see if the model support it
    Returns:
       image, image_flipped if flip is True
    """
    image = None
    image_flipped = None
    if gray :
        image = cv2.imread(image_path, 0) # shape (160, 320)
        image = np.reshape(image, image.shape[:] + (1,))   # shape (160, 320, 1)

    else :
        image = cv2.imread(image_path) # shape (160, 320,3)
    
    image = region_of_interest(image) # (160,320)
    image = resize_img_square(image,200) # shape (200,200)

    if FLAGS.flip_img:
        image_flipped = cv2.flip(image, 1)
    
    return image,image_flipped
    

def load_data(samples,FLAGS):
    images = []
    steerings = []
    for index, row in samples.iterrows():
        # create adjusted steering measurements for the side camera images
        corrections = [0, 0.3, -0.2]
        for header, correction in zip(samples.columns[:FLAGS.img_use], corrections[:FLAGS.img_use]):
            
            steering = float(row['steering'])
            image , image_flipped =  preprocess_image(row[header],FLAGS,gray=False)
            if FLAGS.flip_img:
                images.append(image_flipped)
                steerings.append(-(steering + correction))

            images.append(image)
            steerings.append(steering + correction)
    # convert to numpy array since this what keras required

    return shuffle(np.array(images), np.array(steerings))


# @samples: panads dataFrame
def load_data_generator(samples,FLAGS, batch_size=32):
    num_samples = samples.shape[0]
    while 1:  # Loop forever so the generator never terminates
        # The reason this should work is that over many epochs,
        # random selection should ensure that all of your training data is taken into account.
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            yield load_data(batch_samples,FLAGS)
