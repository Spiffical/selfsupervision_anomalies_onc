from PIL import Image
import os


def crop_image(image_path, saved_location, box = (107,66,1040,810)):
    
    #Split the file path based on the file-system:
    if os.name == 'posix':
        delim = '/'
    elif os.name == 'nt':
        delim = '\\'

    image_obj = Image.open(image_path)
    area = image_obj.crop(box)
    image_name = image_path.split(delim)[-1]
    area.save(saved_location + image_name)
    # print ("Image {} cropped & saved to path {}.".format(image_path, saved_location + image_name))