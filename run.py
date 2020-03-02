#%%writefile C:\Users\DataAnalyst\Desktop\OFFLINE_DEPLOY\ANPR_OFFLINE_VER2\src\score.py

from PIL import Image
import numpy as np
from azureml.core.model import Model
import base64
import io
import os
import json
import tensorflow as tf

import json
import requests
import cv2
import io


# importing Libraries

import re
import time
import threading
import queue
import base64

# from azureml.contrib.services.aml_request  import AMLRequest, rawhttp
# from azureml.contrib.services.aml_response import AMLResponse
#from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
#from azure.cognitiveservices.vision.customvision.prediction import *

import urllib.request
import urllib.parse
from urllib.request import urlopen

import math
import json
from io import BytesIO

import linecache
import sys



from object_detection import ObjectDetection
from predict import TFObjectDetection,predict_image,predict_url

# Azure ML model loader
def init():
    
    global char_to_numbers
    global state_codes
    global graph_def
    global MODEL_FILENAME 
    global od_model
    global labels


    # Common OCR errors where numbers are processed as characters.
    char_to_numbers = {
        'A' : 4,
        'B' : 8,
        'D' : 0,
        'G' : 6,
        'I' : 1,
        'T' : 7,
        'J' : 1,
        'O' : 0,
        'Q' : 0,
        'S' : 5,
        'Z' : 2
    }

    # Valid Indian state codes.
    state_codes= {"AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JK", "JH",
                  "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ",
                  "PY", "SK", "TN", "TS", "TR", "UA", "UK", "UP", "WB", "AN", "CH",
                  "DN", "DD", "DL", "LD"}

    for i in range(10):
        char_to_numbers[str(i)] = i # Add numbers to the dictionary.

    
    
    
    
    
    
    
    filename = 'model.pb'
    labels_filename = 'labels.txt'
    
    #def initialize():
    print('Loading model...', end='')
    #model_path = Model.get_model_path(model_name='anpr-predictor')
    model_path = r'C:\Users\DataAnalyst\Desktop\OFFLINE_DEPLOY\ANPR_OFFLINE_VER2\anpr'
    
    graph_def = tf.compat.v1.GraphDef()
    
    
    with open(os.path.join(model_path, filename), 'rb') as f:
        graph_def.ParseFromString(f.read())
    print('Success!')
    

    print('Loading labels...', end='')
    with open(os.path.join(model_path, labels_filename), 'rt') as f:
        labels = [l.strip() for l in f.readlines()]
    print("{} found. Success!".format(len(labels)))
    
    global od_model
    od_model = TFObjectDetection(graph_def, labels)

# Helper to predict an image encoded as base64


# Azure ML entry point
#def run(json_input):
#     try:
#         results = None
#         input = json.loads(json_input)
#         url = input.get("url", None)
#         image = input.get("image", None)

#         if url:
#             results = predict_url(url)
#         elif image:
#             results = predict_image_base64(image)
#         else:
#             raise Exception("Invalid input. Expected url or image")
#         return (results)
#     except Exception as e:
#         return (str(e))

    # note you can pass in multiple rows for scoring
def run(request):
    
   
    try:
        #if request.method == 'POST':
        if True:
            request_data = json.loads(request)
            results=[]
            length=len(request_data)
            if(length==2):
 
                front_base64 = (request_data['FrontCamera'].encode('utf-8'))
                back_base64 = (request_data['BackCamera'].encode('utf-8'))

                front_url_binary = base64.decodebytes(front_base64)
                back_url_binary = base64.decodebytes(back_base64)

                que = queue.Queue()
                threads_list = list()

                t1 = threading.Thread(name="FrontCamera",target=lambda q, arg1: q.put(GetInfo(arg1)), args=(que, front_url_binary))
                t1.start()
                threads_list.append(t1)

                t2 = threading.Thread(name="BackCamera",target=lambda q, arg1: q.put(GetInfo(arg1)), args=(que, back_url_binary))
                t2.start()
                threads_list.append(t2)
                
            # Join all the threads
                for t in threads_list:
                    t.join()
            
            # Check thread's return value
                results=[]
                while not que.empty():
                    result = que.get()
                    results.append(result)
                    
            elif(length == 0):
                print("No Image available!")
                
            elif(length == 1):
                for key in request_data.keys():
                    data = (request_data[key].encode('utf-8'))
                    data_binary = base64.decodebytes(data)
                    result=GetInfo(data_binary)
                    result['got_from'] = key
                    results.append(result)
            
                          
            output={}
            output2={}
            
            
            if len(results)==1:
                plate0 = results[0]['plate']
                #print(plate0)
                if  plate0 in ('', '_NOPLATE_'):
                   output2 = results[0]
                   
                else:
                    output=results[0]
                    #print('plate found in {} {}'.format(results[0]['got_from'], results[0]['plate']))
                                
                    
            # If Both Cameras are present 
            else:
                
                plate0 = results[0]['plate']
                plate1 = results[1]['plate']
                #print(plate0,plate1)
   
                if (plate0 not in ['', '_NOPLATE_']) or (plate1 not in ['', '_NOPLATE_']):
                    item = validate_plate(results)
                    if item['plate'] not in ('', '_NOPLATE_'):
                        output = item
                        #print('plate found in {} {}'.format(item['got_from'], item['plate']))
                           
                    else:
                        output2 = item
                
    
            
               
            
            if(len(output) != 0):
                #print("*******************OutPUT1******************************************")
                #print('Before Correction Plate_Number: ',output['plate'])
                #print('Location: ',output['info'])
                #print('Camera: ',output['got_from'])
                #print('string_img: ',output['plate_image'])
                
                output['plate'] = state_correction(output['plate'])
                #print('After Correction Plate_Number: ',output['plate'])
                output = json.dumps(output)
                #print("Returning OUTPUT1 as JSON")
                #print(output)
                
                
                #return AMLResponse(output, 200)
                return output

            else:
                #print("**************************OUTPUT2************************************************")
                #print("Returning OUTPUT2 as JSON", output2)
                output2 = json.dumps({
                                         "plate" : "_NOPLATE_",
                                         "info": "-2",
                                         "got_from":"",
                                         "plate_image":""
                                         })
                #print(output2)
                #return AMLResponse(output2, 200)

                return output2

    except:
                e = PrintException()
                response_dump =         {
                                         "plate" : "",
                                         "info": "-1," + str(e),
                                         "got_from":"",
                                         "plate_image":""
                                         }
                #print(response_dump)
                #return AMLResponse(response_dump, 200)
                return json.dumps(response_dump)
                
                
def validate_plate(results):
    
    if is_number_plate(results[0]['plate']) and is_number_plate(results[1]['plate']):
        if (results[0]['plate']==results[1]['plate']):
            results[0]['got_from'] = 'FrontCamera'
            return results[0]
        else:
            # Check for prob and return max(p1,p2)
            return results[0]
        
    elif is_number_plate(results[0]['plate']):
        return results[0]
    
    elif is_number_plate(results[1]['plate']):
        
        return results[1]
        
    else:
        
        if (results[0]['plate']==results[1]['plate']):
                results[0]['got_from'] = 'FrontCamera'
                return results[0]
            
        else:
            # Check for prob and return max(p1,p2)
            if results[1]['plate'] not in ('', '_NOPLATE_'):
                return results[1]
            else:
                results[0]['got_from'] = 'FrontCamera'
                return results[0]
                
        
        
 ############################################ CUSTOM LOGIC ######################################################################

def GetInfo(image1):
            b = image1
            
            res1 = main(image1, (0, 0, 0, 0))
            
            correct = process(res1[0])[0]
            
            img_string=[]
            thread=(threading.currentThread().getName())
            if res1[0] == "__NOPLATE__":
                response_dump =  {
                                         "plate" : "_NOPLATE_",
                                         "info": "-2" ,
                                         "got_from":thread,
                                         "plate_image":""
                                         }
                return response_dump

            elif is_number_plate(correct):
                bbox = ','.join([str(x) for x in res1[1]])
                #bASE64

                bboxN=res1[1]
                img_string=getBase64(image1,bboxN,img_string)

                response_dump ={
                                 "plate" : correct,
                                 "info": "0," + bbox,
                                 "got_from": thread,
                                 "plate_image": img_string.decode('utf-8')
                               }
                return response_dump

            else:
                attempts = 2
                cor = correct
                bbox = res1[1]

                while attempts > 0 and not is_number_plate(process(cor)[0]):
                    time.sleep(1)
                    corrections= (check_state(cor), check_number(cor),
                                   check_up(cor), check_down(cor))
                    roi_image = extract_roi(bbox, b, corrections)
                    img = roi_image[:, :, ::-1]
                    img = preprocessbeforeocr(img)
                    success, encoded_image = cv2.imencode('.jpg', img)
                    encoded_img = encoded_image.tobytes()
                    
                    result = detect_text_azure(encoded_img)

                    numplate = ""
                    for rres in result['recognitionResult']['lines']:
                        numplate = numplate + " "+ rres['text']
                    cor = process(numplate)[0]
                    
                    attempts -= 1
                    for i in range(4):
                        bbox[i + 1] += evaluate_corrections(corrections)[i]

                bbox = ','.join([str(x) for x in bbox])
                bboxN=res1[1]

                img_string=getBase64(image1,bboxN,img_string)
                
                #print("Returning JSON:")
                response_dump = {
                                 "plate" : str(cor),
                                 "info": "1," + bbox,
                                 "got_from":thread,
                                 "plate_image": img_string.decode('utf-8')
                                }

                return response_dump


def main(binaryImage, corrections): 
    ## prediction bbox
    #print("Binary image :",type(binaryImage))
   
    prediction_bbox = GetPredictionResults(binaryImage)
    
    #roi image
    if len(prediction_bbox) == 0:
        return "__NOPLATE__", [0,0,0,0]
    
    else :
        roi_image = extract_roi(prediction_bbox,binaryImage,
                                evaluate_corrections(corrections))

        #convert to opencv formate of numpy matrix
        img = roi_image[:, :, ::-1]

        #send to preprocessing
        img = preprocessbeforeocr(img)

        #encode and extract the bytearray
        success, encoded_image = cv2.imencode('.jpg', img)
        encoded_img = encoded_image.tobytes()

        #send to ocr
        result = detect_text_azure(encoded_img)

        #instantiate the numberplate object
        numplate = ""

        #extract the number plate out and return
        for rres in result['recognitionResult']['lines']:
            numplate = numplate + " "+ rres['text']
            
        #print(numplate, prediction_bbox)
        return numplate, prediction_bbox



def getBase64(image1,bbox,img_string):

    im = Image.open(BytesIO(image1)).convert('RGB')
    np_image = np.array(im)

    # Convert RGB to BGR
    crop_img = np_image[ bbox[3] :bbox[4],bbox[1] :bbox[2]]
    im_cr = Image.fromarray(crop_img)
    
    # Buffer array to store Image in memory
    output = BytesIO()
    saved = im_cr.save(output, format='JPEG')
    encoded_string = base64.b64encode(output.getvalue())

    return encoded_string



def evaluate_corrections(corrections):
    final = [0, 0, 0, 0]
    if corrections[0] == -1:
        final[0] -= 8
    elif corrections[0] == 1:
        final[0] += 8
    if corrections[1] == -1:
        final[1] -= 8
    elif corrections[1] == 1:
        final[1] += 8
    if corrections[2] == -1:
        final[2] -= 8
    elif corrections[2] == 1:
        final[2] += 8
    if corrections[3] == -1:
        final[3] -= 8
    elif corrections[3] == 1:
        final[3] += 8
    return tuple(final)

###################################################
##   Get prediction Results
###################################################
def GetPredictionResults(binaryImage):
    #config_vars
    threshold = 0.2

    
    image = Image.open(io.BytesIO(binaryImage))
    results = predict_image(image,od_model)

    image = Image.open(BytesIO(binaryImage),)
    size = image.size
    Bbox = []
    
    #take the one with the highest probability
    for item in results['predictions']:
        if(item['probability'] > threshold):
            xmin = item['boundingBox']['left'] * (size[0]-5) 
            ymin = item['boundingBox']['top'] * (size[1]+5)
            xmax = (item['boundingBox']['width'] * (size[0]+5)) + xmin
            ymax = (item['boundingBox']['height'] * (size[1]-5)) + ymin
            Bbox =  [ item['probability'], math.ceil(xmin), int(xmax), math.ceil(ymin), int(ymax) ]
            #print('bbox :',Bbox)
            #print(item.probability)
            threshold = item['probability']

    #print(Bbox)
    return Bbox #send the one with highest probability


##################################
#   image resize
##################################

def image_resize(image):
    
    #global variables
    width = None
    height = 100
    inter = cv2.INTER_AREA
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
        
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions6
        r = width / float(w)
        dim = (width, int(h * r))
    
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

##########################
#### Extract ROI
##########################

def extract_roi(Bbox, imagebinary,corrections):
    # Opening Image from bytesarray as Image object 
    im = Image.open(BytesIO(imagebinary)).convert('RGB')
    np_image = np.array(im)
    
    # cropping image using index and saving in CROP_IMG
    crop_img = np_image[ Bbox[3] + corrections[0]:Bbox[4] + corrections[1],
                        Bbox[1] + corrections[2]:Bbox[2] + corrections[3]]
    
    # Returning Cropping the IMAGE
    return crop_img


###############################################################################
#   preprocessing images before OCR
###############################################################################
def preprocessbeforeocr(img):
    #resize
    image = image_resize(img)

    #denoising  -  src[, dst[, h[, templateWindowSize[, searchWindowSize]]]]
    image_denoised = cv2.fastNlMeansDenoising(image,None,15,7,21)

    #erosion and dialation - opening
    kernel = np.ones((2,2),np.uint8)
    image_opening = cv2.morphologyEx(image_denoised, cv2.MORPH_CLOSE, kernel)

    return image_opening


def state_correction(plate):
    dict1 = {'IN':'TN','TH':'TN', 'IM':'TN', 'IH':'TN','TM':'TN', 'MA':'MH' }
    
    if plate[:2] in dict1.keys():
        plate = dict1[plate[:2]]+plate[2:]
    
    return plate
    

def detect_text_azure(databytearray):
    ''' Takes image as input  in bytearray i.e binary format
    and calls the amureML api
    
    return the response object containing TEXT'''
    
    
    # Request headers
    headers = {'Content-Type': 'application/octet-stream',
               'Ocp-Apim-Subscription-Key': '9afc6fce426a4fafb9d82e428beab151',}

    headers2 = { 'Ocp-Apim-Subscription-Key': '9afc6fce426a4fafb9d82e428beab151', }

    response = requests.post('https://southeastasia.api.cognitive.microsoft.com/vision/v2.0/recognizeText?mode=Printed', data=databytearray, headers=headers)

    while True:
        response2 = requests.get(response.headers['Operation-Location'],  headers=headers2)
        data = json.loads(response2.content.decode('utf8'))
        
        if(data['status'] == 'Failed'):
            return (data)
            

        elif(data['status'] == 'Succeeded'):
            return (data)
            



def check_down(number_plate):
    """
    Checks if the lower part of the number-plate is recognized. This function\
    is useful for number-plates split into two lines, one above the other.
    
    Parameters:
        number_plate (str) : Number plate

    Returns:
        int : -1 : Lower edge of bounding box to be shrunk upwards
               0 : Lower edge of bounding-box unchanged.
               1 : Lower edge of bounding-box expanded downwards.
    """
    if not number_plate:
        return 1
    try:
        if number_plate[-6:-4].isalpha() and number_plate[-4:].isnumeric():
            return 0
        else:
            return 1
    except IndexError:
        return 1



def check_up(number_plate):
    """
    Checks if the upper part of the number-plate is recognized. This function
    is useful for number-plates split into two lines, one above the other.

    Parameters:
        number_plate (str) : Number plate

    Returns:
        int : -1 : Upper edge of bounding box to be expanded upwards
               0 : Upper edge of bounding-box unchanged.
               1 : Upper edge of bounding-box shrunk downwards.
    """
    if not number_plate:
        return -1
    try:
       if number_plate[:2] in state_codes and number_plate[2:4].isnumeric():
            return 0
       elif (number_plate[:2].isalpha() and not number_plate[:2] in state_codes and not number_plate[2:6].isnumeric()):
            return 1
       else:
            return -1
    except IndexError:
        return -1



def check_number(number_plate):
    """
    Checks the 4-digit number of the number plate and suggests corrections
    to the right-edge of the OCR bounding box.

    Parameters:
        number_plate (str) : Number-plate

    Returns:
        int : -1 : Right edge of bounding box to be shrunk to the left
               0 : Right edge of bounding-box unchanged.
               1 : Right edge of bounding-box expanded to the right.
    """
    if not number_plate:
        return 1
    try:
        if len(number_plate) == 4:
            if number_plate[:2].isalpha() and number_plate[2:].isnumeric():
                return 1
        if len(number_plate) == 6:
            if number_plate[:2].isalpha() and number_plate[2:].isnumeric():
                return 0

        if number_plate[-4:].isnumeric():
            return 0
        elif (number_plate[-1].isalpha() or number_plate[-3:].isnumeric()
          or number_plate[-2:].isnumeric() or number_plate[-1].isnumeric()):
            return 1
        else:
            return -1
    except IndexError:
        return 1



def check_state(number_plate):
    """
    Checks if the state-code in the number_plate is valid or not,
    followed by suggesting corrections to the left-edge of the OCR
    bounding box.

    Parameters:
        number_plate (str) : Number plate
    Returns:
        int : -1 : Bounding box to be expanded to the left
               0 : Left edge of bounding-box unchanged.
               1 : Left edge of bounding-box shrunk to the right.
    """
    if not number_plate:
        return -1
    try:
        if len(number_plate) == 4:
            if number_plate[:2].isalpha() and number_plate[2:].isnumeric():
                return 0
        if len(number_plate) == 6:
            if number_plate[:2].isalpha() and number_plate[2:].isnumeric():
                return -1

        if number_plate[:2] in state_codes:
            return 0
        elif number_plate[0].isalpha() and number_plate[1].isnumeric():
            return -1
        elif number_plate[0].isnumeric():
            return -1
        else:
            return 1
    except IndexError:
        return -1



def is_number_plate(number_plate):
    # Checks if the input text is a valid number_plate or not.
    if re.search(r"^[A-Za-z]{2}[0-9]{2}[A-Za-z]{0,2}[0-9]{4}$", number_plate):
        return True
    else:
        return False

def process(text):
    """
    Returns final processed number plate given a raw OCR tt.

    Parameters:
        text (str) : Raw OCR output.
    Returns:
        str : Final processed output.
    """
    plate = clean_number_plate(text)
    if len(plate) < 8:
        return [re.sub(r"[^0-9a-zA-Z]", "", text)]

    width = len(plate) - 8
    indices = get_indices(plate, width)
    plate = correct_number(plate, indices, width)
    plate = correct_area_code(plate, indices)
    return get_number_plates(plate, width)



def get_number_plates(text, width):
    # Returns the final number_plate
    if len(text) < 8 + width:
        return []
    elif len(text) == 8 + width:
        return [text]
    else:
        return [text[:4 + width] + text[-4:], text[:4] + text[-(4 + width):]]


def correct_area_code(text, indices):
    """
    Returns number-plate with fixes, if any,
    to the numbers succeeding the state code.
    
    Parameters:
        text (str)      : Number plate
        indices (list)  : List of indices

    Returns:
        str : Corrected number-plate.
    """
    idx = len(indices) - 1
    while indices[idx] < 4:
        idx -= 1
    l = indices[idx]

    while l >= 2:
        if all(char in char_to_numbers for char in text[l - 2:l]):
            return (text[:l - 2]
                    + "".join(str(char_to_numbers[char])
                              for char in text[l - 2:l])
                    + text[l:])
        l -= 1

    return text


def correct_number(text, indices, width):
    """
    Returns the number-plate string with corrections made to the 4-digit number, if any.
    
    Example: TN01AB1QI4 will be modified to TN01AB1014
    
    Parameters:
        text (str)      : Number plate
        indices (list)  : List of aforementioned indices
        width (int)     : Number of letters preceding the 4-digit number.
    Returns:
        str : Corrected number-plate.
    """
    r = indices[0] + width
    while r < len(text) - 3:
        if all(char in char_to_numbers for char in text[r:r + 4]):
            return text[:r] + "".join(
                str(char_to_numbers[char]) for char in text[r:r + 4]) + text[r + 4:]
        r += 1

    return text



def get_indices(text, width):
    """
    Returns a list of all possible indices of the two-letters before the
    4-digit number.

    For example, consider TN-01-AAB-1234. The two letters preceding the number
    could be AA or AB. The output will therefore be [4,5], where 4 and 5 are
    the indices of the start of "AA" or "AB" in the number plate. Currently,
    the first index is chosen for further processing downstream.

    Parameters:
        text (str)  : Number plate
        width (int) : Number of letters preceding the 4-digit number.
                      Typically in the range [0-2].
    Returns:
        list : List of indices.
    """
    focus = text[:-4]
    marker = len(focus)
    indices = []

    while marker >= width - 1:

        if (all(char in char_to_numbers for char in focus[marker - width:marker])):
            if indices == []:
                indices.append(marker - width)
            break
        
        indices.append(marker - width)
        marker -= 1
    return indices




def clean_number_plate(text):
    """
    Returns a cleaned version of the number-plate.
    
    Normalizes the raw OCR output into a string with no white spaces or
    special characters. Characters that are likely to be unnecessary to the
    left of the state-code and to the right of the 4-digit number are removed.

    Parameters:
        text (str) : Raw OCR output
    Returns:
        str : The cleaned number-plate.
    """
    n = re.sub(r"[^0-9a-zA-Z]", "", text)
    left = 0

    while left < len(n):
        if n[left:left + 2] in state_codes:
            break
        left += 1
        
    right = len(n) - 4
    while right >= 0:
        if all(char in char_to_numbers for char in n[right:right+4]):
            break
        right -= 1

    return n[left:right + 4]


# Function frp excpetion handling return error name with line no
def PrintException():
    
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    return 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)


init()
