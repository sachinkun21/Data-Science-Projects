%%time
import glob
list_of_Incorrect_LE160486 = glob.glob('F:\LE160486\IncorrectPlate\*')
print(len(list_of_Incorrect_LE160486))

dict_Incor_LE160486 = {'Filename':[], 'Plate' : [], 'B-Box' : []}
i = 1
for image in list_of_Incorrect_LE160486:
    print(i,  end = ' ')
    i+=1
    with open(image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    str1_plate = encoded_string.decode('utf-8')


    dict1 = json.dumps({'FrontCamera':str1_plate , 'BackCamera': str1_NP})

    result = json.loads(run(dict1))
    
    dict_Incor_LE160486['Filename'].append(image)
    dict_Incor_LE160486['Plate'].append(result['plate'])
    dict_Incor_LE160486['B-Box'].append(result['info'])
   

df_Incorrect_LE160486 = pd.DataFrame(dict_Incor_LE160486)
df_Incorrect_LE160486.head()
