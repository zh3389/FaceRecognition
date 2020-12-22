import math
import json
import requests


def get_face_encoding(img_path):
    img_file = {'file': open(img_path, 'rb')}
    upload_data = {"accept": "application/json",
                   "Content-Type": "multipart/form-data"}
    upload_res = requests.post("http://192.168.31.220:8000/uploadfile", upload_data, files=img_file)
    encoding = json.loads(upload_res.content)
    return encoding


def euclidean_distance(A, B):
    '''用于计算两个列表中, 人脸特征之间的欧式距离.'''
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


obama_1_encoding = get_face_encoding("assets/obama.jpg")
obama_2_encoding = get_face_encoding("assets/obama2.jpg")
biden_encoding = get_face_encoding("assets/biden.jpg")
print("obama_1 and obama_2:", euclidean_distance(obama_1_encoding[0], obama_2_encoding[0]))
print("obama_1 and biden  :", euclidean_distance(obama_1_encoding[0], biden_encoding[0]))
print("obama_2 and biden  :", euclidean_distance(obama_2_encoding[0], biden_encoding[0]))
