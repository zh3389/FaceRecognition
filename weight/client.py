import json
import requests
import detect_face
import numpy as np
from scipy import misc
import tensorflow as tf


class Predictor():
    def __init__(self):
        self.url = 'http://127.0.0.1:8501/v1/models/docker_test:predict'
        self.margin = 44
        self.image_size = 160
        self.gpu_memory_fraction = 0.75

    def encoder(self, img_path):
        """Loads an image into PIL format."""
        images = load_and_align_data(img_path, self.image_size, self.margin, self.gpu_memory_fraction)
        images = np.asarray(images[0], dtype='float32')
        payload = {"inputs": {'input': images.tolist(), 'phase_train': False}}
        return payload

    def predict(self, payload):
        r = requests.post(self.url, json=payload)
        embedding = json.loads(r.content.decode('utf-8'))['outputs'][0]
        embedding = np.array(embedding)
        return embedding


def load_and_align_data(image_path, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    img = misc.imread(image_path)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    image_list = []
    for box in bounding_boxes:
        det = np.squeeze(box[0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        mean = np.mean(aligned)
        std = np.std(aligned)
        std_adj = np.maximum(std, 1.0 / np.sqrt(aligned.size))
        prewhitened = np.multiply(np.subtract(aligned, mean), 1 / std_adj)
        images = np.stack([prewhitened])
        image_list.append(images)
    return image_list


if __name__ == '__main__':
    img_path = "../data/input_data/Aaron_Patterson/Aaron_Patterson_0001.jpg"
    pre = Predictor()
    result = pre.encoder(img_path)
    print("=" * 100)
    result = pre.predict(result)
    print(result)

    # 批量测试人脸 并比对
    # import os
    # img_path = "./test"
    # file_path_list = os.listdir(img_path)
    # file_path_list = [os.path.join(img_path, x) for x in file_path_list]
    # encode_dict = {}
    # for file_path in file_path_list:
    #     embedding = pre.predict(pre.encoder(file_path))
    #     encode_dict[file_path] = embedding
    # for i in range(len(encode_dict) - 1):
    #     temp_key, temp_value = encode_dict.popitem()
    #     for key, value in encode_dict.items():
    #         print("{} - {}:".format(temp_key.split("/")[-1], key.split("/")[-1]), np.linalg.norm(temp_value - value))
