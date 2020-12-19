import facenet
import detect_face
import numpy as np
from scipy import misc
import tensorflow as tf

model_dir = "weight/facenet.pb"
image_size = 160
margin = 44
gpu_memory_fraction = 0.65


def main(image_BytesIO, is_aligned=False):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model_dir)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            if is_aligned is True:
                images = facenet.load_data(image_BytesIO, False, False, image_size)
                image_list = [images]
            else:
                img = misc.imread(image_BytesIO)
                image_list = load_and_align_data(img, image_size, margin, gpu_memory_fraction)
            embed_list = []
            for image in image_list:
                feed_dict = {images_placeholder: image, phase_train_placeholder: False}
                # Use the facenet model to calcualte embeddings
                embed = sess.run(embeddings, feed_dict=feed_dict)
                embed = embed.tolist()
                embed_list.append(embed[0])
    return embed_list


def load_and_align_data(img, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
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
    result = main("./2.jpg", is_aligned=False)
    print(result)
    print(len(result))

    # # 批量测试比对
    # import os
    # img_path = "./test"
    # file_path_list = os.listdir(img_path)
    # file_path_list = [os.path.join(img_path, x) for x in file_path_list]
    # encode_dict = main(file_path_list, is_aligned=False)
    # for i in range(len(encode_dict) - 1):
    #     temp_key, temp_value = encode_dict.popitem()
    #     for key, value in encode_dict.items():
    #         print("{} - {}:".format(temp_key.split("/")[-1], key.split("/")[-1]), np.linalg.norm(temp_value - value))
