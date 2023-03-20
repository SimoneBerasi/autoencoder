import cv2
import numpy as np

from AutoencoderModels import Model_noise_skip, Model_noise_skip_new

from Evaluation import *
from PIL import Image


test_dir = "Dataset/MVTec_Data/wood/test"
#test_dir = "Dataset/SEM/Anomalous/images"

if __name__ == "__main__":

    vailed_ext = [".jpg", ".png", ".tif"]
    import os

    f_list = []


    def Test2(rootDir):
        for lists in os.listdir(rootDir):
            path = os.path.join(rootDir, lists)
            filename, file_extension = os.path.splitext(path)
            if file_extension in vailed_ext:
                print(path)
                f_list.append(path)
            if os.path.isdir(path):
                Test2(path)


    Test2(test_dir)

    autoencoder = Model_noise_skip_new(input_shape=(None, None, 1))
    #autoencoder.load_weights("Weights/new_weights/check_epoch40.h5")
    autoencoder.load_weights("/home/simo/Desktop/Thesis Projects/DefaultBiondaAutoencoder/Weights/new_weights/wood_new_ae.h5")

    i=0
    for item in f_list:
        print(i)
        i=i+1

        img = cv2.imread(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_array = np.array(img)/255

        patches, y_valid = load_patches_from_image(img, 256, random=False, stride=16)
        patches = np.array(patches)

       # print(img_array)

        _, pred = batch_evaluation(patches, autoencoder, 100)
        pred = np.array(pred)
        #print(pred.shape)
        pred = image_reconstruction(pred)

        pred = pred * (-1)


        print(pred)
        print(img_array)
        visualize_results(img_array, pred, "a")
        anomaly_map = get_residual(pred.copy(), img_array.copy())
        print(anomaly_map)
        anomaly_map = np.array(anomaly_map)
        anomaly_map = np.reshape(anomaly_map, (1024, 1024))
        #anomaly_map = anomaly_map - np.min(anomaly_map)
        img = Image.fromarray(anomaly_map)
        path = os.path.splitext(item)[0]
        img.save("AnomalyMaps" + "/" + path + ".tiff")

        #print(np.mean(pred))
        #print(np.mean(img_array))

        visualize_results(anomaly_map, img_array, "a")




