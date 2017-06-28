import numpy as np
import cv2
from demo_utils import *
import matplotlib.pyplot as plt
import glob

def plot_2d(img, true, pred, num):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img.reshape((80,120,3)))
    ax.autoscale(False)
    plt.plot(pred[:,0], pred[:,1], '.w', markersize=7)
    plt.plot(true[:,0], true[:,1], 'sr', markersize=7)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('./real_images/preds/'+str(num)+'.png',bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":

    # Load CNN
    custom_objects={'p_norm_loss': p_norm_loss, 'landmark_accuracy' : landmark_accuracy, 'landmark_loss': landmark_loss}
    model_path = 'models/HeadFalse-spatial-2D-mean_squared_error-2.841274347020494-0.92591288972434949.h5'
    #model_path = 'models/AugmentedHeadFalse-double_tower-2D-mean_squared_error-5.1912007235517406-0.81428417525482011.h5'
    model = load_model(model_path, custom_objects)

    # plot_model(model, to_file='model.png', show_shapes=True,  show_layer_names=False, rankdir='TB')

    im_list = list(glob.iglob('./real_images/*.jpg', recursive=True))

    for num, image_path in enumerate(im_list):
        image = cv2.imread(image_path, 0)
        print(image.shape)
        #image = plt.imread(image_path, flatten = True)
        eye_img = np.reshape(image, (1, 80, 120)) / 255.0
        pred_ldmarks = model_predict(model, eye_img[np.newaxis, :, :, :], 1, 2)[0,:,:]
        plot_2d(plt.imread(image_path), pred_ldmarks, pred_ldmarks, num)
