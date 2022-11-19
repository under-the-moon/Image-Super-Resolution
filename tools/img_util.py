import numpy as np
import cv2
import os


def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt=255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        num, c, h, w = img.shape
        w_scale = 8
        h_scale = num // 8
        img_copy = img.clone().data.cpu().numpy()
        img_copy = img_copy.reshape((h_scale, w_scale, c, h, w))
        # (h_scale, w_scale, h, w, c)
        img_copy = img_copy.transpose((0, 1, 3, 4, 2))
        img_copy = img_copy.transpose((0, 2, 1, 3, 4))
        img_copy = img_copy.reshape(h_scale * h, w_scale * w, c)
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)
