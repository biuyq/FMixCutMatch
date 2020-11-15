import os
import cv2

pathImageNet = "/home/weilegexiang/Desktop/PseudoLabeling-master/miniImagenet/data/miniImagenet/images/"
i=0;
for filename in os.listdir(r"./images"):
    i = i+1
    print(filename)
    im = cv2.imread(os.path.join(pathImageNet,filename))
    if filename != ".DS_Store":
        im_resized = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join("/home/weilegexiang/Desktop/PseudoLabeling-master/miniImagenet/data/miniImagenet/images84/", filename),im_resized)

    #copyfile(os.path.join(pathImageNet,lst_files[selected_images[i]]),os.path.join(pathImages, images[c][i]))
