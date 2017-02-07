import os
import numpy as np
#from scipy.misc import imsave
from PIL import Image
def save_visualizations(imgs, directory):
  for k in range(imgs.shape[0]):
      imgs_folder = os.path.join(directory, 'imgs')
      if not os.path.exists(imgs_folder):
          os.makedirs(imgs_folder)

      #imsave(os.path.join(imgs_folder, '%d.png') % k,
        #     imgs[k].reshape(*shape))
      #print(imgs.shape)
      img = Image.fromarray(imgs[k], mode='RGB')
      img.save(os.path.join(imgs_folder, '{}.png'.format(k)))

def whiten(x):
    mu = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)
    return (x-mu)/std
