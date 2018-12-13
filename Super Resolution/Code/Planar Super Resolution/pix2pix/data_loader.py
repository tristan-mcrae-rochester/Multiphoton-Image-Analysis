import scipy
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False, interpolation = True):
        data_type = "train" if not is_testing else "val"
        imgs_A = []
        imgs_B = []
        if interpolation:
            path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

            batch_images = np.random.choice(path, size=batch_size)

            
            for img_path in batch_images:
                img = self.imread(img_path)

                h, w, _ = img.shape
                _w = int(w/2)
                img_A, img_B = img[:, :_w, :], img[:, _w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                # If training => do random flip or two
                if not is_testing and np.random.random() < 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                if not is_testing and np.random.random() < 0.5:
                    img_A = np.flipud(img_A)
                    img_B = np.flipud(img_B)
                if not is_testing:
                    rotations = random.randint(0,4)
                    img_A = np.rot90(img_A, rotations)
                    img_B = np.rot90(img_B, rotations)


                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/255.0#127.5 - 1.
            imgs_B = np.array(imgs_B)/255.0#127.5 - 1.


        else:
            paths_input = glob('./datasets/%s/%s/%s/*' % (self.dataset_name, data_type, 'inputs'))
            paths_label = glob('./datasets/%s/%s/%s/*' % (self.dataset_name, data_type, 'labels'))

            path_input = paths_input[0:1]
            path_label = paths_label[0:1]

            img_B = self.imread(path_input[0])
            img_A = self.imread(path_label[0])
            #import ipdb; ipdb.set_trace()


            #img_A = scipy.misc.imresize(img_A, self.img_res)
            #img_B = scipy.misc.imresize(img_B, self.img_res)
            #import ipdb; ipdb.set_trace()

            img_A = np.array(img_A)/255.0#127.5 - 1.
            img_B = np.array(img_B)/255.0#127.5 - 1.
            imgs_A.append(img_A)
            imgs_B.append(img_B)


        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False, interpolation = True):
        data_type = "train" if not is_testing else "val"
        
        if interpolation:
            path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

            self.n_batches = int(len(path) / batch_size)
            #import ipdb; ipdb.set_trace()
        
            for i in range(self.n_batches-1):
                batch = path[i*batch_size:(i+1)*batch_size]
                imgs_A, imgs_B = [], []
                for img in batch:
                    img = self.imread(img)
                    h, w, _ = img.shape
                    half_w = int(w/2)
                    img_A = img[:, :half_w, :]
                    img_B = img[:, half_w:, :]

                    img_A = scipy.misc.imresize(img_A, self.img_res)
                    img_B = scipy.misc.imresize(img_B, self.img_res)

                    if not is_testing and np.random.random() < 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)
                    if not is_testing and np.random.random() < 0.5:
                        img_A = np.flipud(img_A)
                        img_B = np.flipud(img_B)
                    if not is_testing:
                        rotations = random.randint(0,4)
                        img_A = np.rot90(img_A, rotations)
                        img_B = np.rot90(img_B, rotations)

                    imgs_A.append(img_A)
                    imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/255.0#127.5 - 1.
            imgs_B = np.array(imgs_B)/255.0#127.5 - 1.

            yield imgs_A, imgs_B

        else:
            path_input = glob('./datasets/%s/%s/%s/*' % (self.dataset_name, data_type, 'inputs'))
            path_label = glob('./datasets/%s/%s/%s/*' % (self.dataset_name, data_type, 'labels'))

            self.n_batches = int(len(path_input) / batch_size)
            #import ipdb; ipdb.set_trace()
            imgs_A_list, imgs_B_list = [], []
            for i in range(self.n_batches-1):
                batch_input = path_input[i*batch_size:(i+1)*batch_size]
                batch_label = path_label[i*batch_size:(i+1)*batch_size]
                imgs_A, imgs_B = [], []
                for img in batch_input:
                    img_B = self.imread(img)
                    imgs_B.append(img_B)

                for img in batch_label:
                    img_A = self.imread(img)
                    imgs_A.append(img_A)

                if not is_testing:
                    for j in range(batch_size):
                        #print(np.asarray(imgs_A).shape)
                        if np.random.random() < 0.5:
                            imgs_A[j] = np.fliplr(imgs_A[j])
                            imgs_B[j] = np.fliplr(imgs_B[j])
                        if  np.random.random() < 0.5:
                            imgs_A[j] = np.flipud(imgs_A[j])
                            imgs_B[j] = np.flipud(imgs_B[j])
                        rotations = random.randint(0,4)
                        imgs_A[j] = np.rot90(imgs_A[j], rotations)
                        imgs_B[j] = np.rot90(imgs_B[j], rotations)
                imgs_A_list.append(imgs_A)
                imgs_B_list.append(imgs_B)
                #import ipdb; ipdb.set_trace()
                #imgs_A.append(img_A)
                #imgs_B.append(img_B)

            imgs_A_list = np.array(imgs_A_list)/255.0#127.5 - 1.
            imgs_B_list = np.array(imgs_B_list)/255.0#127.5 - 1.

            yield imgs_A_list, imgs_B_list



    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)