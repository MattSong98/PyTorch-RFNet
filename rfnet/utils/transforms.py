import random
import collections
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import rotate


class Uniform(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self):
        return random.uniform(self.a, self.b)

class Gaussian(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def sample(self):
        return random.gauss(self.mean, self.std)

class Constant(object):
    def __init__(self, val):
        self.val = val

    def sample(self):
        return self.val


##############################
######### Transforms #########
##############################


class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False): # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            # how to know  if the last dim is channel??
            # nhwtc vs nhwt??
            shape = im.shape[1:dim+1]
            # print(dim,shape) # 3, (240,240,155)
            self.sample(*shape)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)] # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

Identity = Base


class RandomRotion(Base):
    def __init__(self,angle_spectrum=10):
        assert isinstance(angle_spectrum,int)
        # axes = [(2, 1), (3, 1),(3, 2)]
        axes = [(1, 0), (2, 1),(2, 0)]
        self.angle_spectrum = angle_spectrum
        self.axes = axes

    def sample(self,*shape):
        self.axes_buffer = self.axes[np.random.choice(list(range(len(self.axes))))] # choose the random direction
        self.angle_buffer = np.random.randint(-self.angle_spectrum, self.angle_spectrum) # choose the random direction
        return list(shape)

    def tf(self, img, k=0):
        """ Introduction: The rotation function supports the shape [H,W,D,C] or shape [H,W,D]
        :param img: if x, shape is [1,H,W,D,c]; if label, shape is [1,H,W,D]
        :param k: if x, k=0; if label, k=1
        """
        bsize = img.shape[0]

        for bs in range(bsize):
            if k == 0:
                # [[H,W,D], ...]
                # print(img.shape) # (1, 128, 128, 128, 4)
                channels = [rotate(img[bs,:,:,:,c], self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0, mode='constant', cval=-1) for c in
                            range(img.shape[4])]
                img[bs,...] = np.stack(channels, axis=-1)

            if k == 1:
                img[bs,...] = rotate(img[bs,...], self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0, mode='constant', cval=-1)

        return img

    def __str__(self):
        return 'RandomRotion(axes={},Angle:{}'.format(self.axes_buffer,self.angle_buffer)



class RandomFlip(Base):
    # mirror flip across all x,y,z
    def __init__(self,axis=0):
        # assert axis == (1,2,3) # For both data and label, it has to specify the axis.
        self.axis = (1,2,3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True,False])
        self.y_buffer = np.random.choice([True,False])
        self.z_buffer = np.random.choice([True,False])
        return list(shape) # the shape is not changed

    def tf(self,img,k=0): # img shape is (1, 240, 240, 155, 4)
        if self.x_buffer:
            img = np.flip(img,axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img,axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img,axis=self.axis[2])
        return img




class CenterCrop(Base):
    def __init__(self, size):
        self.size = size
        self.buffer = None

    def sample(self, *shape):
        size = self.size
        start = [(s -size)//2 for s in shape]
        self.buffer = [slice(None)] + [slice(s, s+size) for s in start]
        return [size] * len(shape)

    def tf(self, img, k=0):
        # print(img.shape)#(1, 240, 240, 155, 4)
        return img[tuple(self.buffer)]
        # return img[self.buffer]

    def __str__(self):
        return 'CenterCrop({})'.format(self.size)



class RandCrop3D(CenterCrop):
    def sample(self, *shape): # shape : [240,240,155]
        assert len(self.size)==3 # random crop [H,W,T] from img [240,240,155]
        if not isinstance(self.size,list):
            size = list(self.size)
        else:
            size = self.size
        start = [random.randint(0, s-i) for i,s in zip(size,shape)]
        self.buffer = [slice(None)] + [slice(s, s+k) for s,k in zip(start,size)]
        return size

    def __str__(self):
        return 'RandCrop({})'.format(self.size)



# for data only
class RandomIntensityChange(Base):
    def __init__(self,factor):
        shift,scale = factor
        assert (shift >0) and (scale >0)
        self.shift = shift
        self.scale = scale

    def tf(self,img,k=0):
        if k==1:
            return img

        shift_factor = np.random.uniform(-self.shift,self.shift,size=[1,img.shape[1],1,1,img.shape[4]]) # [-0.1,+0.1]
        scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale,size=[1,img.shape[1],1,1,img.shape[4]]) # [0.9,1.1)
        # shift_factor = np.random.uniform(-self.shift,self.shift,size=[1,1,1,img.shape[3],img.shape[4]]) # [-0.1,+0.1]
        # scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale,size=[1,1,1,img.shape[3],img.shape[4]]) # [0.9,1.1)
        return img * scale_factor + shift_factor

    def __str__(self):
        return 'random intensity shift per channels on the input image, including'





class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)



class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            shape = op.sample(*shape)

    def tf(self, img, k=0):
        #is_tensor = isinstance(img, torch.Tensor)
        #if is_tensor:
        #    img = img.numpy()

        for op in self.ops:
            # print(op,img.shape,k)
            img = op.tf(img, k) # do not use op(img) here

        #if is_tensor:
        #    img = np.ascontiguousarray(img)
        #    img = torch.from_numpy(img)

        return img

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)


