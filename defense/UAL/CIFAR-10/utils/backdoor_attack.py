import numpy as np
from skimage.io import imread
# import pdb

def add_patch(img,trigger):
    flag=False
    if img.max()>1.:
        img=img/255.
        flag=True
    if trigger.max()>1.:
        trigger=trigger/255.

    # x,y=np.random.randint(10,20,size=(2,))
    x,y = np.random.choice([3, 28]), np.random.choice([3, 28])

    m,n,_=trigger.shape
    #img[x-int(m/2):x+m-int(m/2),y-int(n/2):y+n-int(n/2),:]=img[x-int(m/2):x+m-int(m/2),
    #                                                           y-int(n/2):y+n-int(n/2),:]*(1-trigger)+trigger

    img[x-int(m/2):x+m-int(m/2),y-int(n/2):y+n-int(n/2),:]=trigger              # opaque trigger
    if flag:
        img=(img*255).astype('uint8')
    return img

def generate_poisoned_data(X_train,Y_train,source,target, trigger):
    ind=np.argwhere(Y_train==source)
    Y_poisoned=target*np.ones((ind.shape[0])).astype(int)

    # k=np.random.randint(6,11)
    # trigger=imread('Data/Masks_Test_5/mask%1d.bmp'%(k))

    # pdb.set_trace()

    X_poisoned=np.stack([add_patch(X_train[i,...],trigger) for i in ind.squeeze()],0)

    return X_poisoned,Y_poisoned,trigger,ind.squeeze()