'''
This script aims to do transformation between nCHW and nHWC
please note that these two function both need to have 3 or 4 dimension,
    PIL or list DOES NOT SUPPORT !
'''
def nCHW_to_nHWC(images):
    if images.shape.__len__() == 3:
        return images.transpose((1,2,0))
    elif images.shape.__len__() == 4:
        return images.transpose((0, 2, 3, 1))

def nHWC_to_nCHW(images):
    if images.shape.__len__() == 3:
        return images.transpose((2,0,1))
    elif images.shape.__len__() == 4:
        return images.transpose((0, 3, 1, 2))