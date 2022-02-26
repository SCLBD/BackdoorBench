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