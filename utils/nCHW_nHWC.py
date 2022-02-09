def nCHW_to_nHWC(images):
    return images.transpose((0, 2, 3, 1))

def nHWC_to_nCHW(images):
    return images.transpose((0, 3, 1, 2))