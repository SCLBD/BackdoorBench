### `generate_white_square.py`

`generate_white_square.py` is a simple example of how to generate a white square image. 

The white square image is used to generate the white square attack in the paper.

If you want to draw more complicated shapes, you can modify the code in `generate_white_square.py` or first generate a black image then stamp the trigger onto it and convert the image to npy file.

The last step is to specify the parameter `--patch_mask_path` for `badnet.py`.

### `generate_grid.py`

Similarly, `generate_grid.py` is to generate a grid trigger.

But note that , in the trigger area, the smallest number is 1 (we assume pixel values range from 0 to 255), since for the trigger file, we only treat area with pixle value > 0 as the part of area that we need to use. (That's also why we only need one file to both locate the mask and also record the pixel value in patch)

### Remainder

Since the trigger png file has a fixed size (eg. 32 * 32), in badnet.py if you attack a dataset with other resolution (eg. tiny with 64 * 64), then we will resize the trigger to the resolution of the dataset.

So, please note that for grid trigger, under different resolution, the grid can be finer or coarser! (eg. 32 * 32 -> 64 * 64, the grid will be coarser)