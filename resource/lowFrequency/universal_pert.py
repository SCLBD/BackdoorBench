import logging
import numpy as np
from deepfool import tar_deepfool
from gauss_smooth import gauss_smooth, normalization#, smooth_clip
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from dataset import CIFAR10Dataset
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

def universal_perturbation(dataset,
                           test_dataset,
                           net,
                           target,
                           delta=0.8,
                           max_iter_uni = 50,
                           num_classes=10,
                           overshoot=0.02,
                           max_iter_df=200,
                           device = 'cpu',
                           save_path_prefix = None,
                           ):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)

    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).

    :param grads: gradient functions with respect to input (as many gradients as classes).

    :param delta: controls the desired fooling rate (default = 80% fooling rate)

    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)

    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

    :param max_iter_df: maximum number of iterations for deepfool (default = 10)

    :return: the universal perturbation.
    """
    net.eval()

    if torch.cuda.is_available():
        device = device
        net.to(device)
        cudnn.benchmark = True
        logging.info('use cuda')
    else:
        device = 'cpu'
        logging.info('use cpu')

    num_images =  np.shape(dataset)[0]

    v = np.zeros(dataset.shape[1:]).astype('float32')
    # best_frate = 0.0
    fooling_rate = 0.0
    # file_perturbation = os.path.join('data', 'best_universal.npy')
    itr = 0
    fooling_rate_list = []
    while fooling_rate <  1-delta and itr < max_iter_uni:

    # while itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)

        logging.info (f'Starting pass number {itr}')

        # Go through the data set and compute the perturbation increments sequentially
        for k in tqdm(range(0, num_images)):

            logging.info(f'    image : {k}')

            cur_img = dataset[k:(k+1), :, :, :]
            data = np.transpose(cur_img, (0,3,1,2))
            data = torch.from_numpy(data)
            data = data.to(device)
            r2 = int(net(data).max(1)[1])
            torch.cuda.empty_cache()

            
            add_v = cur_img + v
            data_p = np.transpose(add_v, (0,3,1,2))
            data_p = torch.from_numpy(data_p)
            data_p = data_p.to(device)
            r1 = int(net(data_p).max(1)[1])
            torch.cuda.empty_cache()

            if r1 == r2:

                # Compute adversarial perturbation
                dr, iter_i, _, _ = tar_deepfool(data_p[0], net, target=target, num_classes=num_classes,
                                              overshoot=overshoot, max_iter=max_iter_df, device=device)
                # Make sure it converged...
                if iter_i < max_iter_df-1:
                    assert not np.any(np.isnan(dr))
                    assert np.all(np.isfinite(dr))
                    
                    
                    # v = v + dr.astype('float32')
                    v = v + gauss_smooth(dr)
                    v = gauss_smooth(v)
                    v = normalization(v)

        logging.info(f"iter_i:{iter_i} end")

        logging.info(f"v min max {v.min()}, {v.max()}")
        logging.info(f"v*255 min max {(v.min()*255)}, {v.max()*255}")

        itr = itr + 1

        with torch.no_grad():
            est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
            est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))
            batch_size = 100
            test_data_orig = test_dataset
            test_loader_orig = DataLoader(dataset=test_data_orig, batch_size=batch_size, pin_memory=True)
            # test_data_pert = CIFAR10Dataset(dataset, pert=v)
            # test_loader_pert = DataLoader(dataset=test_data_pert, batch_size=batch_size, pin_memory=True)

            net.eval()

            print(f"v.shape:{v.shape}")

            v_tensor = torch.from_numpy(
                np.transpose(v.squeeze(), (2,0,1))
            )[None,...].to(device)

            for batch_idx, (inputs, _, *other) in enumerate(test_loader_orig):
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
            torch.cuda.empty_cache()

            for batch_idx, (inputs, _, *other) in enumerate(test_loader_orig):
                inputs = inputs.to(device)
                inputs += v_tensor
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_pert = torch.cat((est_labels_pert, predicted.cpu()))
            torch.cuda.empty_cache()

            fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert))/float(len(est_labels_orig))
            fooling_rate_list.append(fooling_rate)

            logging.info(f"FOOLING RATE: {fooling_rate}")
            dif_count = est_labels_pert[np.where(est_labels_pert != est_labels_orig)].cpu().numpy()
            logging.info(f"dif_count:{dif_count}")

        np.save(f'{save_path_prefix}_{iter_i}.npy', v)

        v_lossy_image = np.clip(deepcopy(v) * 255 + 255 / 2, 0, 255).squeeze()  # since v is [0,1]

        Image.fromarray(v_lossy_image.astype(np.uint8)).save(f'{save_path_prefix}_{iter_i}_lossy.jpg')

        last_ten_fool_rate = np.array(fooling_rate_list[-5:])
    
        logging.info(f"last_ten_fool_rate :{last_ten_fool_rate}")

        if len(last_ten_fool_rate) == 5 and last_ten_fool_rate.max() - last_ten_fool_rate.min() < 0.01:

            return v


    #     if len(dif_count)>(dataset.shape[0]*0.05):
        #         counts = np.bincount(dif_count.astype(np.int))
        #         target = np.argmax(counts)
        #     logging.info(dif_count)
        #     logging.info('the dominant label is:', target)
        #     if fooling_rate >= best_frate:
        #         best_v = v
        #         best_frate = fooling_rate
        #         new_target = target
        #         logging.info('the best fooling rate updating to:',best_frate)
        #         logging.info('the target label is updating to:', new_target)
        #         np.save(os.path.join(file_perturbation), best_v)

    return  v
    #return best_v,new_target