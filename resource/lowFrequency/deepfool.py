import numpy as np
from torch.autograd import Variable
import torch as torch
import copy

#@resource_check
def tar_deepfool(image, net, target, num_classes=10, overshoot=0.02, max_iter=100, device = 'cpu'):

    """
       :param image: Image of size 1x3xHxW
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        image = image.to(device)
        net = net.to(device)

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image[None, :, :, :].cpu().numpy().shape
    pert_image = copy.deepcopy(image)

    x = Variable(pert_image[None, :, :, :], requires_grad=True)
    fs = net.forward(x)
    f_i = fs.data.cpu().numpy().flatten()
    k_i = np.argmax(f_i)

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while k_i != target and loop_i < max_iter:
        pert = np.inf
        gradients = []
        for k in range(0, num_classes):
            if x.grad is not None:
                x.grad.zero_()
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
            gradients.append(cur_grad)
        gradients = np.expand_dims(np.vstack(gradients), axis=1)
        k = np.where(I == target)[0][0]
        # set new w_k and new f_k
        w_k = gradients[k, :, :, :, :] - gradients[0, :, :, :, :]
        f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

        pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

        # determine which w_k to use
        if pert_k < pert:
            pert = pert_k
            w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        if not np.all(np.isfinite(r_i)):
            r_i = np.zeros_like(r_i)
        # r_tot = r_tot + r_i
        r_tot = np.float32(r_tot + r_i)
        # Added 1e-4 for numerical stability
        # r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        loop_i += 1

    r_tot = (1+overshoot) * r_tot
    r_tot = r_tot.transpose(0, 2, 3, 1)

    return r_tot, loop_i, k_i, pert_image
