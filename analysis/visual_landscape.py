import sys
import os
import yaml
sys.path.append("../")
assert os.path.exists(
    "./visualization/loss-landscape"), "Please clone the repo https://github.com/tomgoldstein/loss-landscape to ./visualization/"
sys.path.append("./visualization/loss-landscape")
sys.path.append(os.getcwd())
import time
import scheduler
import torch.nn as nn
import evaluation

import mpi4pytorch as mpi
import h52vtp as h52vtp
import plot_surface as plot_surface
import plot_1D as plot_1D
import plot_2D as plot_2D
import net_plotter as net_plotter
import projection as proj
from utils.defense_utils.dbd.model.model import SelfModel, LinearModel
from utils.defense_utils.dbd.model.utils import (
    get_network_dbd,
    load_state,
    get_criterion,
    get_optimizer,
    get_scheduler,
)
from utils.save_load_attack import load_attack_result
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import (
    get_transform,
    get_dataset_denormalization,
)
from visual_utils import *
import torch
import numpy as np
import torchvision.transforms as transforms
import socket
import h5py
from matplotlib import pyplot as plt
from matplotlib import cm
import h5_util
from os.path import exists, commonprefix


# modified from https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py by changing the load model part.
def setup_direction(args, dir_file, net, net2 = None, net3 = None):
    """
        Setup the h5 file to store the directions.
        - xdirection, ydirection: The pertubation direction added to the mdoel.
          The direction is a list of tensors.
    """
    print('-------------------------------------------------------------------')
    print(f'setup_direction {dir_file}')
    print('-------------------------------------------------------------------')

    # Skip if the direction file already exists
    if exists(dir_file):
        f = h5py.File(dir_file, 'r')
        if (args.y and 'ydirection' in f.keys()) or 'xdirection' in f.keys():
            f.close()
            print ("%s is already setted up" % dir_file)
            return
        f.close()

    # Create the plotting directions
    f = h5py.File(dir_file,'w') # create file, fail if exists
    if not args.dir_file:
        print("Setting up the plotting directions...")
        if net2:
            print("Using target direction")
            xdirection = net_plotter.create_target_direction(net, net2, args.dir_type)
        else:
            print("Using random direction")
            xdirection = net_plotter.create_random_direction(net, args.dir_type, args.xignore, args.xnorm)
        h5_util.write_list(f, 'xdirection', xdirection)

        if args.y:
            if net3:
                print("Using target direction")
                ydirection = net_plotter.create_target_direction(net, net3, args.dir_type)
            else:
                print("Using random direction")
                ydirection = net_plotter.create_random_direction(net, args.dir_type, args.yignore, args.ynorm)
            h5_util.write_list(f, 'ydirection', ydirection)

    f.close()
    print ("direction file created: %s" % dir_file)

# modified from https://github.com/tomgoldstein/loss-landscape/blob/master/plot_surface.py by change the f.close() to avoid some bugs
def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """
    
    loaded = False
    while not loaded:
        try:
            # read only to avoid conflict with other processes
            f = h5py.File(surf_file, 'r')
            loaded = True
        except:
            print(f"rank-{rank}:Error opening file, retrying...", flush=True)
            time.sleep(5)
    
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    fkeys = list(f.keys())
    f.close()
    
    if loss_key not in fkeys:
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
    else:
        print(f"rank-{rank}:losses and accuracies already calculated", flush=True)
        return
    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f = h5py.File(surf_file, 'r+')
            try:
                f[loss_key][:] = losses
                f[acc_key][:] = accuracies
            except:
                f[loss_key] = losses
                f[acc_key] = accuracies

            f.flush()
            f.close()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time

    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))


# Basic setting: args
args = get_args()

with open(args.yaml_path, "r") as stream:
    config = yaml.safe_load(stream)
config.update({k: v for k, v in args.__dict__.items() if v is not None})
args.__dict__ = config
args = preprocess_args(args)
fix_random(int(args.random_seed))

save_path_attack = "./record/" + args.result_file_attack
visual_save_path = save_path_attack + "/visual"

# Load result
if args.prototype:
    result_attack = load_prototype_result(args, save_path_attack)
else:
    result_attack = load_attack_result(save_path_attack + "/attack_result.pt")

selected_classes = np.arange(args.num_classes)

# keep the same transforms for train and test dataset for better visualization
result_attack["clean_train"].wrap_img_transform = result_attack["clean_test"].wrap_img_transform 
result_attack["bd_train"].wrap_img_transform = result_attack["bd_test"].wrap_img_transform 

# Create dataset
if args.visual_dataset == 'clean_train':
    visual_dataset = result_attack["clean_train"]
elif args.visual_dataset == 'bd_train':
    visual_dataset = result_attack["bd_train"]
    visual_dataset.wrapped_dataset.getitem_all = False  # only return img and label
else:
    assert False, "Illegal vis_class"

print(
    f'Create visualization dataset with \n \t Dataset: {args.visual_dataset} \n \t Number of samples: {len(visual_dataset)}  \n \t Selected classes: {selected_classes}')

# Create data loader
data_loader = torch.utils.data.DataLoader(
    visual_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
)

# Create denormalization function
for trans_t in data_loader.dataset.wrap_img_transform.transforms:
    if isinstance(trans_t, transforms.Normalize):
        denormalizer = get_dataset_denormalization(trans_t)


# Load model
model_visual = generate_cls_model(args.model, args.num_classes)

if args.result_file_defense != "None":
    save_path_defense = "./record/" + args.result_file_defense
    visual_save_path = save_path_defense + "/visual"

    result_defense = load_attack_result(
        save_path_defense + "/defense_result.pt")
    defense_method = args.result_file_defense.split('/')[-1]
    if defense_method == 'fp':
        model_visual.layer4[1].conv2 = torch.nn.Conv2d(
            512, 512 - result_defense['index'], (3, 3), stride=1, padding=1, bias=False)
        model_visual.linear = torch.nn.Linear(
            (512 - result_defense['index'])*1, args.num_classes)
    if defense_method == 'dbd':
        backbone = get_network_dbd(args)
        model_visual = LinearModel(
            backbone, backbone.feature_dim, args.num_classes)
    model_visual.load_state_dict(result_defense["model"])
    print(f"Load model {args.model} from {args.result_file_defense}")
else:
    model_visual.load_state_dict(result_attack["model"])
    print(f"Load model {args.model} from {args.result_file_attack}")



# !!! Important to set eval mode !!!
model_visual.eval()

# make visual_save_path if not exist
os.mkdir(visual_save_path) if not os.path.exists(visual_save_path) else None


############################################
######## 2. Plot the loss landscape  #######
############################################
print('Plotting the loss landscape')

# additonal args
args.mpi = True
args.cuda = True if "cuda" in args.device else False
args.show = False
args.proj_file = ""
args.dir_file = ''

# --------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------
if args.mpi:
    comm = mpi.setup_MPI()
    rank, nproc = comm.Get_rank(), comm.Get_size()
    print(f"Get rank {rank}")
else:
    comm, rank, nproc = None, 0, 1

# in case of multiple GPUs per node, set the GPU to use for each rank
if args.cuda:
    if not torch.cuda.is_available():
        raise Exception(
            'User selected cuda option, but cuda is not available on this machine')
    gpu_count = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpu_count)
    print('Rank %d use GPU %d of %d GPUs on %s' %
          (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

# --------------------------------------------------------------------------
# Check plotting resolution
# --------------------------------------------------------------------------
try:
    args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
    args.ymin, args.ymax, args.ynum = (None, None, None)
    args.xnum = int(args.xnum)
    if args.y:
        args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
        assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
        args.ynum = int(args.ynum)

except:
    raise Exception(
        'Improper format for x- or y-coordinates. Try something like -1:1:51')

if args.dir_file:
    print('Use given dir_file in args:', args.dir_file)
else:
    dir_file = save_path_attack + '/' + args.result_file_attack + '_direction.h5'
    print(f'No dir_file is given, generate dir_file at {dir_file} now')

# --------------------------------------------------------------------------
# Load models and extract parameters
# --------------------------------------------------------------------------
w = net_plotter.get_weights(model_visual)  # initial parameters
# deepcopy since state_dict are references
s = copy.deepcopy(model_visual.state_dict())
if args.ngpu > 1:
    # data parallel with multiple GPUs on a single node
    net = torch.nn.DataParallel(
        model_visual, device_ids=range(torch.cuda.device_count()))


# --------------------------------------------------------------------------
# Setup the direction file and the surface file
# --------------------------------------------------------------------------

# Only used for saving direction and surface file
args.model_file = visual_save_path + f'/{args.result_file_attack}'
args.model_file1 = ""
args.model_file2 = ""
args.model_file3 = ""
model_1_perb = None
model_2_perb = None

criterion = nn.CrossEntropyLoss()
if args.loss_name == 'mse':
    criterion = nn.MSELoss()

if rank == 0 and args.dir_gen == 'hessian':
    args.model_file1 = visual_save_path + f'/{args.result_file_attack}_model_1.pt'
    args.model_file2 = visual_save_path + f'/{args.result_file_attack}_model_2.pt'

    if os.path.exists(args.model_file1) and os.path.exists(args.model_file2):
        print(f'Load model_1 and model_2 from {args.model_file1} and {args.model_file2}')
        model_1_perb = generate_cls_model(args.model, args.num_classes)
        model_2_perb = generate_cls_model(args.model, args.num_classes)
        model_1_perb.load_state_dict(torch.load(args.model_file1))
        model_2_perb.load_state_dict(torch.load(args.model_file2))
    else:
        # compute the top-2 eigenvector of hessian matrix as directions
        from pyhessian import hessian # Hessian computation
        
        # This is a simple function, that will allow us to perturb the model paramters and get the result
        # from https://github.com/amirgholami/PyHessian/blob/master/Hessian_Tutorial.ipynb
        def get_params(model_orig,  model_perb, direction, alpha):
            for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
                m_perb.data = m_orig.data + alpha * d
            return model_perb   
        
        model_1 = generate_cls_model(args.model, args.num_classes)
        model_2 = generate_cls_model(args.model, args.num_classes)
        
        model_visual = model_visual.to(args.device)
        model_1 = model_1.to(args.device)
        model_2 = model_2.to(args.device)

        # get a batch of data
        batch_x, batch_y = next(iter(data_loader))
        batch_x = batch_x.to(args.device)
        batch_y = batch_y.to(args.device)
        
        # create the hessian computation module
        hessian_comp = hessian(model_visual, criterion, data=(batch_x, batch_y), cuda=args.cuda)    
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

        
        model_1_perb = get_params(model_visual, model_1, top_eigenvector[0], 1)
        model_2_perb = get_params(model_visual, model_2, top_eigenvector[1], 1)

        model_1_perb.eval()
        model_2_perb.eval()
        
        
        torch.save(model_1_perb.cpu().state_dict(), args.model_file1)
        torch.save(model_2_perb.cpu().state_dict(), args.model_file2)

        print('Use eigenvectors of hessian matrix as directions.')
        
# resume all parameters to keep the same as other ranks
model_visual = model_visual.cpu()
args.model_file1 = ""
args.model_file2 = ""

args.surf_file = ""
args.plot = True
args.data_split = 0
args.proj_file = ""

dir_file = net_plotter.name_direction_file(args)  # name the direction file

if rank == 0:
    setup_direction(args, dir_file, net = model_visual, net2 = model_1_perb, net3 = model_2_perb)

surf_file = plot_surface.name_surface_file(args, dir_file)
if rank == 0:
    plot_surface.setup_surface_file(args, surf_file, dir_file)

# load directions
loaded = False
while not loaded:
    try:
        d = net_plotter.load_directions(dir_file)
        print(f'rank-{rank}: directions loaded')
        loaded = True
    except:
        print(f'rank-{rank}: Waiting for direction file {dir_file} to be loaded...', flush=True)
        print('Please restart the program if the direction file is not loaded after 30 seconds.')
        time.sleep(rank*2)
        
        
# calculate the consine similarity of the two directions
if len(d) == 2 and rank == 0:
    similarity = proj.cal_angle(proj.nplist_to_tensor(
        d[0]), proj.nplist_to_tensor(d[1]))
    print('cosine similarity between x-axis and y-axis: %f' % similarity)

# --------------------------------------------------------------------------
# Start the computation
# --------------------------------------------------------------------------
crunch(surf_file, model_visual, w, s, d,
                    data_loader, 'train_loss', 'train_acc', comm, rank, args)

# --------------------------------------------------------------------------
# Plot figures
# --------------------------------------------------------------------------
if args.plot and rank == 0:
    print("plotting landscape")
    # wait 3 seconds
    time.sleep(2.5)
    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    surf_name = "train_loss"

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err':
        Z = 100 - np.array(f[surf_name][:])
    else:
        print('%s is not found in %s' % (surf_name, surf_file))

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------
    fig = plt.figure()

    def Axes3D(fig):
        return fig.add_subplot(projection='3d')
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.tight_layout()
    plt.savefig(visual_save_path + f"/landscape_{args.visual_dataset}.png")
    
    print(f'Save to {visual_save_path + f"/landscape_{args.visual_dataset}"}.png')

    # save to vtk file. you can use paraview to visualize the results
    h52vtp.h5_to_vtp(surf_file, surf_name, log=False, zmax=10, interp=1000)

    # Another way to show the results is the function provided by plot_2D
    # if rank == 0:
    #     args.vmin = 0.1
    #     args.vmax = 10
    #     args.vlevel = 0.5
    #     if args.y and args.proj_file:
    #         plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
    #     elif args.y:
    #         plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
    #     else:
    #         plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)


