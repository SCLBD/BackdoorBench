# The original image who activates the given layer most
python analysis/visual_act.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2 --visual_dataset bd_test --target_class 0

# The activation image distribution, i.e., the distribution of top-k images who activate the neurons most
python analysis/visual_actdist.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --visual_dataset mixed --target_class 0

# The confusion matrix
python analysis/visual_cm.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --visual_dataset bd_train --target_class 0

# The feature map of a (random) given image after a given layer
python analysis/visual_fm.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --visual_dataset bd_test

# The Frequency saliency map
python analysis/visual_fre.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2 --visual_dataset mixed --target_class 0

# The synthetic image who activates the given layer (found by gradient descend)
python analysis/visual_fv.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2

# The Grad-CAM of 4 random selected images
python analysis/visual_gradcam.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2 --visual_dataset bd_test --target_class 0

# The Landscape of a neuron network with MPI for parallel computing, e.g., 8 processes
mpirun -n 8 python analysis/visual_landscape.py --x=-1:1:51 --y=-1:1:51 --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --visual_dataset bd_train

# The Lipschitz constant of a neuron network
python analysis/visual_lips.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --normalize_by_layer

# The Neuron Activation of a given layer
python analysis/visual_na.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2 --visual_dataset bd_test --target_class 0

# The Shapely value of 4 random selected images
python analysis/visual_shap.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2 --visual_dataset bd_test --target_class 0

# The Total Activation Change of a neural network
python analysis/visual_tac.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2 --visual_dataset bd_test --target_class 0 --normalize_by_layer

# The T-SNE of features of a given layer
python analysis/visual_tsne.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2 --visual_dataset mixed --target_class 0

# The UMAP of features of a given layer
python analysis/visual_umap.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --target_layer_name layer4.1.conv2 --visual_dataset bd_train --target_class 0 --n_sub 50000

# The network structure. result_file_attack is only used for saving the result
python analysis/visual_network.py --result_file_attack badnet_demo --model preactresnet18

# The Eigenvalue Dense Plot of the Hessian Matrix
python analysis/visual_hessian.py --result_file_attack badnet_demo --result_file_defense badnet_demo/defense/ac --visual_dataset bd_train --batch_size 128
