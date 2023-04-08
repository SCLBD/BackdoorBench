import sys
import os
import yaml
import torch
from torchviz import make_dot, make_dot_from_trace

sys.path.append("../")
sys.path.append(os.getcwd())

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.fix_random import fix_random
from visual_utils import *

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

# Load model
model_visual = generate_cls_model(args.model, args.num_classes)


# make visual_save_path if not exist
os.mkdir(visual_save_path) if not os.path.exists(visual_save_path) else None

############## Model Structure ##################
print("Plotting Model Structure using pytorchviz")

# pip install -U git+https://github.com/szagoruyko/pytorchviz.git@master

x = torch.zeros([10, args.input_channel, args.input_height, args.input_width])

dot = make_dot(model_visual(x), params=dict(model_visual.named_parameters()))
dot.format = "png"
dot.render(f'structure_{args.model}', directory=visual_save_path, cleanup=True)

print(f'Save to {visual_save_path + f"/structure_{args.model}"}.png')


# Another way to show model structure using hiddenlayer
print("Plotting Model Structure using hiddenlayer")

import hiddenlayer as hl

def build_dot(graph, rankdir = 'TB'):
    """Generate a GraphViz Dot graph.
    Returns a GraphViz Digraph object.
    This is modified from https://github.com/waleedka/hiddenlayer/blob/master/hiddenlayer/graph.py
    by changing rankdir="TB" to allow a vertical plot.
    see https://github.com/waleedka/hiddenlayer/issues/63
    args:
        graph: hiddlen layer graph
        rankdir: direction for show plot. Left to right (LR) or Top to down (TD).
    """
    from graphviz import Digraph

    # Build GraphViz Digraph
    dot = Digraph()
    dot.attr("graph", 
                bgcolor=graph.theme["background_color"],
                color=graph.theme["outline_color"],
                fontsize=graph.theme["font_size"],
                fontcolor=graph.theme["font_color"],
                fontname=graph.theme["font_name"],
                margin=graph.theme["margin"],
                rankdir=rankdir,
                pad=graph.theme["padding"])
    dot.attr("node", shape="box", 
                style="filled", margin="0,0",
                fillcolor=graph.theme["fill_color"],
                color=graph.theme["outline_color"],
                fontsize=graph.theme["font_size"],
                fontcolor=graph.theme["font_color"],
                fontname=graph.theme["font_name"])
    dot.attr("edge", style="solid", 
                color=graph.theme["outline_color"],
                fontsize=graph.theme["font_size"],
                fontcolor=graph.theme["font_color"],
                fontname=graph.theme["font_name"])

    for k, n in graph.nodes.items():
        label = "<tr><td cellpadding='6'>{}</td></tr>".format(n.title)
        if n.caption:
            label += "<tr><td>{}</td></tr>".format(n.caption)
        if n.repeat > 1:
            label += "<tr><td align='right' cellpadding='2'>x{}</td></tr>".format(n.repeat)
        label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
        dot.node(str(k), label)
    for a, b, label in graph.edges:
        if isinstance(label, (list, tuple)):
            label = "x".join([str(l or "?") for l in label])

        dot.edge(str(a), str(b), label)
    return dot

transforms="default"

'''
For AdaptivePool, ONNX only support pool with output_size = 1 for all dimensions or output shape is a factor of input shape.
It's recommended to replace the adaptive pooling with regular pooling if possible.
Otherwise, you can uncomment the following code to use a self-defined pooling layer to run it anyway.
'''

# for name, module in model_visual.named_modules():
#     if isinstance(module, torch.nn.AdaptiveAvgPool2d) or isinstance(module, torch.nn.AdaptiveMaxPool2d):
#         # hook a function to get input shape
#         def shape_hook(module, input_, output_):
#             global out_shape
#             out_shape = output_.shape
#             return None
#         h = module.register_forward_hook(shape_hook)
#         model_visual(torch.zeros([1, args.input_channel, args.input_height, args.input_width]))
        
#         class pseduo_pool(torch.nn.AdaptiveAvgPool2d):
#             def __init__(self) -> None:
#                 super().__init__(output_size=(1,1))
                
#             def forward(self, input):
#                 pseduo_out = torch.zeros(out_shape) * torch.sum(input)
#                 return pseduo_out
        
#         setattr(model_visual, name, pseduo_pool())
#         print(f"replace {module} by s self-defined pool layer.")

#         model_visual(torch.zeros([1, args.input_channel, args.input_height, args.input_width]))
        
#         transforms = [
#             # Fold the self-defined operations into SelfDefined Pooling
#             # may cause name problem if you have the same operation pattern in your model
#             hl.transforms.Fold("ReduceSum > Mul", "AvgPool"),
#             hl.transforms.Fold("Constant > AvgPool", "AvgPool2", name = "Self-Defined Pooling")
#         ]

try:
    graph = hl.build_graph(model_visual, torch.zeros([10, args.input_channel, args.input_height, args.input_width]), transforms=transforms)
    dot = build_dot(graph)
    dot.format = "png"
    dot.render(f'structure_{args.model}_hl', directory=visual_save_path, cleanup=True)

    print(f'Save to {visual_save_path + f"/structure_{args.model}_hl"}.png')

except:
    print("Unsupported operation in hiddenlayer, recommend to use pytorchviz only.")
