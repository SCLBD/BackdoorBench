import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--attack_mode", type=str, default="all2one", help="all2one or all2all")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--outfile")

    return parser
