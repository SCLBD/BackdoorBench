import argparse


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints/")
    parser.add_argument("--data_root", type=str, default="/home/ubuntu/workspaces/chenweixin/bdzoo/datasetgtsrb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")

    # ---------------------------- For Neural Cleanse --------------------------
    # Model hyperparameters
    parser.add_argument('--classifier', type=str, default='preactresnet18')
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--init_cost", type=float, default=1e-3)
    parser.add_argument("--atk_succ_threshold", type=float, default=98.0)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--early_stop_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cost_multiplier", type=float, default=2)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--target_label", type=int)
    parser.add_argument("--total_label", type=int)
    parser.add_argument("--EPSILON", type=float, default=1e-7)

    parser.add_argument("--to_file", type=bool, default=True)
    parser.add_argument("--n_times_test", type=int, default=10)

    parser.add_argument("--use_norm", type=int, default=1)

    parser.add_argument("--k", type=int)

    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of training data')
    parser.add_argument('--checkpoint_load', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--trigger_type', type=str, default=None)

    return parser
