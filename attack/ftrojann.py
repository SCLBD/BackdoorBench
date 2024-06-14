"""
An Invisible Black-box Backdoor Attack through Frequency Domain
this script is for ctrl attack from
https://github.com/SoftWiser-group/FTrojan.

@inproceedings{wang2022invisible,
  title={An Invisible Black-Box Backdoor Attack Through Frequency Domain},
  author={Wang, Tong and Yao, Yuan and Xu, Feng and An, Shengwei and Tong, Hanghang and Wang, Ting},
  booktitle={17th European Conference on Computer Vision, ECCV 2022},
  pages={396--413},
  year={2022},
  organization={Springer}
}

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. save the attack result for defense

LICENSE is at the end of this file

"""
import argparse
import os
import sys

sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args
import yaml


class Ftrojann(BadNet):
    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument("--channel_list", type=int, nargs="+", help="channel list")
        parser.add_argument("--magnitude", type=float, help="magnitude")
        parser.add_argument("--YUV", type=bool, help="YUV")
        parser.add_argument("--window_size", type=int, help="window size")
        parser.add_argument("--pos_list", type=int, nargs="+", help="pos list")

        parser.add_argument(
            "--bd_yaml_path",
            type=str,
            default="./config/attack/ftrojann/default.yaml",
            help="path for yaml file provide additional default attributes",
        )
        return parser
    
    def add_bd_yaml_to_args(self, args):
        with open(args.bd_yaml_path, "r") as f:
            mix_defaults = yaml.safe_load(f)
        mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = mix_defaults
        
        if len(args.pos_list) % 2 != 0:
            raise ValueError("pos_list should be even number")
        pos_list = []
        for i in range(len(args.pos_list) // 2):
            pos_list.append((args.pos_list[i * 2], args.pos_list[i * 2 + 1]))
        args.pos_list = pos_list

if __name__ == "__main__":
    attack = Ftrojann()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()

'''
MIT License

Copyright (c) 2022 SoftWiser-Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''