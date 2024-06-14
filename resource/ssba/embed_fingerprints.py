import argparse
import os
import glob
import PIL
import bchlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument("--encoder_path", type=str, help="Path to trained StegaStamp encoder.")
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument("--output_dir", type=str, help="Path to save watermarked images to.")
parser.add_argument("--image_resolution", type=int, help="Height and width of square images.")
parser.add_argument("--identical_fingerprints", action="store_true", help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints.")
parser.add_argument("--check", action="store_true", help="Validate fingerprint detection accuracy.")
parser.add_argument("--decoder_path",type=str,help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--use_residual", type=int, default=0, help="Use residual mode or not",)

parser.add_argument("--encode_method", type=str, default='bch', help="['bch', 'seed', 'diff', 'manual', 'entropy']")
parser.add_argument('--secret', type=str, default='stega!!')
parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")
parser.add_argument("--diff_bits", type=int, default=0, help="number of different weights from ground truth")
parser.add_argument("--manual_str", type=str, default=None, help="The manual string given by user")
parser.add_argument("--proportion", type=float, default=1.0, help="The propotion of 1 in the encode sequence")

parser.add_argument("--use_modulated", type=int, default=0, help="Use modulated convolution or not", )
parser.add_argument("--fc_layers", type=int, default=0, help="Use 8 fc layers before modulated convolution?", )
parser.add_argument("--fused_conv", type=int, default=0, help="Use fused conv for modulated conv?",)
parser.add_argument("--bias_init", type=int, default=None, help="Specified bias initialization for modulated conv",)

parser.add_argument("--test_save_file", type=str, default=None, help="where to save test file")

parser.add_argument("--poison_rate", type=float, default=1.0, help="the poison rate set in original version")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size 

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, poison_rate=1.0, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.JPEG")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform
        self.poison_rate = poison_rate
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return int(self.poison_rate * len(self.filenames))

def load_data():
    global dataset, dataloader

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, poison_rate=args.poison_rate, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

import models
import models_modulated
from generate_fingerprints import generate_fingerprints

def load_models():
    global HideNet, RevealNet
    global FINGERPRINT_SIZE
    
    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    state_dict = torch.load(args.encoder_path)
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]
    
    if not args.use_modulated:
        print("----------Not using modulated conv!----------")
        HideNet = models.StegaStampEncoder(
            IMAGE_RESOLUTION,
            IMAGE_CHANNELS,
            fingerprint_size=FINGERPRINT_SIZE,
            return_residual=args.use_residual
        )
        RevealNet = models.StegaStampDecoder(
            IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
        )
    else:
        print("----------Using modulated conv!----------")
        HideNet = models_modulated.StegaStampEncoder(
            IMAGE_RESOLUTION,
            IMAGE_CHANNELS,
            fingerprint_size=FINGERPRINT_SIZE,
            return_residual=args.use_residual,
            bias_init=args.bias_init,
            fused_modconv=args.fused_conv,
            fc_layers=args.fc_layers
        )
        RevealNet = models_modulated.StegaStampDecoder(
            IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
        )        

    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs) 
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs)) 

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)


def embed_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []
    all_code = []
    BCH_POLYNOMIAL = 137

    print("Fingerprinting the images...")
    fingerprints = generate_fingerprints(
        type = args.encode_method, 
        batch_size = BATCH_SIZE, 
        fingerprint_size = FINGERPRINT_SIZE, 
        secret = args.secret, 
        seed = args.seed, 
        diff_bits = args.diff_bits,
        manual_str = args.manual_str, 
        proportion = args.proportion, 
        identical = args.identical_fingerprints)
    fingerprints = fingerprints.to(device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)

    bitwise_accuracy = 0
    correct = 0
 
    for images, _ in tqdm(dataloader):
        images = images.to(device)

        if args.use_residual:
            residual = HideNet(fingerprints[: images.size(0)], images)
            fingerprinted_images = images + residual
        else:
            fingerprinted_images = HideNet(fingerprints[: images.size(0)], images)
        
        
        all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())

        if args.check: 
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()
            if args.encode_method == 'bch': 
                for sec in detected_fingerprints:
                    sec = np.array(sec.cpu())
                    if  FINGERPRINT_SIZE == 100:
                        BCH_BITS = 5
                        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
                        packet_binary = "".join([str(int(bit)) for bit in sec[:96]])
                    elif FINGERPRINT_SIZE == 50:
                        BCH_BITS = 2
                        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
                        packet_binary = "".join([str(int(bit)) for bit in sec[:48]])
                    packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
                    packet = bytearray(packet)
                    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
                    bitflips = bch.decode_inplace(data, ecc)
                    if bitflips != -1:
                        try:
                            correct += 1
                            code = data.decode("utf-8")
                            all_code.append(code)
                            continue
                        except:
                            all_code.append("Something went wrong")
                            continue
                    all_code.append('Failed to decode')

    dirname = args.output_dir
    if not os.path.exists(dirname):
        os.makedirs(dirname) 

    #if not os.path.exists(os.path.join(dirname, "fingerprinted_images")):
    

    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    
    for idx in range(len(all_fingerprinted_images)):
        image = all_fingerprinted_images[idx]
        fingerprint = all_fingerprints[idx]
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + "_hidden.png"
        save_image(image, os.path.join(args.output_dir, f"{filename}"), padding=0) 
        
        if args.encode_method == 'bch':
            code = all_code[idx]
            fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
            
        else:
            fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
            
    

    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
        if args.encode_method == 'bch':
            decode_acc = correct / len(all_fingerprints)
            print(f"Decoding accuracy on fingerprinted images: {decode_acc}")
        print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")

        #save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        #save_image(fingerprinted_images[:49], os.path.join(args.output_dir, "test_samples_fingerprinted.png"), nrow=7)
        #save_image(torch.abs(images - fingerprinted_images)[:49], os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=7)

    if args.test_save_file:
        with open(args.test_save_file,'a') as f:
            if args.encode_method == 'bch':
                f.write('Encode String: ' + str(args.secret) + ', ' + 'Test bitwise accuracy:' + str(bitwise_accuracy) + '\n')
            elif args.encode_method == 'seed':
                f.write('Encode Seed: ' + str(args.seed) + ', ' + 'Test bitwise accuracy:' + str(bitwise_accuracy) + '\n')
            elif args.encode_method == 'diff':
                f.write('Bits difference: ' + str(args.diff_bits) + ', ' + 'Test bitwise accuracy:' + str(bitwise_accuracy) + '\n')
            elif args.encode_method == 'entropy':
                f.write('Proportion: ' + str(args.proportion) + ', ' + 'Test bitwise accuracy:' + str(bitwise_accuracy) + '\n')
        f.close()

def main():
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args,arg))), '<')
    load_data()
    load_models()

    embed_fingerprints()

if __name__ == "__main__":
    main()
