import argparse
import glob
import PIL
import bchlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument("--output_dir", type=str, help="Path to save watermarked images to.")
parser.add_argument("--image_resolution", type=int, required=True, help="Height and width of square images.")
parser.add_argument("--decoder_path", type=str, required=True, help="Path to trained StegaStamp decoder.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--bch", action='store_true', help="Use bch code")
parser.add_argument('--secret', type=str, default='CUHKSZ!')

args = parser.parse_args()

import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2) #(B, 100)
    return z

if args.cuda != -1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def load_decoder():
    global RevealNet
    global FINGERPRINT_SIZE

    from models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path)
    FINGERPRINT_SIZE = state_dict["dense.2.weight"].shape[0]

    RevealNet = StegaStampDecoder(args.image_resolution, 3, FINGERPRINT_SIZE)
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path, **kwargs))
    RevealNet = RevealNet.to(device)


def load_data():
    global dataset, dataloader

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")


def extract_fingerprints():
    
    all_fingerprints = []
    all_code = []

    BATCH_SIZE = args.batch_size
    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    print("Generating Ground Truth...")
    torch.manual_seed(args.seed)

    #生成正确的编码用于计算指标
    if not args.bch: #如果不采用BCH编码，默认对所有图片采用相同指纹
        fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, 1)
        fingerprints = fingerprints.view(1, FINGERPRINT_SIZE).expand(BATCH_SIZE, FINGERPRINT_SIZE)
        fingerprints = fingerprints.to(device)
    else: #采用BCH编码
        print("Using bch code along with secret string:", args.secret)
        if len(args.secret) > 7:
            print('Error: Can only encode 56bits (7 characters) with ECC')
            return
        data = bytearray(args.secret + ' ' * (7 - len(args.secret)), 'utf-8')#转化为bytearray对象
        ecc = bch.encode(data)#获得对应编码
        packet = data + ecc#对数据进行编码
        packet_binary = ''.join(format(x, '08b') for x in packet)#转换成二进制
        fingerprints = [int(x) for x in packet_binary]
        fingerprints.extend([0, 0, 0, 0])
        fingerprints = torch.tensor(fingerprints, dtype=torch.float).unsqueeze(0).expand(BATCH_SIZE, FINGERPRINT_SIZE)
        fingerprints = fingerprints.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    bitwise_accuracy = 0
    correct = 0

    for images, _ in tqdm(dataloader):
        images = images.to(device)

        detected_fingerprints = RevealNet(images)
        detected_fingerprints = (detected_fingerprints > 0).long()
        bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()
        if args.bch: 
            for sec in detected_fingerprints:
                sec = np.array(sec.cpu())
                packet_binary = "".join([str(int(bit)) for bit in sec[:96]])
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
        
        all_fingerprints.append(detected_fingerprints.detach().cpu())

    
    
    
    
    

    
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    

    
    for idx in range(len(all_fingerprints)):
        
        fingerprint = all_fingerprints[idx]
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".png"
        if args.bch:
            code = all_code[idx]
            
            
            fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
            
        else:
            fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
            
    

    bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
    if args.bch:
        decode_acc = correct / len(all_fingerprints)
        print(f"Decoding accuracy on fingerprinted images: {decode_acc}")
    print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")


if __name__ == "__main__":
    load_decoder()
    load_data()
    extract_fingerprints()
