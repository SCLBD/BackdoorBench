import bchlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

BCH_POLYNOMIAL = 137

def generate_fingerprints_from_bch(fingerprint_size, secret='abcd'):
    if fingerprint_size == 100: 
        BCH_BITS = 5
        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
        if len(secret) > 7:
            print('Error: Can only encode 56bits (7 characters) with ECC')
            return
        data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')#转化为bytearray对象        
    elif fingerprint_size == 50: 
        BCH_BITS = 2
        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
        if len(secret) > 4:
            print('Error: Can only encode 32bits (4 characters) with ECC')
            return
        data = bytearray(secret + ' ' * (4 - len(secret)), 'utf-8')#转化为bytearray对象
    else:
        raise ValueError('fingerprint_size must be 100 or 50!')
    ecc = bch.encode(data)#获得对应编码
    packet = data + ecc#对数据进行编码
    packet_binary = ''.join(format(x, '08b') for x in packet)#转换成二进制
    fingerprints = [int(x) for x in packet_binary]
    if fingerprint_size == 100: 
        fingerprints.extend([0, 0, 0, 0])
    elif fingerprint_size == 50: 
        fingerprints.extend([0, 0])

    return fingerprints

uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2) #(B, 100)
    return z

def generate_fingerprints(
    type, 
    batch_size, 
    fingerprint_size, 
    secret='abcd', 
    gd_secret='abcd', 
    seed=0, 
    diff_bits=1, 
    manual_str=None,
    proportion=1.0, 
    identical=True, 
    compare=True):

    assert type in ['bch', 'seed', 'diff', 'manual', 'entropy'], 'type must be one of [bch, seed, diff, manual]!'
    
    if type == 'bch':
        print("Using bch code with secret string:", secret)
        fingerprints = generate_fingerprints_from_bch(fingerprint_size, secret)
        if compare:
            compare_bit_difference(fingerprints, fingerprint_size, gd_secret)
        fingerprints = torch.tensor(fingerprints, dtype=torch.float).unsqueeze(0).expand(batch_size, fingerprint_size)

    elif type == 'seed':
        print("Generating fingerprints from seed:", seed)
        torch.manual_seed(seed)
        fingerprints = generate_random_fingerprints(fingerprint_size, 1)
        fingerprints = fingerprints.view(1, fingerprint_size)
        if compare: 
            compare_bit_difference(fingerprints, fingerprint_size, gd_secret)
        fingerprints = fingerprints.expand(batch_size, fingerprint_size)
        if not identical: 
            print('Not using identical fingerprints!!')
            fingerprints = generate_random_fingerprints(fingerprint_size, batch_size)
            fingerprints = fingerprints.view(batch_size, fingerprint_size)

    elif type == 'diff':
        print("Using bit difference from ground truth string:", secret)
        gd_list = generate_fingerprints_from_bch(fingerprint_size, secret)
        gd_list = np.array(gd_list)
        fingerprints = gd_list.copy()

        
        indexes = np.random.choice(fingerprint_size, size=diff_bits, replace=False)
        for ind in indexes:
            fingerprints[ind] = 1 - gd_list[ind]
        print('number of bits overlap from original string {} is: {}'.format(secret, np.sum(fingerprints==gd_list)))
        fingerprints = fingerprints.tolist()
        fingerprints = torch.tensor(fingerprints, dtype=torch.float).unsqueeze(0).expand(batch_size, fingerprint_size)

    elif type == 'entropy':
        print("Using entropy to generate encode sequence, the proportion of diffs is", proportion)
        gd_list = generate_fingerprints_from_bch(fingerprint_size, secret)
        gd_list = np.array(gd_list)
        fingerprints = gd_list.copy()

        num = int(fingerprint_size * proportion)
        if num:
            indexes = np.random.choice(fingerprint_size, size=num, replace=False)
            for ind in indexes:
                fingerprints[ind] = 1 - gd_list[ind]
        print('number of bits overlap from original string {} is: {}'.format(secret, np.sum(fingerprints==gd_list)))
        fingerprints = fingerprints.tolist()
        fingerprints = torch.tensor(fingerprints, dtype=torch.float).unsqueeze(0).expand(batch_size, fingerprint_size)        

    elif type == 'manual':
        print("Using manually string defined by user!")
        manual_str = manual_str.strip('[]').split(',') 
        fingerprints = list(map(float, manual_str)) 
        assert len(fingerprints) == fingerprint_size, 'The length of the manual string does not match the fingerprint size!'
        
        if compare: 
            compare_bit_difference(fingerprints, fingerprint_size, gd_secret)
        
        fingerprints = torch.tensor(fingerprints, dtype=torch.float).unsqueeze(0).expand(batch_size, fingerprint_size)

    return fingerprints

def compare_bit_difference(current_fp, fingerprint_size, secret='abcd'):
        print('Generating ground truth from bch code with string:', secret)
        gd_list = generate_fingerprints_from_bch(fingerprint_size, secret)
        gd_list = np.array(gd_list)
        try: 
            fp_list = current_fp.squeeze().numpy()
        except: 
            fp_list = np.array(current_fp)
        bits_overlap = np.sum(fp_list==gd_list)
        print('number of bits overlap from original string {} is: {}'.format(secret, bits_overlap))

if __name__  == '__main__':
    a = generate_fingerprints(
    type = 'bch' ,
    batch_size = 1,  
    fingerprint_size = 50, 
    secret='abch', 
    gd_secret='abcd', 
    seed=0, 
    diff_bits=1, 
    manual_str=None,
    proportion=1.0, 
    identical=True, 
    compare=True)
    print(a)