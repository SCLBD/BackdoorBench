# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import torch
import torch.utils.cpp_extension
import importlib
import hashlib
import shutil
from pathlib import Path

from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------


verbosity = 'brief' 

#----------------------------------------------------------------------------


def _find_compiler_bindir():
    patterns = [
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------


_cached_plugins = dict()

def get_plugin(module_name, sources, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']

    
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)

    try: 
        
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        
        verbose_build = (verbosity == 'full')

        
        
        
        
        
        #
        
        
        
        
        source_dirs_set = set(os.path.dirname(source) for source in sources)
        if len(source_dirs_set) == 1 and ('TORCH_EXTENSIONS_DIR' in os.environ):
            all_source_files = sorted(list(x for x in Path(list(source_dirs_set)[0]).iterdir() if x.is_file()))

            
            
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())
            build_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build) 
            digest_build_dir = os.path.join(build_dir, hash_md5.hexdigest())

            if not os.path.isdir(digest_build_dir):
                os.makedirs(digest_build_dir, exist_ok=True)
                baton = FileBaton(os.path.join(digest_build_dir, 'lock'))
                if baton.try_acquire():
                    try:
                        for src in all_source_files:
                            shutil.copyfile(src, os.path.join(digest_build_dir, os.path.basename(src)))
                    finally:
                        baton.release()
                else:
                    
                    
                    baton.wait()
            digest_sources = [os.path.join(digest_build_dir, os.path.basename(x)) for x in sources]
            torch.utils.cpp_extension.load(name=module_name, build_directory=build_dir,
                verbose=verbose_build, sources=digest_sources, **build_kwargs)
        else:
            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)
        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
