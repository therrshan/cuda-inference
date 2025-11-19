#!/usr/bin/env python3
"""
Export GPT-2 weights from HuggingFace to binary format for C++ inference.
Saves weights in a simple binary format that C++ can read.
"""

import torch
import json
import struct
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from pathlib import Path

def export_gpt2_weights(model_name="gpt2", output_dir="../weights"):
    
    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = model.config.to_dict()
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_file}")

    vocab_dir = Path("../vocab")
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = tokenizer.get_vocab()
    vocab_file = vocab_dir / "vocab.json"
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)

    merges_file = vocab_dir / "merges.txt"
    with open(merges_file, 'w') as f:
        for merge in tokenizer.bpe_ranks.keys():
            f.write(f"{merge[0]} {merge[1]}\n")
    
    print(f"Saved vocabulary to {vocab_file}")
    print(f"Saved merges to {merges_file}")

    weights_file = output_path / "gpt2_weights.bin"
    
    print(f"\nExporting weights to {weights_file}...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    with open(weights_file, 'wb') as f:

        state_dict = model.state_dict()
        num_tensors = len(state_dict)
        f.write(struct.pack('I', num_tensors))
        
        for name, tensor in state_dict.items():
            weight = tensor.cpu().numpy()
            
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            f.write(struct.pack('I', len(weight.shape)))
            for dim in weight.shape:
                f.write(struct.pack('I', dim))
            
            dtype_map = {
                np.float32: 0,
                np.float16: 1,
                np.int32: 2,
                np.int64: 3
            }
            dtype_code = dtype_map.get(weight.dtype.type, 0)
            f.write(struct.pack('I', dtype_code))
      
            weight_bytes = weight.tobytes()
            f.write(struct.pack('Q', len(weight_bytes)))
            f.write(weight_bytes)
            
            print(f"  {name}: {weight.shape} ({weight.dtype})")
    
    print(f"\n Successfully exported weights to {weights_file}")
    print(f" File size: {weights_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n=== Model Architecture ===")
    print(f"Model: {model_name}")
    print(f"Vocab size: {config['vocab_size']}")
    print(f"Context length: {config['n_positions']}")
    print(f"Embedding dim: {config['n_embd']}")
    print(f"Num layers: {config['n_layer']}")
    print(f"Num heads: {config['n_head']}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

def print_weight_info(weights_file="../weights/gpt2_weights.bin"):

    with open(weights_file, 'rb') as f:
        num_tensors = struct.unpack('I', f.read(4))[0]
        print(f"Number of tensors: {num_tensors}\n")
        
        for i in range(min(5, num_tensors)): 
            name_len = struct.unpack('I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            ndims = struct.unpack('I', f.read(4))[0]
            shape = struct.unpack(f'{ndims}I', f.read(4 * ndims))

            dtype = struct.unpack('I', f.read(4))[0]
            dtype_names = {0: 'float32', 1: 'float16', 2: 'int32', 3: 'int64'}

            data_size = struct.unpack('Q', f.read(8))[0]
            f.seek(data_size, 1)
            
            print(f"{i+1}. {name}")
            print(f"   Shape: {shape}")
            print(f"   Dtype: {dtype_names[dtype]}")
            print(f"   Size: {data_size} bytes\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export GPT-2 weights for C++ inference')
    parser.add_argument('--model', type=str, default='gpt2', 
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'distilgpt2'],
                        help='Model to export')
    parser.add_argument('--output', type=str, default='../weights',
                        help='Output directory')
    parser.add_argument('--info', action='store_true',
                        help='Print info about existing weights file')
    
    args = parser.parse_args()
    
    if args.info:
        print_weight_info()
    else:
        export_gpt2_weights(args.model, args.output)