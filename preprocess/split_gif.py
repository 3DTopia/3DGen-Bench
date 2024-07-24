import os
import argparse
from PIL import Image, ImageSequence

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['rgb', 'normal', 'geo'], required=True)
    parser.add_argument('--task', '-t', type=str, choices=['text2shape', 'image2shape'])
    parser.add_argument('--method', '-m', type=str, required=True)
    parser.add_argument("--in_dir", '-i', type=str, default="data/surrounding_views")
    parser.add_argument("--out_dir", '-o', type=str, default="data/images")
    args = parser.parse_args()
    
    mode = args.mode
    task = args.task
    method = args.method
    in_dir = args.in_dir
    out_dir = args.out_dir

    file_names = os.listdir(os.path.join(in_dir, method))
    for fname in file_names:
        if not fname.endswith(f"_{mode}.gif"):
            continue
        idx = fname.split('_')[0]
        gif_path = os.path.join(in_dir, method, fname)

        img_dir = os.path.join(out_dir, task, method, idx, mode)
        if os.path.exists(img_dir):
            continue
        os.makedirs(img_dir, exist_ok=True)

        gif = Image.open(gif_path)
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
        for i, frame in enumerate(frames):
            frame.save(os.path.join(img_dir, f'{i}.png'), 'PNG')