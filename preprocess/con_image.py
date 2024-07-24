import os
import argparse
import numpy as np
from PIL import Image

def create_square_images(img_lst, ncols):
    img_array = []
    img_row_array = []
    for img_path in img_lst:
        curr_img = Image.open(img_path).convert('RGB')
        curr_img = np.array(curr_img)
        # print(curr_img.shape)
        # if len(curr_img) != 512:
        #     continue
        if len(img_row_array) < ncols - 1:
            img_row_array.append(curr_img)
        else: # len(img_row_array) == ncols
            assert len(img_row_array) == ncols - 1
            img_row_array.append(curr_img)
            img_array.append(np.concatenate(img_row_array, axis=1))
            img_row_array = []
    return np.concatenate(img_array, axis=0)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['rgb', 'normal', 'geo'], required=True)
    parser.add_argument("--task", choices=['text2shape', 'image2shape'], required=True)
    parser.add_argument('--method', '-m', type=str, default=None)
    parser.add_argument("--in_dir", '-i', type=str, default="data/images")
    parser.add_argument("--out_dir", '-o', type=str, default="data/con_images")
    parser.add_argument("--nsample", '-N', type=int, default=4)
    parser.add_argument("--ncols", type=int, default=2)
    args = parser.parse_args()

    mode = args.mode
    task = args.task
    in_dir = args.in_dir
    out_dir = args.out_dir
    sampled_views = [i for i in range(0, 40, 40//args.nsample)]
    ncols = args.ncols
    
    if args.method:
        methods = [args.method]
    else:
        methods = os.listdir(os.path.join(in_dir, task))
    print(methods)

    for method in methods:
        idxs = os.listdir(os.path.join(in_dir, task, method))
        os.makedirs(os.path.join(out_dir, task, method), exist_ok=True)
        for idx in idxs:
            out_path = os.path.join(out_dir, task, method, f"{idx}_{mode}.png")
            if os.path.exists(out_path):
                continue

            img_list = []
            for view in sampled_views:
                img_path = os.path.join(in_dir, task, method, idx, mode, f'{view}.png')
                if not os.path.exists(img_path):
                    img_path = os.path.join(in_dir, task, method, idx, mode, f'0.png')
                img_list.append(img_path)
            con_img = create_square_images(img_list, ncols)
            Image.fromarray(con_img).convert("RGB").save(out_path)