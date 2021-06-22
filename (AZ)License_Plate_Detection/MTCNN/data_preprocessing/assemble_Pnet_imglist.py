import os
import sys
sys.path.append(os.getcwd())
import numpy.random as npr
import numpy as np


pnet_postive_file = 'anno_store/pos_12_val.txt'
pnet_part_file = 'anno_store/part_12_val.txt'
pnet_neg_file = 'anno_store/neg_12_val.txt'
# pnet_landmark_file = './anno_store/landmark_12.txt'
imglist_filename = 'anno_store/imglist_anno_12_val.txt'


def assemble_data(output_file, anno_file_list=[]):

    #assemble the pos, neg, part annotations to one file

    if len(anno_file_list)==0:
        return 0

    if os.path.exists(output_file):
        os.remove(output_file)

    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            print(anno_file)
            anno_lines = f.readlines()

        base_num = 250000

        if len(anno_lines) > base_num * 3:
            idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
        elif len(anno_lines) > 100000:
            idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
        else:
            idx_keep = np.arange(len(anno_lines))
            np.random.shuffle(idx_keep)
        chose_count = 0
        with open(output_file, 'a+') as f:
            for idx in idx_keep:
                # write lables of pos, neg, part images
                f.write(anno_lines[idx])
                chose_count+=1

    return chose_count


if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
