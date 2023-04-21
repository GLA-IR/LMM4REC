# coding: utf-8
"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='LATTICE_ete', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='sports', help='name of datasets')
    # parser.add_argument('--image_size', type=int, default=224, help='image size')
    # parser.add_argument('--multi_modal_encoder', type=str, default='vlmo',  help='multi-modal encoder')
    # parser.add_argument('--movie_file', type=str, default='data/Amazon_Sports_Dataset/raw_text.json', help='movie file')
    # parser.add_argument('--image_root', type=str, default='data/Amazon_Sports_Dataset/raw_image_id', help='image root')
    # parser.add_argument('--whole_word_masking', type=bool, default=True, help='whole word masking')
    #


    config_dict = {
        'gpu_id': 0,
        'image_size': 224,
        'multi_modal_encoder': 'vlmo',
        'movie_file': 'data/Amazon_Sports_Dataset/raw_text.json',
        'image_root': 'data/Amazon_Sports_Dataset/raw_image_id',
        'whole_word_masking': True,
        'mlm_prob': 0.15,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


