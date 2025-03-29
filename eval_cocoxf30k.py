import os
import torch
import argparse
import logging
from lib import evaluation


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default='C:/Users/user/Desktop/通用检索/data', type=str, help='path to datasets')
    parser.add_argument('--dataset', default='f30k', type=str, help='the dataset choice, coco or f30k')
    parser.add_argument('--save_results', type=int, default=0, help='if save the similarity matrix for ensemble')
    parser.add_argument('--gpu-id', type=int, default=0, help='the gpu-id for evaluation')
    
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    if opt.dataset == 'coco':
        weights_bases = [
            'runs/coco_test'
        ]
    else:
        weights_bases = [
            # 'runs/f30k_test'
            'runs/coco_test_5'

            # 'runs/f30k_test_5'

        ]

    for base in weights_bases:

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        
        # Save the final similarity matrix 
        if opt.save_results:  
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            # Evaluate COCO 5K
            evaluation.evalrank_cocoxf30k(model_path, split='testall', fold5=True, save_path=save_path, data_path=opt.data_path)
        else:
            # Evaluate Flickr30K

            # INFO: lib.evaluation:rsum: 533.1
            # INFO: lib.evaluation:Image to text(R @ 1, R @ 5, R @ 10): 84.5 97.1 98.5
            # INFO: lib.evaluation:Text to image(R @ 1, R @ 5, R @ 10): 69.8 89.4 93.8
            evaluation.evalrank_cocoxf30k(model_path, split='test', fold5=False, save_path=save_path, data_path=opt.data_path)


if __name__ == '__main__':
    
    main()

