import os
import time
import numpy as np
import torch
import logging

from tqdm import tqdm
from transformers import BertTokenizer
import tensorboard_logger as tb_logger
import arguments

from lib import evaluation
from lib import image_caption
from lib.vse import VSEModel
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, compute_sim
from graph_lib import *

class DummyTBLogger:
    def configure(self, *args, **kwargs):
        pass

    def log_value(self, *args, **kwargs):
        pass

    # 添加其他你可能用到的 TensorBoard 方法
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    # 创建一个开关来控制是否使用 TensorBoard


use_tensorboard = False  # 设置为 False 来禁用 TensorBoard

if not use_tensorboard:
    import sys

    sys.modules['tb_logger'] = DummyTBLogger()
    tb_logger = DummyTBLogger()
else:
    import tensorboard_logger as tb_logger  # 实际的 TensorBoard 导入

def main():

    # Hyper Parameters
    parser = arguments.get_argument_parser()
    parser = extra_parameters(parser)
    opt = parser.parse_args()

    opt.model_name = opt.logger_name

    # set the gpu-id for training
    if not opt.multi_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # create the folder for logger and checkpoint
    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)

    # initialize logger
    logging.basicConfig(filename=os.path.join(opt.logger_name, 'train.txt'), filemode='w', 
                        format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(opt)  

    tb_logger.configure(opt.logger_name, flush_secs=5)

    # record parameters
    arguments.save_parameters(opt, opt.logger_name)

    # load tokenizer for TextEncoder
    # tokenizer = BertTokenizer.from_pretrained(opt.bert_path) 
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

    # get the train-set 
    train_loader = image_caption.get_train_loader(opt.data_path, tokenizer, opt.batch_size, opt.workers, opt)

    # get the test-set
    split = 'testall' if opt.dataset == 'coco' else 'test'
    test_loader = image_caption.get_test_loader(opt.data_path, split, tokenizer, opt.batch_size, opt.workers, opt)

    logger.info('Number of images for train-set: {}'.format(train_loader.dataset.num_images))

    # load the multi-modal model
    model = VSEModel(opt)

    start_epoch = 0

    # use the multi gpu
    if (not model.is_data_parallel) and opt.multi_gpu:
        model.make_data_parallel()

    best_rsum = 0

    # start the training process
    # for epoch in range(start_epoch, opt.num_epochs):
    for epoch in tqdm(range(start_epoch, opt.num_epochs), desc="Training Progress"):
        if epoch == 0:
            logger.info('Log saving path: ' + opt.logger_name)
            logger.info('Models saving path: ' + opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # set hard negative for vse loss
        if (epoch >= opt.vse_mean_warmup_epochs):
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)

        # train for one epoch
        train(opt, train_loader, model, epoch)

        # evaluate on test set for every epoch
        rsum = validate(opt, test_loader, model)

        # remember best rsum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        logger.info("Epoch: [{}], Best rsum: {:.1f}".format(epoch, best_rsum))

        # save the checkpoint
        state = {'model': model.state_dict(), 'opt': opt, 'epoch': epoch + 1, 'rsum': rsum, 'Eiters': model.Eiters}
        save_checkpoint(state, prefix=opt.model_name)

    logger.info('Train finish.')    

    # evaluation after training process
    logger.info('Evaluate the model')
    
    base = opt.logger_name
    logging.basicConfig(filename=os.path.join(base, 'eval.txt'), filemode='w', 
                        format='%(asctime)s %(message)s', level=logging.INFO, force=True)
    logger = logging.getLogger()

    logger.info('Evaluating {}'.format(base))
    model_path = os.path.join(base, 'model_best.pth')

    # Save the final results for computing ensemble results
    save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset)) if opt.save_results else None

    if opt.dataset == 'coco':
        # Evaluate COCO 5-fold 1K
        # Evaluate COCO 5K
        evaluation.evalrank(model_path, opt=opt, tokenizer=tokenizer, model=model, split='testall', fold5=True, save_path=save_path)
    else:
        # Evaluate Flickr30K
        evaluation.evalrank(model_path, opt=opt, tokenizer=tokenizer, model=model, split='test', fold5=False, save_path=save_path)

    logger.info('Evaluation finish')    


def train(opt, train_loader, model, epoch):
   
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    if epoch == 0:
        logger.info('image encoder trainable parameters: {}M'.format(count_params(model.img_enc)))
        logger.info('txt encoder trainable parameters: {}M'.format(count_params(model.txt_enc)))
        logger.info('criterion trainable parameters: {}M'.format(count_params(model.criterion)))

    end = time.time()
    repeat_list = []
    n_batch = len(train_loader.dataset) // opt.batch_size 

    model.train_start()    

    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        # images, img_lengths, captions, lengths, ids, img_ids, repeat = train_data
        images, img_lengths, captions, teacher_aux_captions, teacher_captions, lengths, ids, img_ids, repeat = train_data

        model.train_emb(images, captions, teacher_aux_captions, teacher_captions, opt, lengths, image_lengths=img_lengths, img_ids=img_ids)
  
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if model.Eiters % opt.log_step == 0:               
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Batch-Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    .format(
                    epoch, i+1, n_batch, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

    return repeat_list


def validate(opt, val_loader, model):

    logger = logging.getLogger(__name__)
    model.val_start()

    with torch.no_grad():

        # compute the encoding for all the validation images and captions
        img_embs, cap_embs = encode_data(model, val_loader, opt.log_step, logging.info)

    # have repetitive image features
    img_embs = img_embs[::5]
    npts = img_embs.shape[0]

    sims = compute_sim(img_embs, cap_embs)

    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logging.info("Image to text (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1, r5, r10))

    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logging.info("Text to image (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1i, r5i, r10i))

    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(round(currscore, 1)))

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    if opt.dataset == 'coco':
        if opt.save_fold5:
            # logger.info('Start evaluation on 5-fold 1K for MSCOCO.')
            results = []
            sims_all = sims
            for i in range(5):
                sims = sims_all[i * 1000:(i + 1) * 1000, i * 5000:(i + 1) * 5000]

                npts = sims.shape[0]
                r, rt0 = i2t(npts, sims, return_ranks=True)
                ri, rti0 = t2i(npts, sims, return_ranks=True)

                # logger.info("Image to text: %.1f, %.1f, %.1f" % r[:3])
                # logger.info("Text to image: %.1f, %.1f, %.1f" % ri[:3])

                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            # logger.info("-----------------------------------")
            # logger.info("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            logger.info("5 folds mean metrics: rsum: %.1f" % (mean_metrics[12]))
            logger.info("Image to text (R@1, R@5, R@10): %.1f %.1f %.1f" % mean_metrics[:3])
            logger.info("Text to image (R@1, R@5, R@10): %.1f %.1f %.1f" % mean_metrics[5:8])

            return mean_metrics[12]

    return currscore


def save_checkpoint(state, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 2

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # don't save checkpoint
            # torch.save(state, prefix + filename)
            torch.save(state, os.path.join(prefix, 'model_best.pth'))
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    logger = logging.getLogger(__name__)

    decay_rate = opt.decay_rate
    lr_schedules = opt.lr_schedules

    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * decay_rate
            param_group['lr'] = new_lr
            logger.info('new lr: {}'.format(new_lr))


def count_params(model):

    # The unit is M (million)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    params = round(params/(1024**2), 2)

    return params


if __name__ == '__main__':
    
    main()
