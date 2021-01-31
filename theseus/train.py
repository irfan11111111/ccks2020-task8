import argparse
import collections
import torch
import numpy as np
# import data_process.data_loaders as module_data
from data_process import military_data_process as module_data_process
from torch.utils.data import dataloader as module_dataloader
from base import base_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import transformers as optimization
import os
from utils.model_utils import get_restrain_crf
from model.replacement_scheduler import LinearReplacementScheduler,ConstantReplacementScheduler

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    # device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup data_set, data_process instances
    data_set = config.init_obj('data_set', module_data_process)
    # setup data_loader instances
    train_dataloader = config.init_obj('data_loader', module_dataloader, data_set.train_set,collate_fn=data_set.collate_fn)
    valid_dataloader = config.init_obj('data_loader', module_dataloader, data_set.valid_set, collate_fn=data_set.collate_fn)


    restrain = get_restrain_crf(9)
    # build model architecture, then print to console
    model_o = config.init_obj('model_arch', module_arch, restrain=restrain)
    # logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load("./saved_predecessor/pytorch_model.bin")
    state_dict = checkpoint['state_dict']
    model_o.load_state_dict(state_dict)

    model = config.init_obj('model_arch_theseus', module_arch, restrain=restrain,model_o=model_o)
    # logger.info(model)

    # get function handles of loss and metrics
    criterion = [getattr(module_loss, crit) for crit in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    bert_params = filter(lambda p: p.requires_grad, model.bert.parameters())
    fc_tags_params = filter(lambda p: p.requires_grad, model.fc_tags.parameters())
    crf_params = filter(lambda p: p.requires_grad, model.crf.parameters())
    ##只更新scc的参数
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                    not any(nd in n for nd in no_decay)], 'lr': 5e-5,'weight_decay': 0.0},
        {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                    any(nd in n for nd in no_decay)], 'lr': 5e-5,'weight_decay': 0.0},
    ]
    # optimizer = optimization.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = config.init_obj('optimizer', optimization, optimizer_grouped_parameters)

    lr_scheduler = config.init_obj('lr_scheduler', optimization.optimization, optimizer,
                                   num_training_steps=int(len(train_dataloader.dataset) / train_dataloader.batch_size))

    replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.bert.encoder,
                                                            replacing_rate=0.5,
                                                            replacing_steps=800)


    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      train_iter=train_dataloader,
                      valid_iter=valid_dataloader,
                      lr_scheduler=lr_scheduler,
                      replacing_rate_scheduler=replacing_rate_scheduler)

    trainer.train()


def run(config_file):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='1', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    print(config.config['model_arch']['type'].lower())


    main(config)


if __name__ == '__main__':
    run('configs/base_config.json')

