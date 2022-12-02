import torch
import ntemp.opts as opts
from configargparse import ArgumentParser
from ntemp.trainer import trainer as Trainer
import logging

def segment(args):
    trainer = Trainer(args)     
    trainer.label_train()    
    return


def _get_parser():
    parser = ArgumentParser(description='segment.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    opts.segs_opts(parser)
    return parser

def main():   
    parser = _get_parser()
    opt = parser.parse_args()
    logging.basicConfig(filename= opt.logs,
                        level=logging.INFO,
                        format='%(asctime)s : %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                       )    
    logging.info('Ready for segment...')
    segment(opt)
    

if __name__ == "__main__":
    main()