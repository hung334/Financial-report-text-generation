import torch
import ntemp.opts as opts
from configargparse import ArgumentParser
from ntemp.trainer import trainer as Trainer
import logging

def train(args):
    
    global decay_count
    global display_loss
    
    trainer = Trainer(args)    
    prev_valloss, best_valloss = float("inf"), float("inf")
    decayed = False
    if args.prev_loss is not None:
        prev_valloss = args.prev_loss
        if args.best_loss is None:
            best_valloss = prev_valloss
        else:
            decayed = True
            best_valloss = args.best_loss
        print "starting with", prev_valloss, best_valloss
        logging.info("starting with %f, and best valid loss is %f", prev_valloss, best_valloss)

    for epoch in range(1, args.epochs + 1):
        if epoch > args.no_ar_epochs and not trainer.net.ar and decayed:
            trainer.net.ar = True

            # hack
            if args.word_ar and not trainer.net.word_ar:
                print "turning on word ar..."
                logging.info("turning on word ar...")
                trainer.net.word_ar = True

        print "ar:", trainer.net.ar
        logging.info("ar: %s", trainer.net.ar)
        
        
        trainloss = trainer.train(epoch)
        valloss = trainer.test(epoch)
        
        display_loss.append([trainloss,valloss])
        
        if valloss < best_valloss:
            best_valloss = valloss
            if len(args.save) > 0:
                print "saving to", args.save
                logging.info("saving to %s", args.save)
                state = {"opt": args, 
                         "state_dict": trainer.net.state_dict(),
                         "lr": args.lr, 
                         "dict": trainer.corpus.dictionary}
                torch.save(state, args.save + "." + str(int(decayed)))
                decay_count = 0
        else:
            decay_count+=1
        
        print'decay_count:',decay_count
        logging.info("decay_count: %d", decay_count)
        
        print'now lr:',args.lr
        logging.info("now lr: %f", args.lr)
        
        if ((args.optim == "sgd" and valloss >= prev_valloss) or (args.onmt_decay and decayed)) and decay_count>=args.max_decay_count:
            decayed = True
            args.lr *= args.lr_decay
            if args.ar_after_decay and not trainer.net.ar:
                trainer.net.ar = True
                # hack
                if args.word_ar and not trainer.net.word_ar:
                    print "turning on word ar..."
                    logging.info("turning on word ar...")
                    trainer.net.word_ar = True
            print "decaying lr to:", args.lr
            logging.info("decaying lr to: %f", args.lr)
            if args.lr < 1e-5:
                break
            decay_count = 0
            
        prev_valloss = valloss
        
        if args.cuda:
            print "ugh...."
            shmocals = locals()
            for shk in shmocals.keys():
                shv = shmocals[shk]
                if hasattr(shv, "is_cuda") and shv.is_cuda:
                    shv = shv.cpu()
            print "done!"
            logging.info("Done")
            print
        else:
            print
               
def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser

def main():   
    parser = _get_parser()
    opt = parser.parse_args()
    logging.basicConfig(filename= opt.logs,
                        level=logging.INFO,
                        format='%(asctime)s : %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                       )    
    logging.info('Ready for training...')
    train(opt)
    

if __name__ == "__main__":
    
    decay_count = 0
    display_loss = []
    
    main()
    
    import numpy as np
    display_loss = np.array(display_loss)
    np.save('display_loss',display_loss)

