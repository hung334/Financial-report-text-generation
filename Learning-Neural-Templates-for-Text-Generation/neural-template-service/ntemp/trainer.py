import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from ntemp import labeled_data
from ntemp.utils.utils import logsumexp1, make_fwd_constr_idxs, make_bwd_constr_idxs
from data.utils import get_wikibio_poswrds, get_e2e_poswrds
from ntemp.utils import infc
from ntemp.chsmm import HSMM
from ntemp.utils.preprocess import make_combo_targs, get_uniq_fields, make_masks



class trainer(object):    
    def __init__(self, args):
        torch.manual_seed(args.seed)
        #print 'torch.cuda.is_available():',torch.cuda.is_available()
        if torch.cuda.is_available():
            #print torch.cuda.device_count()
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
            if not args.cuda:
                print "WARNING: You have a CUDA device, so you should probably run with -cuda"
            else:
                torch.cuda.manual_seed(args.seed)

        '''
        Build model
        Returns: 
            hsmm model
        '''
        saved_args, saved_state = None, None
        if len(args.load) > 0:
            if args.cuda:
                saved_stuff = torch.load(args.load)
            else:
                saved_stuff = torch.load(args.load, map_location='cpu')
            saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
	    
            corpus = labeled_data.SentenceCorpus(saved_args.data, saved_args.bsz, thresh=saved_args.thresh, add_bos=False, add_eos=False, test=saved_args.test)

            saved_args.cuda = args.cuda # set the same as the generation arguments.
            for k, v in args.__dict__.iteritems():
                if k not in saved_args.__dict__:
                    saved_args.__dict__[k] = v
            net = HSMM(corpus, len(corpus.dictionary), corpus.ngen_types, saved_args)
            # for some reason selfmask breaks load_state
            del saved_state['trans_logprobs.selfmask']
            net.load_state_dict(saved_state, strict=False)
            args.pad_idx = corpus.dictionary.word2idx["<pad>"]
            if args.fine_tune:
                for name, param in net.named_parameters():
                    if name in saved_state:
                        param.requires_grad = False
        else:
            corpus = labeled_data.SentenceCorpus(args.data, args.bsz, thresh=args.thresh, add_bos=False, add_eos=False, test=args.test)
            args.pad_idx = corpus.dictionary.word2idx["<pad>"]        
            net = HSMM(corpus, len(corpus.dictionary), corpus.ngen_types, args)
            
        if args.cuda:
            net = net.cuda()            
        '''
        Build Optimizer
        Returns:
            optimizer.
        '''
        if args.optim == "adagrad":
            optalg = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
            for group in optalg.param_groups:
                for p in group['params']:
                    optalg.state[p]['sum'].fill_(0.1)
        elif args.optim == "rmsprop":
            optalg = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        elif args.optim == "adam":
            optalg = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        else:
            optalg = None    

            
            
        if len(args.gen_from_fi) == 0:
            # make constraint things from labels
            train_cidxs, train_fwd_cidxs = [], []
            for i in xrange(len(corpus.train)):
                x, constrs, _, _, _ = corpus.train[i]
                train_cidxs.append(make_bwd_constr_idxs(args.L, x.size(0), constrs))
                train_fwd_cidxs.append(make_fwd_constr_idxs(args.L, x.size(0), constrs))
                
        '''
        set class parameters.
        '''
        self.args = args
        self.saved_args, self.saved_state = saved_args, saved_state
        self.corpus = corpus
        self.train_cidxs = train_cidxs
        self.net = net
        self.optalg = optalg
        
        
    def train(self, epoch):
        # Turn on training mode which enables dropout.
        self.net.train()
        neglogev = 0.0 #  negative log evidence
        nsents = 0
        trainperm = torch.randperm(len(self.corpus.train))
        nmini_batches = min(len(self.corpus.train), self.args.max_mbs_per_epoch)
        for batch_idx in xrange(nmini_batches):
            self.net.zero_grad()
            x, _, src, locs, inps = self.corpus.train[trainperm[batch_idx]]
            cidxs = self.train_cidxs[trainperm[batch_idx]] if epoch <= self.args.constr_tr_epochs else None

            seqlen, bsz = x.size()
            
            nfields = src.size(1)
            if seqlen < self.args.L or seqlen > self.args.max_seqlen:
                continue

            combotargs = make_combo_targs(locs, x, self.args.L, nfields, self.corpus.ngen_types)
            # get bsz x nfields, bsz x nfields masks
            fmask, amask = make_masks(src, self.args.pad_idx, max_pool=self.args.max_pool)

            uniqfields = get_uniq_fields(src, self.args.pad_idx) # bsz x max_fields
            
            if self.args.cuda:
                combotargs = combotargs.cuda()
                if cidxs is not None:
                    cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
                src = src.cuda()
                inps = inps.cuda()
                fmask, amask = fmask.cuda(), amask.cuda()
                uniqfields = uniqfields.cuda()
            
            init_logps, trans_logps, len_logprobs, fwd_obs_logps = self.net(Variable(src), Variable(amask), Variable(uniqfields), 
                                                                            seqlen, 
                                                                            Variable(inps), Variable(fmask), Variable(combotargs), bsz)
            # get T+1 x bsz x K beta quantities
            beta, beta_star = infc.just_bwd(trans_logps, fwd_obs_logps,len_logprobs, constraints=cidxs)
            log_marg = logsumexp1(beta_star[0] + init_logps).sum() # bsz x 1 -> 1


            neglogev -= log_marg.data[0] 
            lossvar = -log_marg/bsz
            lossvar.backward()
            torch.nn.utils.clip_grad_norm(self.net.parameters(), self.args.clip)
            if self.optalg is not None:
                self.optalg.step()
            else:
                for p in self.net.parameters():
                    if p.grad is not None:
                        p.data.add_(-self.args.lr, p.grad.data)

            nsents += bsz
            #print "nsents",nsents
            #print "neglogev",neglogev
            if (batch_idx+1) % self.args.log_interval == 0:
                print "batch %d/%d | train neglogev %g " % (batch_idx+1,
                                                            nmini_batches,
                                                            neglogev/nsents)
                logging.info("batch %d/%d | train neglogev %g ", batch_idx+1,
                                                                 nmini_batches,
                                                                 neglogev/nsents)
        #try:
        print "epoch %d | train neglogev %g " % (epoch, neglogev/nsents)
        logging.info("epoch %d | train neglogev %g ", epoch, neglogev/nsents)
        #except :
        #    print "epoch %d | train neglogev %g " % (epoch, 0)
        #    logging.info("epoch %d | train neglogev %g ", epoch, 0)
        
        return neglogev/nsents 
        
        #print "epoch %d | train neglogev %g " % (epoch, neglogev/nsents)
        #logging.info("epoch %d | train neglogev %g ", epoch, neglogev/nsents)
        #print "nsents",nsents
        #print "neglogev",neglogev
        #try:
        #    return neglogev/nsents 
        #except:
        #    return 0
    
    def test(self, epoch):
        self.net.eval()
        neglogev = 0.0
        nsents = 0

        for i in xrange(len(self.corpus.valid)):
            x, _, src, locs, inps = self.corpus.valid[i]
            cidxs = None

            seqlen, bsz = x.size()
            nfields = src.size(1)
            if seqlen < self.args.L or seqlen > self.args.max_seqlen:
                continue

            combotargs = make_combo_targs(locs, x, self.args.L, nfields, self.corpus.ngen_types)
            # get bsz x nfields, bsz x nfields masks
            fmask, amask = make_masks(src, self.args.pad_idx, max_pool=self.args.max_pool)

            uniqfields = get_uniq_fields(src, self.args.pad_idx) # bsz x max_fields

            if self.args.cuda:
                combotargs = combotargs.cuda()
                if cidxs is not None:
                    cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
                src = src.cuda()
                inps = inps.cuda()
                fmask, amask = fmask.cuda(), amask.cuda()
                uniqfields = uniqfields.cuda()

            init_logps, trans_logps, len_logprobs, fwd_obs_logps = self.net(Variable(src, volatile=True), Variable(amask, volatile=True), 
                                                                            Variable(uniqfields, volatile=True), seqlen, 
                                                                            Variable(inps, volatile=True), Variable(fmask, volatile=True), 
                                                                            Variable(combotargs, volatile=True), bsz)
            # get T+1 x bsz x K beta quantities
            beta, beta_star = infc.just_bwd(trans_logps, fwd_obs_logps,
                                            len_logprobs, constraints=cidxs)
            log_marg = logsumexp1(beta_star[0] + init_logps).sum() # bsz x 1 -> 1           
            neglogev -= log_marg.data[0]
            nsents += bsz
        print "epoch %d | valid neglogev %g" % (epoch, neglogev/nsents)
        logging.info("epoch %d | valid neglogev %g", epoch, neglogev/nsents)
        
            
        return neglogev/nsents
    
    def label_train(self):
        self.net.ar = self.saved_args.ar_after_decay and not self.args.no_ar_for_vit
        print "btw, net.ar:", self.net.ar
        seg_file = open(self.args.seg_file, 'w')
        
        for i in xrange(len(self.corpus.train)):
            x, _, src, locs, inps = self.corpus.train[i]
            fwd_cidxs = None

            seqlen, bsz = x.size()
            nfields = src.size(1)
            if seqlen <= self.saved_args.L: #or seqlen > args.max_seqlen:
                continue

            combotargs = make_combo_targs(locs, x, self.saved_args.L, nfields, self.corpus.ngen_types)
            # get bsz x nfields, bsz x nfields masks
            fmask, amask = make_masks(src, self.saved_args.pad_idx, max_pool=self.saved_args.max_pool)
            uniqfields = get_uniq_fields(src, self.saved_args.pad_idx) # bsz x max_fields

            if self.args.cuda:
                combotargs = combotargs.cuda()
                if fwd_cidxs is not None:
                    fwd_cidxs = [tens.cuda() if tens is not None else None for tens in fwd_cidxs]
                src = src.cuda()
                inps = inps.cuda()
                fmask, amask = fmask.cuda(), amask.cuda()
                uniqfields = uniqfields.cuda()

            
            init_logps, trans_logps, len_logprobs, fwd_obs_logps = self.net(Variable(src, volatile=True), Variable(amask, volatile=True), 
                                                                            Variable(uniqfields, volatile=True), seqlen, 
                                                                            Variable(inps, volatile=True), Variable(fmask, volatile=True), 
                                                                            Variable(combotargs, volatile=True), bsz)            
            
            bwd_obs_logprobs = infc.bwd_from_fwd_obs_logprobs(fwd_obs_logps.data)
            seqs = infc.viterbi(init_logps.data, trans_logps.data, bwd_obs_logprobs,
                                [t.data for t in len_logprobs], constraints=fwd_cidxs)
            
            for b in xrange(bsz):
                words = [self.corpus.dictionary.idx2word[w] for w in x[:, b]]
                for (start, end, label) in seqs[b]:
                    seg_file.writelines("%s|%d " % (" ".join(words[start:end]), label))
                    logging.info("%s|%d", " ".join(words[start:end]), label)
                    print "%s|%d" % (" ".join(words[start:end]), label),
                seg_file.write(os.linesep)
                print
        seg_file.close()
    
