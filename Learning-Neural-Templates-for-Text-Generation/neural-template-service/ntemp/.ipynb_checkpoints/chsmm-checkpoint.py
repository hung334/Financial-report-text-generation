import sys
import os
import math
import random
import argparse
from collections import defaultdict, Counter
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import ntemp.labeled_data as labeled_data
from ntemp.utils.utils import logsumexp1, make_fwd_constr_idxs, make_bwd_constr_idxs, backtrace3, backtrace

class HSMM(nn.Module):
    def __init__(self, corpus, wordtypes, gentypes, opt):
        super(HSMM, self).__init__()
        
        self.initrange = opt.initrange
        self.ar = False
        self.pad_idx = opt.pad_idx
        self.corpus = corpus
        
        
        self.lut = nn.Embedding(wordtypes, opt.emb_size, padding_idx=opt.pad_idx)    
        self.encoder = Encoder(opt.emb_size, self.lut, opt.max_pool)        
        self.trans_logprobs = Transition(opt.smaller_cond_dim, opt.emb_size, 
                                         opt.K, opt.Kmul, opt.cond_A_dim, opt.A_dim, 
                                         opt.yes_self_trans, opt.dropout, opt.cuda)                
        self.len_logprobs = Length(opt.K, opt.Kmul, 
                                   self.trans_logprobs.get_A_from(), self.trans_logprobs.get_A_to(), 
                                   opt.unif_lenps, opt.L)                
        self.obs_logprobs = Emission(self.lut, opt.mlpinp, opt.emb_size, opt.mlp_sz_mult, 
                                     opt.emb_drop, opt.dropout, self.ar, opt.word_ar, 
                                     opt.hid_size, opt.layers, opt.lse_obj,
                                     opt.one_rnn, opt.L, opt.K, opt.Kmul, opt.sep_attn, 
                                     gentypes, opt.cuda)
        self.init_weights()
        
    def init_weights(self):
        """ initialize embedding layer"""
        initrange = self.initrange
        self.lut.weight.data.uniform_(-initrange, initrange)
        self.lut.weight.data[self.pad_idx].zero_()
        self.lut.weight.data[self.corpus.dictionary.word2idx["<ncf1>"]].zero_()
        self.lut.weight.data[self.corpus.dictionary.word2idx["<ncf2>"]].zero_()
        self.lut.weight.data[self.corpus.dictionary.word2idx["<ncf3>"]].zero_()
        self.encoder.init_weights(self.initrange)
        self.trans_logprobs.init_weights(self.initrange)
        self.obs_logprobs.init_weights(self.initrange)
        
        
    def forward(self, src, amask, uniqfields, seqlen, inps, fmask, combotargs, bsz):
        
        
        srcenc, srcfieldenc, uniqenc = self.encoder(src, amask, uniqfields)
        
        init_logps, trans_logps = self.trans_logprobs(uniqenc, seqlen) # bsz x K, T-1 x bsz x KxK
        
        len_logprobs, _ = self.len_logprobs()
        
        fwd_obs_logps = self.obs_logprobs(inps, srcenc, srcfieldenc, fmask,
                                          combotargs, bsz) # L x T x bsz x K
        
        #print "src", src
        #print "uniqfields", uniqfields
        #print "srcenc: ", srcenc
        #print "srcfieldenc: ", srcfieldenc 
        #print "uniqenc: ", uniqenc
        #print "init_logps: ", init_logps 
        #print "trans_logps: ", trans_logps
        #print "len_logprobs", len_logprobs
        #print "fwd_obs_logps: ",  fwd_obs_logps
        return init_logps, trans_logps, len_logprobs, fwd_obs_logps
        
        
class Encoder(nn.Module):
    def __init__(self, emb_size, lut, max_pool):
        super(Encoder, self).__init__()
        
        self.lut = lut
        self.src_bias = nn.Parameter(torch.Tensor(1, emb_size))
        self.uniq_bias = nn.Parameter(torch.Tensor(1, emb_size))
        self.max_pool = max_pool

    def init_weights(self, initrange):
        self.src_bias.data.uniform_(-initrange, initrange)
        self.uniq_bias.data.uniform_(-initrange, initrange)
        
    def forward(self, src, avgmask, uniqfields):
        """
        args:
          src - bsz x nfields x nfeats
          avgmask - bsz x nfields, with 0s for pad and 1/tru_nfields for rest
          uniqfields - bsz x maxfields
        returns bsz x emb_size, bsz x nfields x emb_size
        """
        
        bsz, nfields, nfeats = src.size()
        emb_size = self.lut.embedding_dim
        # do src stuff that depends on words
        embs = self.lut(src.view(-1, nfeats)) # bsz*nfields x nfeats x emb_size
        if self.max_pool:
            embs = F.relu(embs.sum(1) + self.src_bias.expand(bsz*nfields, emb_size))
            if avgmask is not None:
                masked = (embs.view(bsz, nfields, emb_size)
                          * avgmask.unsqueeze(2).expand(bsz, nfields, emb_size))
            else:
                masked = embs.view(bsz, nfields, emb_size)
            srcenc = F.max_pool1d(masked.transpose(1, 2), nfields).squeeze(2)  # bsz x emb_size
        else:
            embs = F.tanh(embs.sum(1) + self.src_bias.expand(bsz*nfields, emb_size))
            # average it manually, bleh
            if avgmask is not None:
                srcenc = (embs.view(bsz, nfields, emb_size)
                          * avgmask.unsqueeze(2).expand(bsz, nfields, emb_size)).sum(1)
            else:
                srcenc = embs.view(bsz, nfields, emb_size).mean(1) # bsz x emb_size

        srcfieldenc = embs.view(bsz, nfields, emb_size)

        # do stuff that depends only on uniq fields
        uniqenc = self.lut(uniqfields).sum(1) # bsz x nfields x emb_size -> bsz x emb_size

        # add a bias
        uniqenc = uniqenc + self.uniq_bias.expand_as(uniqenc)
        uniqenc = F.relu(uniqenc)
        
        return srcenc, srcfieldenc, uniqenc

    
class Transition(nn.Module):
    def __init__(self, smaller_cond_dim, emb_size, K, Kmul, cond_A_dim, A_dim, yes_self_trans, dropout, cuda):
        super(Transition, self).__init__()
        
        self.smaller_cond_dim = smaller_cond_dim
        self.K = K
        self.Kmul = Kmul
        self.A_dim = A_dim
        self.A_from = nn.Parameter(torch.Tensor(K*Kmul, A_dim))
        self.A_to = nn.Parameter(torch.Tensor(A_dim, K*Kmul))
        self.cond_A_dim = cond_A_dim
        
        ## cond_trans_lin
        if smaller_cond_dim > 0:
            self.cond_trans_lin = nn.Sequential(
                nn.Linear(emb_size, smaller_cond_dim),
                nn.ReLU(),
                nn.Linear(smaller_cond_dim, K*Kmul*cond_A_dim*2))
        else:
            self.cond_trans_lin = nn.Linear(emb_size, K*Kmul*cond_A_dim*2)     
            
        ## selfmask
        self.yes_self_trans = yes_self_trans
        if not self.yes_self_trans:
            selfmask = torch.Tensor(K*Kmul).fill_(-float("inf"))
            self.register_buffer('selfmask', Variable(torch.diag(selfmask), requires_grad=False))
            #self.selfmask = Variable(torch.diag(selfmask), requires_grad=False)
            #if cuda:
                #self.selfmask = self.selfmask.cuda()
        
        ## LogSoftmax, Dropout, Linear    
        self.lsm = nn.LogSoftmax(dim=1)        
        self.init_lin = nn.Linear(emb_size, K*Kmul)        
        self.drop = nn.Dropout(dropout)
    
    def init_weights(self, initrange):
        self.A_from.data.uniform_(-initrange, initrange)
        self.A_to.data.uniform_(-initrange, initrange)
        
    def get_A_from(self):
        
        return self.A_from
    
    def get_A_to(self):
        
        return self.A_to
    
    def forward(self, uniqenc, seqlen):
        """
        args:
          uniqenc - bsz x emb_size
        returns:
          1 x K tensor and seqlen-1 x bsz x K x K tensor of log probabilities,
                           where lps[i] is p(q_{i+1} | q_i)
        """
        bsz = uniqenc.size(0)
        K = self.K*self.Kmul
        A_dim = self.A_dim
        # bsz x K*A_dim*2 -> bsz x K x A_dim or bsz x K x 2*A_dim
        cond_trans_mat = self.cond_trans_lin(uniqenc).view(bsz, K, -1)
        # nufrom, nuto each bsz x K x A_dim
        A_dim = self.cond_A_dim
        nufrom, nuto = cond_trans_mat[:, :, :A_dim], cond_trans_mat[:, :, A_dim:]
        A_from, A_to = self.A_from, self.A_to
        if self.drop.p > 0:
            A_from = self.drop(A_from)
            nufrom = self.drop(nufrom)
        tscores = torch.mm(A_from, A_to)
        if not self.yes_self_trans:
            tscores = tscores + self.selfmask
        trans_lps = tscores.unsqueeze(0).expand(bsz, K, K)
        trans_lps = trans_lps + torch.bmm(nufrom, nuto.transpose(1, 2))
        trans_lps = self.lsm(trans_lps.view(-1, K)).view(bsz, K, K)

        init_logps = self.lsm(self.init_lin(uniqenc)) # bsz x K        
        trans_logps = trans_lps.view(1, bsz, K, K).expand(seqlen-1, bsz, K, K)
        
        return init_logps, trans_logps


class Length(nn.Module):
    def __init__(self, K, Kmul, A_from, A_to, unif_lenps, L):
        super(Length, self).__init__()
        
        self.K = K
        self.Kmul = Kmul
        self.A_from, self.A_to = A_from, A_to
        self.unif_lenps = unif_lenps
        if self.unif_lenps:
            self.len_scores = nn.Parameter(torch.ones(1, L))
            self.len_scores.requires_grad = False
        else:
            self.len_decoder = nn.Linear(2*A_dim, L)
        self.L = L
        self.lsm = nn.LogSoftmax(dim=1)
        
    def forward(self):
        """
        returns:
           [1xK tensor, 2 x K tensor, .., L-1 x K tensor, L x K tensor] of logprobs
        """
        K = self.K*self.Kmul
        state_embs = torch.cat([self.A_from, self.A_to.t()], 1) # K x 2*A_dim
        if self.unif_lenps:
            len_scores = self.len_scores.expand(K, self.L)
        else:
            len_scores = self.len_decoder(state_embs) # K x L
        lplist = [Variable(len_scores.data.new(1, K).zero_())]
        for l in xrange(2, self.L+1):
            lplist.append(self.lsm(len_scores.narrow(1, 0, l)).t())
   
        return lplist, len_scores
    
        
class Emission(nn.Module):
    def __init__ (self, lut, mlpinp, emb_size, mlp_sz_mult, 
                  emb_drop, dropout, ar, word_ar, 
                  hid_size, layers, lse_obj,
                  one_rnn, L, K, Kmul, sep_attn, gentypes, cuda):
        super(Emission, self).__init__()
        
        self.lut = lut
        self.mlpinp = mlpinp
        sz_mult = mlp_sz_mult
        inp_feats = 4
        self.emb_drop = emb_drop
        self.layers, self.hid_size = layers, hid_size
        self.ar = ar
        self.word_ar = word_ar
        self.L = L
        self.K = K
        self.Kmul = Kmul
        self.one_rnn = one_rnn
        
        if mlpinp:
            rnninsz = sz_mult*emb_size
            mlpinp_sz = inp_feats*emb_size
            self.inpmlp = nn.Sequential(nn.Linear(mlpinp_sz, sz_mult*emb_size),
                                        nn.ReLU())
        else:
            rnninsz = inp_feats*emb_size
            
        self.start_emb = nn.Parameter(torch.Tensor(1, 1, rnninsz))       
        self.pad_emb = nn.Parameter(torch.zeros(1, 1, rnninsz))  
                
        self.seg_rnns = nn.ModuleList()
        if one_rnn:
            rnninsz += emb_size
            self.seg_rnns.append(nn.LSTM(rnninsz, hid_size,
                                         layers, dropout=dropout))
            self.state_embs = nn.Parameter(torch.Tensor(K, 1, 1, emb_size))
        else:
            for _ in xrange(K):
                self.seg_rnns.append(nn.LSTM(rnninsz, hid_size,
                                             layers, dropout=dropout))
            self.state_embs = nn.Parameter(torch.Tensor(K, 1, 1, emb_size))         
        
        self.drop = nn.Dropout(dropout)    
        self.ar_rnn = nn.LSTM(emb_size, hid_size, layers, dropout=dropout)              
        self.h0_lin = nn.Linear(emb_size, 2*hid_size)
                
        self.state_att_gates = nn.Parameter(torch.Tensor(K, 1, 1, hid_size))
        self.state_att_biases = nn.Parameter(torch.Tensor(K, 1, 1, hid_size))
        
        out_hid_sz = hid_size + emb_size
        self.state_out_gates = nn.Parameter(torch.Tensor(K, 1, 1, out_hid_sz))
        self.state_out_biases = nn.Parameter(torch.Tensor(K, 1, 1, out_hid_sz))
        
        self.sep_attn = sep_attn
        if self.sep_attn:
            self.state_att2_gates = nn.Parameter(torch.Tensor(K, 1, 1, hid_size))
            self.state_att2_biases = nn.Parameter(torch.Tensor(K, 1, 1, hid_size))
            
        self.lse_obj = lse_obj
        # add one more output word for eop
        self.decoder = nn.Linear(out_hid_sz, gentypes+1)
        self.eop_idx = gentypes
        self.zeros = torch.Tensor(1, 1).fill_(-float("inf")) if lse_obj else torch.zeros(1, 1)
        if cuda:
            self.zeros = self.zeros.cuda()
    
    def init_weights(self, initrange):
        rnns = [rnn for rnn in self.seg_rnns]
        rnns.append(self.ar_rnn)
        for rnn in rnns:
            for thing in rnn.parameters():
                thing.data.uniform_(-initrange, initrange)
                
        self.state_out_gates.data.uniform_(-initrange, initrange)
        self.state_att_gates.data.uniform_(-initrange, initrange)
        self.state_out_biases.data.uniform_(-initrange, initrange)
        self.state_att_biases.data.uniform_(-initrange, initrange)
        self.start_emb.data.uniform_(-initrange, initrange)
    
    def to_seg_embs(self, xemb):
        """
        xemb - bsz x seqlen x emb_size
        returns - L+1 x bsz*seqlen x emb_size,
           where [1 2 3 4]  becomes [<s> <s> <s> <s> <s> <s> <s> <s>]
                 [5 6 7 8]          [ 1   2   3   4   5   6   7   8 ]
                                    [ 2   3   4  <p>  6   7   8  <p>]
                                    [ 3   4  <p> <p>  7   8  <p> <p>]
        """
        bsz, seqlen, emb_size = xemb.size()
        newx = [self.start_emb.expand(bsz, seqlen, emb_size)]
        newx.append(xemb)
        for i in xrange(1, self.L):
            pad = self.pad_emb.expand(bsz, i, emb_size)
            rowi = torch.cat([xemb[:, i:], pad], 1)
            newx.append(rowi)
        # L+1 x bsz x seqlen x emb_size -> L+1 x bsz*seqlen x emb_size
        return torch.stack(newx).view(self.L+1, -1, emb_size)

    def to_seg_hist(self, states):
        """
        states - bsz x seqlen+1 x rnn_size
        returns - L+1 x bsz*seqlen x emb_size,
           where [<b> 1 2 3 4]  becomes [<b>  1   2   3  <b>  5   6   7 ]
                 [<b> 5 6 7 8]          [ 1   2   3   4   5   6   7   8 ]
                                        [ 2   3   4  <p>  6   7   8  <p>]
                                        [ 3   4  <p> <p>  7   8  <p> <p>]
        """
        bsz, seqlenp1, rnn_size = states.size()
        newh = [states[:, :seqlenp1-1, :]] # [bsz x seqlen x rnn_size]
        newh.append(states[:, 1:, :])
        for i in xrange(1, self.L):
            pad = self.pad_emb[:, :, :rnn_size].expand(bsz, i, rnn_size)
            rowi = torch.cat([states[:, i+1:, :], pad], 1)
            newh.append(rowi)
        # L+1 x bsz x seqlen x rnn_size -> L+1 x bsz*seqlen x rnn_size
        return torch.stack(newh).view(self.L+1, -1, rnn_size)
    
    def forward(self, inps, srcenc, srcfieldenc, fieldmask, combotargs, bsz):
        """
        args:
          inps - seqlen x bsz x max_locs x nfeats
          srcenc - bsz x emb_size
          srcfieldenc - bsz x nfields x dim
          fieldmask - bsz x nfields mask with 0s and -infs where it's a dummy field
          combotargs - L x bsz*seqlen x max_locs
        returns:
          a L x seqlen x bsz x K tensor, where l'th row has prob of sequences of length l+1.
          specifically, obs_logprobs[:,t,i,k] gives p(x_t|k), p(x_{t:t+1}|k), ..., p(x_{t:t+l}|k).
          the infc code ignores the entries rows corresponding to x_{t:t+m} where t+m > T
        """
        seqlen, bsz, maxlocs, nfeats = inps.size()
        embs = self.lut(inps.view(seqlen, -1)) # seqlen x bsz*maxlocs*nfeats x emb_size

        if self.mlpinp:
            inpembs = self.inpmlp(embs.view(seqlen, bsz, maxlocs, -1)).mean(2)
        else:
            inpembs = embs.view(seqlen, bsz, maxlocs, -1).mean(2) # seqlen x bsz x nfeats*emb_size

        if self.emb_drop:
            inpembs = self.drop(inpembs)

        if self.ar:
            if self.word_ar: ## select the word column
                ar_embs = embs.view(seqlen, bsz, maxlocs, nfeats, -1)[:, :, 0, 0] # seqlen x bsz x embsz
            else: # ar on fields 
                ar_embs = embs.view(seqlen, bsz, maxlocs, nfeats, -1)[:, :, :, 1].mean(2) # same
            if self.emb_drop:
                ar_embs = self.drop(ar_embs)

            # add on initial <bos> thing; this is a HACK!
            embsz = ar_embs.size(2)
            ar_embs = torch.cat([self.lut.weight[2].view(1, 1, embsz).expand(1, bsz, embsz),
                                    ar_embs], 0) # seqlen+1 x bsz x emb_size
            ar_states, _ = self.ar_rnn(ar_embs) # seqlen+1 x bsz x rnn_size            

        # get L+1 x bsz*seqlen x emb_size segembs
        segembs = self.to_seg_embs(inpembs.transpose(0, 1))
        
        Lp1, bszsl, _ = segembs.size()
        if self.ar:
            segars = self.to_seg_hist(ar_states.transpose(0, 1)) #L+1 x bsz*seqlen x rnn_size
        bsz, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hid_size

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        inits = self.h0_lin(srcenc) # bsz x 2*dim
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:] # (bsz x dim, bsz x dim)
        h0 = F.tanh(h0).unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        c0 = c0.unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()

        # easiest to just loop over K
        state_emb_sz = self.state_embs.size(3)
        seg_lls = []
        for k in xrange(self.K):
            if self.one_rnn:
                condembs = torch.cat(
                    [segembs, self.state_embs[k].expand(Lp1, bszsl, state_emb_sz)], 2)
                states, _ = self.seg_rnns[0](condembs, (h0, c0)) # L+1 x bsz*seqlen x rnn_size
            else:
                states, _ = self.seg_rnns[k](segembs, (h0, c0)) # L+1 x bsz*seqlen x rnn_size

            if self.ar:
                states = states + segars # L+1 x bsz*seqlen x rnn_size

            if self.drop.p > 0:
                states = self.drop(states)
            attnin1 = (states * self.state_att_gates[k].expand_as(states)
                       + self.state_att_biases[k].expand_as(states)).view(
                           Lp1, bsz, seqlen, -1)
            # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x rnn_size
            attnin1 = attnin1.transpose(0, 1).contiguous().view(bsz, Lp1*seqlen, -1)
            attnin1 = F.tanh(attnin1)
            ascores = torch.bmm(attnin1, srcfieldenc.transpose(1, 2)) # bsz x (L+1)slen x nfield
            ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)
            aprobs = F.softmax(ascores, dim=2)
            # bsz x (L+1)seqlen x nfields * bsz x nfields x dim -> bsz x (L+1)seqlen x dim
            ctx = torch.bmm(aprobs, srcfieldenc)
            # concatenate states and ctx to get L+1 x bsz x seqlen x rnn_size + encdim
                    
            cat_ctx = torch.cat([states.view(Lp1, bsz, seqlen, -1),
                                 ctx.view(bsz, Lp1, seqlen, -1).transpose(0, 1)], 3)

            out_hid_sz = rnn_size + encdim
           
            cat_ctx = cat_ctx.view(Lp1, -1, out_hid_sz)
            # now linear to get L+1 x bsz*seqlen x rnn_size
            states_k = F.tanh(cat_ctx * self.state_out_gates[k].expand_as(cat_ctx)
                              + self.state_out_biases[k].expand_as(cat_ctx)).view(
                                  Lp1, -1, out_hid_sz)

            if self.sep_attn:
                attnin2 = (states * self.state_att2_gates[k].expand_as(states)
                           + self.state_att2_biases[k].expand_as(states)).view(
                               Lp1, bsz, seqlen, -1)
                # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x emb_size
                attnin2 = attnin2.transpose(0, 1).contiguous().view(bsz, Lp1*seqlen, -1)
                attnin2 = F.tanh(attnin2)
                ascores = torch.bmm(attnin2, srcfieldenc.transpose(1, 2)) # bsz x (L+1)slen x nfield
                ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)

            normfn = F.log_softmax if self.lse_obj else F.softmax
            wlps_k = normfn(torch.cat([self.decoder(states_k.view(-1, out_hid_sz)), #L+1*bsz*sl x V
                                       ascores.view(bsz, Lp1, seqlen, nfields).transpose(
                                           0, 1).contiguous().view(-1, nfields)], 1), dim=1)
            # concatenate on dummy column for when only a single answer...
            wlps_k = torch.cat([wlps_k, Variable(self.zeros.expand(wlps_k.size(0), 1))], 1)
            # get scores for predicted next-words (but not for last words in each segment as usual)
            #L+1*bsz*sql X 1
            psk = wlps_k.narrow(0, 0, self.L*bszsl).gather(1, combotargs.view(self.L*bszsl, -1))            
            if self.lse_obj:
                lls_k = logsumexp1(psk)
            else:
                lls_k = psk.sum(1).log()

            # sum up log probs of words in each segment
            seglls_k = lls_k.view(self.L, -1).cumsum(0) # L x bsz*seqlen
            # need to add end-of-phrase prob too
            eop_lps = wlps_k.narrow(0, bszsl, self.L*bszsl)[:, self.eop_idx] # L*bsz*seqlen
            if self.lse_obj:
                seglls_k = seglls_k + eop_lps.contiguous().view(self.L, -1)
            else:
                seglls_k = seglls_k + eop_lps.log().view(self.L, -1)
            seg_lls.append(seglls_k)

        #  K x L x bsz x seqlen -> seqlen x L x bsz x K -> L x seqlen x bsz x K
        obslps = torch.stack(seg_lls).view(self.K, self.L, bsz, -1).transpose(
            0, 3).transpose(0, 1)
        if self.Kmul > 1:
            obslps = obslps.repeat(1, 1, 1, self.Kmul)
        
        return obslps        
    
    


