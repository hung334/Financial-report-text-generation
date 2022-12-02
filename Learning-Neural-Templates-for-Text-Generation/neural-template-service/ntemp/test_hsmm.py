import sys
import os
import math
import random
from collections import Counter
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from ntemp import labeled_data
from ntemp.utils.utils import backtrace3, backtrace
from ntemp.chsmm import HSMM
from ntemp.utils.preprocess import get_uniq_fields


class Generator(nn.Module):    
    def __init__(self, args):
        super(Generator, self).__init__()
        ## the final target is changing the args to only contain the process of generation.
        self.args = args       
        """ Start load saved args and model"""
        saved_args, saved_state = None, None
        if len(args.load) > 0:
            if args.cuda:
                saved_stuff = torch.load(args.load)
            else:
                saved_stuff = torch.load(args.load, map_location='cpu')
            saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
	    
            corpus = labeled_data.SentenceCorpus(saved_args.data, saved_args.bsz, thresh=saved_args.thresh, add_bos=False, 
                                                 add_eos=False, test=saved_args.test)

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

        if args.cuda:
            net = net.cuda()
        """ End load saved args and model"""
        self.args = args
        self.saved_args, self.saved_state = saved_args, saved_state
        self.corpus = corpus
        self.net = net


    def get_next_word_dist(self, hid, k, srcfieldenc):
        """
        hid - 1 x bsz x rnn_size
        srcfieldenc - 1 x nfields x dim
        returns a bsz x nthings dist; not a log dist
        """
        bsz = hid.size(1)
        _, nfields, rnn_size = srcfieldenc.size()
        srcfldenc = srcfieldenc.expand(bsz, nfields, rnn_size)
        attnin1 = (hid * self.net.obs_logprobs.state_att_gates[k].expand_as(hid)
                   + self.net.obs_logprobs.state_att_biases[k].expand_as(hid)) # 1 x bsz x rnn_size
        attnin1 = F.tanh(attnin1)
        ascores = torch.bmm(attnin1.transpose(0, 1), srcfldenc.transpose(1, 2)) # bsz x 1 x nfields
        aprobs = F.softmax(ascores, dim=2)
        ctx = torch.bmm(aprobs, srcfldenc) # bsz x 1 x rnn_size
        cat_ctx = torch.cat([hid, ctx.transpose(0, 1)], 2) # 1 x bsz x rnn_size


        state_k = F.tanh(cat_ctx * self.net.obs_logprobs.state_out_gates[k].expand_as(cat_ctx)
                         + self.net.obs_logprobs.state_out_biases[k].expand_as(cat_ctx)) # 1 x bsz x rnn_size

        if self.net.obs_logprobs.sep_attn:
            attnin2 = (hid * self.net.obs_logprobs.state_att2_gates[k].expand_as(hid)
                       + self.net.obs_logprobs.state_att2_biases[k].expand_as(hid))
            attnin2 = F.tanh(attnin2)
            ascores = torch.bmm(attnin2.transpose(0, 1), srcfldenc.transpose(1, 2)) # bsz x 1 x nfld

        wlps_k = F.softmax(torch.cat([self.net.obs_logprobs.decoder(state_k.squeeze(0)),
                                      ascores.squeeze(1)], 1), dim=1)
        return wlps_k.data

    def collapse_word_probs(self, row2tblent, wrd_dist):
        """
        wrd_dist is a K x nwords matrix and it gets modified.
        this collapsing only makes sense if src_tbl is the same for every row.
        """
        nout_wrds = self.net.obs_logprobs.decoder.out_features
        i2w, w2i = self.corpus.dictionary.idx2word, self.corpus.dictionary.word2idx
        # collapse probabilities
        first_seen = {}
        for i, (field, idx, wrd) in row2tblent.iteritems():
            if field is not None:
                if wrd not in first_seen:
                    first_seen[wrd] = i
                    # add gen prob if any
                    if wrd in self.corpus.genset:
                        widx = w2i[wrd]
                        wrd_dist[:, nout_wrds + i].add_(wrd_dist[:, widx])
                        wrd_dist[:, widx].zero_()
                else: # seen it before, so add its prob
                    wrd_dist[:, nout_wrds + first_seen[wrd]].add_(wrd_dist[:, nout_wrds + i])
                    wrd_dist[:, nout_wrds + i].zero_()
            else: # should really have zeroed out before, but this is easier
                wrd_dist[:, nout_wrds + i].zero_()
    
    def temp_bs(self, ss, start_inp, exh0, exc0, srcfieldenc,
                     len_lps, row2tblent, row2feats, K, final_state=False):
        """
        ss - discrete state index
        exh0 - layers x 1 x rnn_size
        exc0 - layers x 1 x rnn_size
        start_inp - 1 x 1 x emb_size
        len_lps - K x L, log normalized
        """
        rul_ss = ss % self.saved_args.K
        i2w = self.corpus.dictionary.idx2word
        w2i = self.corpus.dictionary.word2idx
        genset = self.corpus.genset
        unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]
        state_emb_sz = self.net.obs_logprobs.state_embs.size(3) if self.saved_args.one_rnn else 0
        if self.saved_args.one_rnn:
            cond_start_inp = torch.cat([start_inp, self.net.obs_logprobs.state_embs[rul_ss]], 2) # 1 x 1 x cat_size
            hid, (hc, cc) = self.net.obs_logprobs.seg_rnns[0](cond_start_inp, (exh0, exc0))
        else:
            hid, (hc, cc) = self.net.obs_logprobs.seg_rnns[rul_ss](start_inp, (exh0, exc0))
        curr_hyps = [(None, None)]
        best_wscore, best_lscore = None, None # so we can truly average over words etc later
        best_hyp, best_hyp_score = None, -float("inf")
        curr_scores = torch.zeros(K, 1)
        # N.B. we assume we have a single feature row for each timestep rather than avg
        # over them as at training time. probably better, but could conceivably average like
        # at training time.
        inps = Variable(torch.LongTensor(K, 4), volatile=True)
        for ell in xrange(self.saved_args.L):
            wrd_dist = self.get_next_word_dist(hid, rul_ss, srcfieldenc).cpu() # K x nwords 
            # disallow unks
            wrd_dist[:, unk_idx].zero_()
            if not final_state:
                wrd_dist[:, eos_idx].zero_()
            self.collapse_word_probs(row2tblent, wrd_dist)
            wrd_dist.log_()
            if ell > 0: # add previous scores
                wrd_dist.add_(curr_scores.expand_as(wrd_dist))
            maxprobs, top2k = torch.topk(wrd_dist.view(-1), 2*K)
            cols = wrd_dist.size(1)
            # we'll break as soon as <eos> is at the top of the beam.
            # this ignores <eop> but whatever
            if top2k[0] == eos_idx:
                final_hyp = backtrace(curr_hyps[0])
                final_hyp.append(eos_idx)
                return final_hyp, maxprobs[0], len_lps[ss][ell]

            new_hyps, anc_hs, anc_cs = [], [], []
            #inps.data.fill_(pad_idx)
            inps.data[:, 1].fill_(w2i["<ncf1>"])
            inps.data[:, 2].fill_(w2i["<ncf2>"])
            inps.data[:, 3].fill_(w2i["<ncf3>"])
            for k in xrange(2*K):
                anc, wrd = top2k[k] / cols, top2k[k] % cols
                # check if any of the maxes are eop
                if wrd == self.saved_args.eop_idx and ell > 0:
                    # add len score (and avg over num words incl eop i guess)
                    wlenscore = maxprobs[k]/(ell+1) + len_lps[ss][ell-1]
                    if wlenscore > best_hyp_score:
                        best_hyp_score = wlenscore
                        best_hyp = backtrace(curr_hyps[anc])
                        best_wscore, best_lscore = maxprobs[k], len_lps[ss][ell-1]
                else:
                    curr_scores[len(new_hyps)][0] = maxprobs[k]
                    if wrd >= self.net.obs_logprobs.decoder.out_features: # a copy
                        tblidx = wrd - self.net.obs_logprobs.decoder.out_features
                        inps.data[len(new_hyps)].copy_(row2feats[tblidx])
                    else:
                        inps.data[len(new_hyps)][0] = wrd if i2w[wrd] in genset else unk_idx
                    new_hyps.append((wrd, curr_hyps[anc]))
                    anc_hs.append(hc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                    anc_cs.append(cc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                if len(new_hyps) == K:
                    break
            assert len(new_hyps) == K
            curr_hyps = new_hyps
            if self.net.lut.weight.data.is_cuda:
                inps = inps.cuda()
            embs = self.net.lut(inps).view(1, K, -1) # 1 x K x nfeats*emb_size
            if self.saved_args.mlpinp:
                embs = self.net.obs_logprobs.inpmlp(embs) # 1 x K x rnninsz
            if self.saved_args.one_rnn:
                cond_embs = torch.cat([embs, self.state_embs[rul_ss].expand(1, K, state_emb_sz)], 2)
                hid, (hc, cc) = self.net.obs_logprobs.seg_rnns[0](cond_embs, (torch.cat(anc_hs, 1), torch.cat(anc_cs, 1)))
            else:
                hid, (hc, cc) = self.net.obs_logprobs.seg_rnns[rul_ss](embs, (torch.cat(anc_hs, 1), torch.cat(anc_cs, 1)))
        # hypotheses of length L still need their end probs added
        # N.B. if the <eos> falls off the beam we could end up with situations
        # where we take an L-length phrase w/ a lower score than 1-word followed by eos.
        wrd_dist = self.get_next_word_dist(hid, rul_ss, srcfieldenc).cpu() # K x nwords
        wrd_dist.log_()
        wrd_dist.add_(curr_scores.expand_as(wrd_dist))
        for k in xrange(K):
            wlenscore = wrd_dist[k][self.saved_args.eop_idx]/(self.L+1) + len_lps[ss][self.saved_args.L-1]
            if wlenscore > best_hyp_score:
                best_hyp_score = wlenscore
                best_hyp = backtrace(curr_hyps[k])
                best_wscore, best_lscore = wrd_dist[k][self.saved_args.eop_idx], len_lps[ss][self.saved_args.L-1]

        return best_hyp, best_wscore, best_lscore
    
    def gen_one(self, templt, h0, c0, srcfieldenc, len_lps, 
                row2tblent, row2feats):
        """
        src - 1 x nfields x nfeatures
        h0 - rnn_size vector
        c0 - rnn_size vector
        srcfieldenc - 1 x nfields x dim
        len_lps - K x L, log normalized
        returns a list of phrases
        """
        phrases = []
        tote_wscore, tote_lscore, tokes, segs = 0.0, 0.0, 0.0, 0.0
        #start_inp = self.lut.weight[start_idx].view(1, 1, -1)
        start_inp = self.net.obs_logprobs.start_emb
        exh0 = h0.view(1, 1, self.saved_args.hid_size).expand(self.saved_args.layers, 1, self.saved_args.hid_size)
        exc0 = c0.view(1, 1, self.saved_args.hid_size).expand(self.saved_args.layers, 1, self.saved_args.hid_size)
        nout_wrds = self.net.obs_logprobs.decoder.out_features
        i2w, w2i = self.corpus.dictionary.idx2word, self.corpus.dictionary.word2idx
        for stidx, k in enumerate(templt):
            phrs_idxs, wscore, lscore = self.temp_bs(k, start_inp, exh0, exc0,
                                                     srcfieldenc, len_lps, row2tblent, row2feats,
                                                     self.args.beamsz, final_state=(stidx == len(templt)-1))
            phrs = []
            for ii in xrange(len(phrs_idxs)):
                if phrs_idxs[ii] < nout_wrds:
                    phrs.append(i2w[phrs_idxs[ii]])
                else:
                    tblidx = phrs_idxs[ii] - nout_wrds
                    _, _, wordstr = row2tblent[tblidx]
                    if args.verbose:
                        phrs.append(wordstr + " (c)")
                    else:
                        phrs.append(wordstr)
            if phrs[-1] == "<eos>":
                break
            phrases.append(phrs)
            tote_wscore += wscore
            tote_lscore += lscore
            tokes += len(phrs_idxs) + 1 # add 1 for <eop> token
            segs += 1

        return phrases, tote_wscore, tote_lscore, tokes, segs

    
    def temp_ar_bs(self, templt, row2tblent, row2feats, h0, c0, srcfieldenc, len_lps,  K):
        assert self.net.len_logprobs.unif_lenps # ignoring lenps
        exh0 = h0.view(1, 1, self.saved_args.hid_size).expand(self.saved_args.layers, 1, self.saved_args.hid_size)
        exc0 = c0.view(1, 1, self.saved_args.hid_size).expand(self.saved_args.layers, 1, self.saved_args.hid_size)
        start_inp = self.net.obs_logprobs.start_emb
        state_emb_sz = self.net.obs_logprobs.state_embs.size(3)
        i2w, w2i = self.corpus.dictionary.idx2word, self.corpus.dictionary.word2idx
        genset = self.corpus.genset
        unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]

        curr_hyps = [(None, None, None)]
        nfeats = 4
        inps = Variable(torch.LongTensor(K, nfeats), volatile=True)
        curr_scores, curr_lens, nulens = torch.zeros(K, 1), torch.zeros(K, 1), torch.zeros(K, 1)
        if self.net.lut.weight.data.is_cuda:
            inps = inps.cuda()
            curr_scores, curr_lens, nulens = curr_scores.cuda(), curr_lens.cuda(), nulens.cuda()

        # start ar rnn; hackily use bos_idx
        rnnsz = self.net.obs_logprobs.ar_rnn.hidden_size
        thid, (thc, tcc) = self.net.obs_logprobs.ar_rnn(self.net.lut.weight[2].view(1, 1, -1)) # 1 x 1 x rnn_size

        for stidx, ss in enumerate(templt):
            final_state = (stidx == len(templt) - 1)
            minq = [] # so we can compare stuff of different lengths
            rul_ss = ss % self.saved_args.K

            if self.saved_args.one_rnn:
                cond_start_inp = torch.cat([start_inp, self.net.obs_logprobs.state_embs[rul_ss]], 2) # 1x1x cat_size
                hid, (hc, cc) = self.net.obs_logprobs.seg_rnns[0](cond_start_inp, (exh0, exc0)) # 1 x 1 x rnn_size
            else:
                hid, (hc, cc) = self.net.obs_logprobs.seg_rnns[rul_ss](start_inp, (exh0, exc0)) # 1 x 1 x rnn_size
            hid = hid.expand_as(thid)
            hc = hc.expand_as(thc)
            cc = cc.expand_as(tcc)

            for ell in xrange(self.saved_args.L+1):
                new_hyps, anc_hs, anc_cs, anc_ths, anc_tcs = [], [], [], [], []
                inps.data[:, 1].fill_(w2i["<ncf1>"])
                inps.data[:, 2].fill_(w2i["<ncf2>"])
                inps.data[:, 3].fill_(w2i["<ncf3>"])

                wrd_dist = self.get_next_word_dist(hid + thid, rul_ss, srcfieldenc) # K x nwords
                currK = wrd_dist.size(0)
                # disallow unks and eos's
                wrd_dist[:, unk_idx].zero_()
                if not final_state:
                    wrd_dist[:, eos_idx].zero_()
                self.collapse_word_probs(row2tblent, wrd_dist)
                wrd_dist.log_()
                curr_scores[:currK].mul_(curr_lens[:currK])
                wrd_dist.add_(curr_scores[:currK].expand_as(wrd_dist))
                wrd_dist.div_((curr_lens[:currK]+1).expand_as(wrd_dist))
                maxprobs, top2k = torch.topk(wrd_dist.view(-1), 2*K)
                cols = wrd_dist.size(1)
                # used to check for eos here, but maybe we shouldn't

                for k in xrange(2*K):
                    anc, wrd = top2k[k] / cols, top2k[k] % cols
                    # check if any of the maxes are eop
                    if wrd == self.net.obs_logprobs.eop_idx and ell > 0 and (not final_state or curr_hyps[anc][0] == eos_idx):
                        ## add len score (and avg over num words *incl eop*)
                        ## actually ignoring len score for now
                        #wlenscore = maxprobs[k]/(ell+1) # + len_lps[ss][ell-1]
                        #assert not final_state or curr_hyps[anc][0] == eos_idx # seems like should hold...
                        heapitem = (maxprobs[k], curr_lens[anc][0]+1, curr_hyps[anc],
                                    thc.narrow(1, anc, 1), tcc.narrow(1, anc, 1))
                        if len(minq) < K:
                            heapq.heappush(minq, heapitem)
                        else:
                            heapq.heappushpop(minq, heapitem)
                    elif ell < self.saved_args.L: # only allow non-eop if < L so far
                        curr_scores[len(new_hyps)][0] = maxprobs[k]
                        nulens[len(new_hyps)][0] = curr_lens[anc][0]+1
                        if wrd >= self.net.obs_logprobs.decoder.out_features: # a copy
                            tblidx = wrd - self.net.obs_logprobs.decoder.out_features
                            inps.data[len(new_hyps)].copy_(row2feats[tblidx])
                        else:
                            inps.data[len(new_hyps)][0] = wrd if i2w[wrd] in genset else unk_idx

                        new_hyps.append((wrd, ss, curr_hyps[anc]))
                        anc_hs.append(hc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                        anc_cs.append(cc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                        anc_ths.append(thc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                        anc_tcs.append(tcc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                    if len(new_hyps) == K:
                        break

                if ell >= self.saved_args.L: # don't want to put in eops
                    break

                assert len(new_hyps) == K
                curr_hyps = new_hyps
                curr_lens.copy_(nulens)
                embs = self.net.lut(inps).view(1, K, -1) # 1 x K x nfeats*emb_size
                if self.saved_args.word_ar:
                    ar_embs = embs.view(1, K, nfeats, -1)[:, :, 0] # 1 x K x emb_size
                else: # ar on fields
                    ar_embs = embs.view(1, K, nfeats, -1)[:, :, 1] # 1 x K x emb_size
                if self.saved_args.mlpinp:
                    embs = self.net.obs_logprobs.inpmlp(embs) # 1 x K x rnninsz
                if self.saved_args.one_rnn:
                    cond_embs = torch.cat([embs, self.net.obs_logprobs.state_embs[rul_ss].expand(
                        1, K, state_emb_sz)], 2)
                    hid, (hc, cc) = self.net.obs_logprobs.seg_rnns[0](cond_embs, (torch.cat(anc_hs, 1),
                                                                                  torch.cat(anc_cs, 1)))
                else:
                    hid, (hc, cc) = self.net.obs_logprobs.seg_rnns[rul_ss](embs, (torch.cat(anc_hs, 1),
                                                                                  torch.cat(anc_cs, 1)))
                thid, (thc, tcc) = self.net.obs_logprobs.ar_rnn(ar_embs, (torch.cat(anc_ths, 1),
                                                                          torch.cat(anc_tcs, 1)))

            # retrieve topk for this segment (in reverse order)
            seghyps = [heapq.heappop(minq) for _ in xrange(len(minq))]
            if len(seghyps) == 0:
                return -float("inf"), None

            if len(seghyps) < K and not final_state:
                # haaaaaaaaaaaaaaack
                ugh = []
                for ick in xrange(K-len(seghyps)):
                    scoreick, lenick, hypick, thcick, tccick = seghyps[0]
                    ugh.append((scoreick - 9999999.0 + ick, lenick, hypick, thcick, tccick))
                    # break ties for the comparison
                ugh.extend(seghyps)
                seghyps = ugh

            #assert final_state or len(seghyps) == K

            if final_state:
                if len(seghyps) > 0:
                    scoreb, lenb, hypb, thcb, tccb = seghyps[-1]
                    return scoreb, backtrace3(hypb)
                else:
                    return -float("inf"), None
            else:
                thidlst, thclst, tcclst = [], [], []
                for i in xrange(K):
                    scorei, leni, hypi, thci, tcci = seghyps[K-i-1]
                    curr_scores[i][0], curr_lens[i][0], curr_hyps[i] = scorei, leni, hypi
                    thidlst.append(thci[-1:, :, :]) # each is 1 x 1 x rnn_size
                    thclst.append(thci) # each is layers x 1 x rnn_size
                    tcclst.append(tcci) # each is layers x 1 x rnn_size

                # we already have the state for the next word b/c we put it thru to also predict eop
                thid, (thc, tcc) = torch.cat(thidlst, 1), (torch.cat(thclst, 1), torch.cat(tcclst, 1))


    def gen_one_ar(self, templt, h0, c0, srcfieldenc, len_lps, row2tblent, row2feats):
        """
        src - 1 x nfields x nfeatures
        h0 - rnn_size vector
        c0 - rnn_size vector
        srcfieldenc - 1 x nfields x dim
        len_lps - K x L, log normalized
        returns a list of phrases
        """
        nout_wrds = self.net.obs_logprobs.decoder.out_features
        i2w, w2i = self.corpus.dictionary.idx2word, self.corpus.dictionary.word2idx
        phrases, phrs = [], []
        tokes = 0.0
        wscore, hyp = self.temp_ar_bs(templt, row2tblent, row2feats, h0, c0, srcfieldenc,
                                      len_lps, self.args.beamsz)
        if hyp is None:
            return None, -float("inf"), 0
        curr_labe = hyp[0][1]
        tokes = 0
        for widx, labe in hyp:
            if labe != curr_labe:
                phrases.append(phrs)
                tokes += len(phrs)
                phrs = []
                curr_labe = labe
            if widx < nout_wrds:
                phrs.append(i2w[widx])
            else:
                tblidx = widx - nout_wrds
                _, _, wordstr = row2tblent[tblidx]
                if args.verbose:
                    phrs.append(wordstr + " (c)")
                else:
                    phrs.append(wordstr)
        if len(phrs) > 0:
            phrases.append(phrs)
            tokes += len(phrs)

        return phrases, wscore, tokes
    

    def translate(self, src_tbl, top_temps):
        '''
        Inputs:
            src_tbl: dict, returns (key, num) -> word
            top_temps: list
        Returns:
            the description which has already replace the annotation with the value in the table.    
        '''
        coeffs = [float(flt.strip()) for flt in self.args.gen_wts.split(',')]
        self.net.ar = self.saved_args.ar_after_decay
        #print "btw2", net.ar
        i2w, w2i = self.corpus.dictionary.idx2word, self.corpus.dictionary.word2idx
        best_score, best_phrases, best_templt = -float("inf"), None, None
        best_len = 0
        best_tscore, best_gscore = None, None

        # get srcrow 2 key, idx
        #src_b = src.narrow(0, b, 1) # 1 x nfields x nfeats
        src_b = self.corpus.featurize_tbl(src_tbl).unsqueeze(0) # 1 x nfields x nfeats
        uniq_b = get_uniq_fields(src_b, self.saved_args.pad_idx) # 1 x max_fields
        
        if self.args.cuda:
            src_b = src_b.cuda()
            uniq_b = uniq_b.cuda()

        srcenc, srcfieldenc, uniqenc = self.net.encoder(Variable(src_b, volatile=True), None,
                                                    Variable(uniq_b, volatile=True))
        init_logps, trans_logps = self.net.trans_logprobs(uniqenc, 2)
        _, len_scores = self.net.len_logprobs()
        len_lps = self.net.len_logprobs.lsm(len_scores).data
        init_logps, trans_logps = init_logps.data.cpu(), trans_logps.data[0].cpu()
        inits = self.net.obs_logprobs.h0_lin(srcenc)
        h0, c0 = F.tanh(inits[:, :inits.size(1)/2]), inits[:, inits.size(1)/2:]

        nfields = src_b.size(1)
        row2tblent = {}
        for ff in xrange(nfields):
            field, idx = i2w[src_b[0][ff][0]], i2w[src_b[0][ff][1]]
            if (field, idx) in src_tbl:
                row2tblent[ff] = (field, idx, src_tbl[field, idx])
            else:
                row2tblent[ff] = (None, None, None)

        # get row to input feats
        row2feats = {}
        # precompute wrd stuff
        fld_cntr = Counter([key for key, _ in src_tbl])
        for row, (k, idx, wrd) in row2tblent.iteritems():
            if k in w2i:
                widx = w2i[wrd] if wrd in w2i else w2i["<unk>"]
                keyidx = w2i[k] if k in w2i else w2i["<unk>"]
                idxidx = w2i[idx]
                cheatfeat = w2i["<stop>"] if fld_cntr[k] == idx else w2i["<go>"]
                #row2feats[row] = torch.LongTensor([keyidx, idxidx, cheatfeat])
                row2feats[row] = torch.LongTensor([widx, keyidx, idxidx, cheatfeat])

        constr_sat = False
        # search over all templates
        for temp_idx, templt in enumerate(top_temps):
            # print "templt is", templt
            # get templt transition prob
            tscores = [init_logps[0][templt[0]]]
            [tscores.append(trans_logps[0][templt[tt-1]][templt[tt]])
                for tt in xrange(1, len(templt))]

            if self.net.ar:
                phrases, wscore, tokes = self.gen_one_ar(templt, h0[0], c0[0], srcfieldenc, len_lps, row2tblent, row2feats)
                rul_tokes = tokes
            else:
                phrases, wscore, lscore, tokes, segs = self.gen_one(templt, h0[0], c0[0], srcfieldenc, len_lps, row2tblent, row2feats)
                rul_tokes = tokes - segs # subtract imaginary toke for each <eop>
                wscore /= tokes
            segs = len(templt)
            if (rul_tokes < self.args.min_gen_tokes or segs < self.args.min_gen_states) and constr_sat:
                continue
            if rul_tokes >= self.args.min_gen_tokes and segs >= self.args.min_gen_states:
                constr_sat = True # satisfied our constraint
            tscore = sum(tscores[:int(segs)])/segs
            if not self.net.len_logprobs.unif_lenps:
                tscore += lscore/segs

            gscore = wscore
            ascore = coeffs[0]*tscore + coeffs[1]*gscore
            if (constr_sat and ascore > best_score) or (not constr_sat and rul_tokes > best_len) or (not constr_sat and rul_tokes == best_len and ascore > best_score):
            # take if improves score or not long enough yet and this is longer...
            #if ascore > best_score: #or (not constr_sat and rul_tokes > best_len):
                best_score, best_tscore, best_gscore = ascore, tscore, gscore
                best_phrases, best_templt = phrases, templt
                best_templt_id = temp_idx
                best_len = rul_tokes
            #str_phrases = [" ".join(phrs) for phrs in phrases]
            #tmpltd = ["%s|%d" % (phrs, templt[k]) for k, phrs in enumerate(str_phrases)]
            #statstr = "a=%.2f t=%.2f g=%.2f" % (ascore, tscore, gscore)
            #print "%s|||%s" % (" ".join(str_phrases), " ".join(tmpltd)), statstr
            ###
            ###
            #assert False
        #assert False
        if (best_phrases!=None):
            try:
                str_phrases = [" ".join(phrs) for phrs in best_phrases]
            except TypeError:
                # sometimes it puts an actual number in
                str_phrases = [" ".join([str(n) if type(n) is int else n for n in phrs]) for phrs in best_phrases]
            # except:
            #     str_phrases=['error']
                
            tmpltd = ["%s|%d" % (phrs, best_templt[kk]) for kk, phrs in enumerate(str_phrases)]
    
    
            #print "%s|||%s" % (" ".join(str_phrases), " ".join(tmpltd))
            #print "%s" % (" ".join(str_phrases))
            return " ".join(str_phrases),tmpltd
        else:
            return "error",['\xe5\xb9\xb4 \xe4\xb8\x8a\xe5\x8d\x8a\xe5\xb9\xb4|37']# chen hung myself
    
    def forward(self, inputs, templates):
        
        final_result,best_phrases = self.translate(inputs, templates)

        return final_result,best_phrases

