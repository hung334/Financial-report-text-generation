import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def make_combo_targs(locs, x, L, nfields, ngen_types):
    """
    combines word and copy targets into a single tensor.
    locs - seqlen x bsz x max_locs
    x - seqlen x bsz
    assumes we have word indices, then fields, then a dummy
    returns L x bsz*seqlen x max_locs tensor corresponding to xsegs[1:]
    """
    seqlen, bsz, max_locs = locs.size()
    # first replace -1s in first loc with target words
    addloc = locs + (ngen_types+1) # seqlen x bsz x max_locs
    firstloc = addloc[:, :, 0] # seqlen x bsz
    targmask = (firstloc == ngen_types) # -1 will now have value ngentypes
    firstloc[targmask] = x[targmask]
    # now replace remaining -1s w/ zero location
    addloc[addloc == ngen_types] = ngen_types+1+nfields # last index
    # finally put in same format as x_segs
    newlocs = torch.LongTensor(L, seqlen, bsz, max_locs).fill_(ngen_types+1+nfields)
    for i in xrange(L):
        newlocs[i][:seqlen-i].copy_(addloc[i:])
    return newlocs.transpose(1, 2).contiguous().view(L, bsz*seqlen, max_locs)


def get_uniq_fields(src, pad_idx, keycol=0):
    """
    src - bsz x nfields x nfeats
    """
    bsz = src.size(0)
    # get unique keys for each example
    keys = [torch.LongTensor(list(set(src[b, :, keycol]))) for b in xrange(bsz)]
    maxkeys = max(keyset.size(0) for keyset in keys)
    fields = torch.LongTensor(bsz, maxkeys).fill_(pad_idx)
    for b, keyset in enumerate(keys):
        fields[b][:len(keyset)].copy_(keyset)
    return fields


def make_masks(src, pad_idx, max_pool=False):
    """
    src - bsz x nfields x nfeats
    """
    neginf = -1e38
    bsz, nfields, nfeats = src.size()
    fieldmask = (src.eq(pad_idx).sum(2) == nfeats) # binary bsz x nfields tensor
    avgmask = (1 - fieldmask).float() # 1s where not padding
    if not max_pool:
        avgmask.div_(avgmask.sum(1, True).expand(bsz, nfields))
    fieldmask = fieldmask.float() * neginf # 0 where not all pad and -1e38 elsewhere
    return fieldmask, avgmask