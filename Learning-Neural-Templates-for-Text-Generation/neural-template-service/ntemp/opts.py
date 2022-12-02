from configargparse import ArgumentParser

parser = ArgumentParser(description='train.py')

def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, 
               default= 'generation.cfg',
               help='config file path, and default is generation.cfg')
    parser.add('-logs', '--logs', type=str, required=False, help='logging file path') 
    parser.add('-debug', '--debug', action='store_true', help='help for debug')    

def model_opts(parser):
    group = parser.add_argument_group('Model-Parameters')
    group.add('-emb_size', '--emb_size', type=int, default=100, help='size of word embeddings')
    group.add('-hid_size', '--hid_size', type=int, default=100, help='size of rnn hidden state')
    group.add('-layers', '--layers', type=int, default=1, help='num rnn layers')
    group.add('-A_dim', '--A_dim', type=int, default=64,
                        help='dim of factors if factoring transition matrix')
    group.add('-cond_A_dim', '--cond_A_dim', type=int, default=32,
                        help='dim of factors if factoring transition matrix')
    group.add('-smaller_cond_dim', '--smaller_cond_dim', type=int, default=64,
                        help='dim of thing we feed into linear to get transitions')
    group.add('-yes_self_trans', '--yes_self_trans', action='store_true', help='')
    group.add('-mlpinp', '--mlpinp', action='store_true', help='')
    group.add('-mlp_sz_mult', '--mlp_sz_mult', type=int, default=2, help='mlp hidsz is this x emb_size')
    group.add('-max_pool', '--max_pool', action='store_true', help='for word-fields')

    group.add('-constr_tr_epochs', '--constr_tr_epochs', type=int, default=100, help='')
    group.add('-no_ar_epochs', '--no_ar_epochs', type=int, default=100, help='')

    group.add('-word_ar', '--word_ar', action='store_true', help='')
    group.add('-ar_after_decay', '--ar_after_decay', action='store_true', help='')
    group.add('-no_ar_for_vit', '--no_ar_for_vit', action='store_true', help='')
    group.add('-fine_tune', '--fine_tune', action='store_true', help='only train ar rnn')

    group.add('-dropout', '--dropout', type=float, default=0.3, help='dropout')
    group.add('-emb_drop', '--emb_drop', action='store_true', help='dropout on embeddings')
    group.add('-lse_obj', '--lse_obj', action='store_true', help='')
    group.add('-sep_attn', '--sep_attn', action='store_true', help='')
    group.add('-max_seqlen', '--max_seqlen', type=int, default=70, help='')

    group.add('-K', '--K', type=int, default=10, help='number of states')
    group.add('-Kmul', '--Kmul', type=int, default=1, help='number of states multiplier')
    group.add('-L', '--L', type=int, default=10, help='max segment length')
    group.add('-unif_lenps', '--unif_lenps', action='store_true', help='')
    group.add('-one_rnn', '--one_rnn', action='store_true', help='')
    group.add('-initrange', '--initrange', type=float, default=0.1, help='uniform init interval')

def train_opts(parser):
    group = parser.add_argument_group('General')
    group.add('-max_decay_count', '--max_decay_count', type=int, default=3, help='myself')
    group.add('-data', '--data', type=str, default='', help='path to data dir')
    group.add('-epochs', '--epochs', type=int, default=40, help='upper epoch limit')
    group.add('-bsz', '--bsz', type=int, default=16, help='batch size')
    group.add('-seed', '--seed', type=int, default=1111, help='random seed')
    group.add('-cuda', '--cuda', action='store_true', help='use CUDA')
    group.add('-log_interval', '--log_interval', type=int, default=200,
                        help='minibatches to wait before logging training status')
    group.add('-save', '--save', type=str, default='', help='path to save the final model')
    group.add('-load', '--load', type=str, default='', help='path to saved model')
    group.add('-test', '--test', action='store_true', help='use test data')
    group.add('-thresh', '--thresh', type=int, default=9, help='prune if occurs <= thresh')
    group.add('-max_mbs_per_epoch', '--max_mbs_per_epoch', type=int, default=35000, help='max minibatches per epoch')

    group.add('-lr', '--lr', type=float, default=1.0, help='initial learning rate')
    group.add('-lr_decay', '--lr_decay', type=float, default=0.5, help='learning rate decay')
    group.add('-optim', '--optim', type=str, default="sgd", help='optimization algorithm')
    group.add('-onmt_decay', '--onmt_decay', action='store_true', help='')
    group.add('-clip', '--clip', type=float, default=5, help='gradient clipping')
    group.add('-interactive', '--interactive', action='store_true', help='')
    #group.add('-label_train', '--label_train', action='store_true', help='')
    group.add('-gen_from_fi', '--gen_from_fi', type=str, default='', help='')
    group.add('-verbose', '--verbose', action='store_true', help='')
    group.add('-prev_loss', '--prev_loss', type=float, default=None, help='')
    group.add('-best_loss', '--best_loss', type=float, default=None, help='')

    group.add('-ntemplates', '--ntemplates', type=int, default=200, help='num templates for gen')
    group.add('-beamsz', '--beamsz', type=int, default=1, help='')
    group.add('-gen_wts', '--gen_wts', type=str, default='1,1', help='')
    group.add('-min_gen_tokes', '--min_gen_tokes', type=int, default=0, help='')
    group.add('-min_gen_states', '--min_gen_states', type=int, default=0, help='')
    group.add('-gen_on_valid', '--gen_on_valid', action='store_true', help='')
    group.add('-align', '--align', action='store_true', help='')
    group.add('-wid_workers', '--wid_workers', type=str, default='', help='')

    
def segs_opts(parser):
    group = parser.add_argument_group('segment-Parameters')
    group.add('-seg_file', '--seg_file', type=str, default='', help='segment file path')
    
def generation_opts(parser):
    group = parser.add_argument_group('generation-Parameters')
    group.add('-tagged_fi', '--tagged_fi', type=str, default='', help='path to tagged fi')
    