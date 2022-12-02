class gen_config(object):
    def __init__(self):
        
        self.beamsz = 4
        self.gen_wts = '1,1'
        self.cuda = False
        self.min_gen_tokes = 0
        self.min_gen_states = 0
        self.fine_tune = False
        
        self.load = 'models/chinese_models/ensemble_v2.pt.1'
        
        self.tagged_fi = 'segs/chinese_segs/ensemble_v2.txt'
    
    
