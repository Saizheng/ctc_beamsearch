import numpy as np
import pdb

def joint(char, y, input_dist, alphabet, B, Pr, Pr0):
    if y == '':
        return input_dist[alphabet.index(char)] + Pr[B.index(y)]
    else:
        if char == y[-1]:
            out = input_dist[alphabet.index(char)] + Pr0[B.index(y)]
        else:
            out = input_dist[alphabet.index(char)] + Pr[B.index(y)]
        return out

def ctc_beamsearch(input_dist, alphabet='-abcdefghijklmnopqrstuvwxyz', k = 10):
    # input_dist should be numpy matrix
    # beamsize is k
    T = input_dist.shape[1]
    B = ['']
    Pr = [0]
    Pr1 = [-1e10]
    Pr0 = [0]
    for t in xrange(T):
        # get k most probable sequences in B
        #print 't is ' + str(t)
        B_new = []
        Pr_new = []
        Pr1_new = []
        Pr0_new = []
        ind = np.argsort(Pr)[::-1]
        B_ = [B[i] for i in ind[:k]]
        for y in B_:
            #print 'y is ' + y
            if y != '':
                pr1 = Pr1[B.index(y)] + input_dist[t, alphabet.index(y[-1])]
                if y[:-1] in B_:
                    pr1 = np.logaddexp(pr1, joint(y[-1], y[:-1], input_dist[t], alphabet, B, Pr, Pr0))
            pr0 = Pr[B.index(y)] + input_dist[t, alphabet.index('-')]
            if y == '':
                pr1 = -1e10
            B_new += [y]
            Pr_new += [np.logaddexp(pr1, pr0)]
            Pr1_new += [pr1]
            Pr0_new += [pr0]
            for c in alphabet[1:]:
                #print 'c is ' + c
                pr0 = -1e10
                pr1 = joint(c, y, input_dist[t], alphabet, B, Pr, Pr0)
                Pr0_new += [pr0]
                Pr1_new += [pr1] 
                Pr_new += [np.logaddexp(pr1, pr0)]
                B_new += [y + c]
        B = B_new; Pr = Pr_new; Pr1 = Pr1_new; Pr0 = Pr0_new;
    out_ind = np.argmax([Pr[i]/len(B[i]) for i in xrange(len(B))])
    return B[out_ind] 

if __name__ == '__main__':
    dist = np.log(np.array([[0.2, 0.3, 0.5],
                             [0.4, 0.4, 0.2],
                             [0.1, 0.3, 0.6], 
                             [0.5, 0.2, 0.3]])) 
    ctc_beamsearch(dist, '-ab', k=2)
