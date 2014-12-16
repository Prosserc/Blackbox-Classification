#/usr/bin/python
from __future__ import division
import sys, os, numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from csv import writer as csvwriter, reader as csvreader
from scipy.stats import nanmean
 
# globals - you will need to alter these for your problem...
default_dir = r'C:\Users\Chris\Dropbox\ProsserSystems\Python\machine_learning\sklearn_data'
default_trn_path = os.path.join(default_dir, 'train.csv')
default_tst_path = os.path.join(default_dir, 'test.csv')
delim = ','
# zero indexed, don't allow for exclusions, they are handled, -1 means no ids
trn_id_col = -1
tst_id_col = -1
labels_col = 0
# col nos to be excluded - zero indexed
excl_trn_cols = [] #e.g. [3,8,10]
excl_tst_cols = [] # e.g. -1 from above: [i-1 for i in excl_trn_cols]
heads_in_trn_file, heads_in_tst_file = False, False
 
# regularisation
auto_find_loops = 30 # WARNING this will run this no * iterations global times.
                     #         set to 1 to turn off and use reg default
                     #         (specified in main)
adj_rate = 2.0 # size of steps to take in auto_find * or / by this number
 
# These should have reasonable defaults if you are not sure...
trn_perc = 90
iterations = 1000
verbose = False
show_graphs = False # plots data by one feature against another, only useful if
                    # you have a very small no of features e.g.~ <=8
 
#initialisations
trn_cols, tst_cols = 0, 0
 
def scale_features(X, mu, sd, m, n):
    """
   Call with a numpy array for the values to be scaled down to approx -3 to 3.
   This is required for algorithms such as gradient descent so
   that the results can converge more efficiently.
   m & n args are for no of rows and no of cols respectively.
   Returns as a Numpy array with a feature added for x0 (a 1 for each row).
   """
    # function for each element (vectorised later)
    def scale(x, mu, sd):
        return (x-mu)/sd if sd != 0 else x
 
    # vectorise function above
    scale_vec = np.vectorize(scale, otypes=[np.float])
 
    if len(mu) == 0:
        mu = np.mean(X, axis=0)
    if len(sd) == 0:
        sd = np.std(X, axis=0)
    X_norm = np.ones((m, n+1))
    X_norm[:, 1:] = scale_vec(X, mu, sd)
    return X_norm, mu, sd
 
def graphs(y, X, m, n, label):
    # get indexes of positive and negative examples
    pos = [i for i in range(m) if y[i] == 1]
    neg = [i for i in range(m) if y[i] == 0]
 
    # plot stuff...
    if show_graphs:
        for j in range(1, n): # miss x0
            for j2 in range(j+1, n): # for all combos
                x1_p, x1_n = [X[j][i] for i in pos], [X[j][i] for i in neg]
                x2_p, x2_n = [X[j2][i] for i in pos], [X[j2][i] for i in neg]
                fig = plt.plot(x1_p, x2_p, 'go', x1_n, x2_n, 'rx')
                x_lab, y_lab = plt.xlabel(label[j]), plt.ylabel(label[j2])
                plt.show()
 
def write_to_file(data, f_name):
    writer = csvwriter(open(f_name, 'wb'))
    writer.writerows(data)
 
def feature_prep(data, heads, stage='training', use_mean=[], use_sd=[],
                 write_output_to_file=False, cur_loop=0):
    """
   use_mean and use_sd should not be specified for initial training data, but
   then should be passed in for cv, test and predictions (to use the same
   feature scaling as inital training data).
   """
 
    if cur_loop == 0 and verbose:
        l_verbose = True
    else:
        l_verbose = False
   
    # load in training data
    if l_verbose: print '\n', '-'*80, '\n', 'Stage:', stage
    ids = None
    if stage not in ['test', 'predict']:
        feature_cols = [i for i in range(trn_cols)
                        if i not in [trn_id_col, labels_col]+excl_trn_cols]
        if l_verbose:
            print 'feature_cols:', feature_cols
            print 'trn_cols:', trn_cols, '| trn_id_col:', trn_id_col, \
                  '| labels_col:', labels_col, '| excl_trn_cols:', excl_trn_cols
        X, y= data[:, feature_cols], data[:, labels_col]
 
        # filter heads
        if heads_in_trn_file:
                heads = [heads[i] for i in range(n) if i in feature_cols]
 
    else:
        feature_cols = [i for i in range(tst_cols)
                        if i not in [tst_id_col]+excl_tst_cols]
        if l_verbose:
            print 'feature_cols:', feature_cols
            print 'tst_cols:', tst_cols, '| tst_id_col:', tst_id_col, \
                  '| excl_tst_cols:', excl_tst_cols
        X, y = data[:, feature_cols], None
 
        # filter heads
        if heads_in_tst_file:
                heads = [heads[i] for i in range(n) if i in feature_cols]
 
        # record tst id columns if needed
        if tst_id_col >= 0:
                data[:, tst_id_col]
 
    m, n = np.size(X, 0), np.size(X, 1) # no of rows and cols
   
    if l_verbose:
        print 'Heads used:\n', ', '.join(i for i in heads)
        print 'X:', np.shape(X), 'y:', np.shape(y), 'ids:', np.shape(ids), \
               'data:', np.shape(data), 'm:', m, 'n:', n, '\n'
 
    # fill blanks with averages
    if len(use_mean) > 0:
        col_default = use_mean
    else:
        # calc means of cols if not passed in as args
        col_default = use_mean = nanmean(X, axis=0)
    inds = np.where(np.isnan(X)) # find indicies of empty cells to be replaced
    X[inds] = np.take(col_default, inds[1])
   
    if show_graphs:
        graphs(y, X, m, n, heads)
 
    #test
    if l_verbose:
        print '\nFirst ten rows before normalisation:'
        np.set_printoptions(precision=4, suppress=True)
        print X[:10, :], '\n'
   
    # scale the features & write output
    X, mu, sd = scale_features(X, use_mean, use_sd, m, n)
    if write_output_to_file:
        write_to_file(X, os.join.path(default_dir, 'X_'+stage+'.csv'))
   
    #test
    if l_verbose:
        print '\nFirst ten rows after normalisation:'
        print X[:10, 1:], '\n'
 
    return X, y, mu, sd, ids
 
def conv(val):
    try:
        return float(val)
    except:
        if not val:
            return None
        return float(sum([ord(i) for i in str(val)])) # sum of ascii vals
 
def import_data(mode='training'):
    global trn_cols, tst_cols
   
    # get input file (features and labels)
    if len(sys.argv) > 1:
        if mode == 'training':
            fname = sys.argv[1]
        elif len(sys.argv) > 2:
            fname = sys.argv[2]
    else:
        if mode == 'training':  
            fname = default_trn_path
        else:
            fname = default_tst_path
 
    if not os.path.exists(fname):
        print "usage:", os.path.split(sys.argv[0])[1], "[default_trn_path]", \
                  "[default_tst_path]"
        print "Valid file paths must be provided as an arg or global varables"
        sys.exit("invalid input")
 
    # get heads
    reader = csvreader(open(fname, 'rb'))
    r, start_row, heads = 0, 0, []
    for row in reader:
        if r == 0:
            # get no of cols in data
            if mode == 'training':
                trn_cols = len(row)
                if heads_in_trn_file:
                        heads = row
                        start_row += 1
            else:
                tst_cols = len(row)
                if heads_in_tst_file:
                        heads = row
                        start_row += 1
            r += 1
        else:
            break
 
    # build a dict to map each col to a conv func (if not excl)
    if mode not in ['test', 'predict']:
        cols = [i for i in range(trn_cols) if i not in excl_trn_cols]
        conv_dict = {c: conv for c in cols}
    else:
        cols = [i for i in range(tst_cols) if i not in excl_tst_cols]
        conv_dict = {c: conv for c in cols}
 
    if verbose:
        print '\nData import:', mode, '| cols:', cols, '\n'
 
    # import data
    #   not excluding unneeded cols, import all, just without conversions
    #   they are exlcuded later in feature_prep
    data = np.genfromtxt(fname, delimiter=delim, converters=conv_dict,
                                         skip_header=start_row)
 
    if verbose:
        print 'all heads:\n',  ', '.join(i for i in heads), '\n'
        print 'shape of data:', np.shape(data)
        print data
   
    return data, heads
 
def split_trn_data(data):
    m = np.size(data, 0)
    rands = np.random.random_sample(m)
   
    # select cases where random no from above is <= threshold
    trn_data = data[rands <= (trn_perc/100), :]
    cv_data = data[rands > (trn_perc/100), :]
 
    return trn_data, cv_data
 
def build_classifier(X, y, reg):
    # rbf is guassian kernal
    clf = svm.SVC(kernel='rbf', C=reg, cache_size=1000)
    return clf.fit(X, y)
 
def main():
 
    global adj_rate
    reg, reg_loop, reg_dir = 1.0, 0, 'up'
    reg_rec, trn_rec, cv_rec = [], [], []
   
    # import training data
    data, heads = import_data('training')
 
    while reg_loop < auto_find_loops:
       
        trn, cv, = [], []
        for i in range(iterations):
 
            # split data into training and cross validation groups
            trn_data, cv_data = split_trn_data(data)
            if verbose:
                print '\nSize of training data:', np.shape(trn_data)
                print 'Size of cross val data:', np.shape(cv_data)
           
            # prep training data and build classifier
            X_trn, y_trn, mu, sd, ids = feature_prep(trn_data, heads,
                                                     'training',
                                                     [], [], verbose, i)
            clf = build_classifier(X_trn, y_trn, reg)
 
            # training accuracy
            trn_pred = clf.predict(X_trn)
            trn_accuracy = 1 - (sum(abs(y_trn - trn_pred)) / len(X_trn))
            trn.append(trn_accuracy)
 
            # load prepare cv set
            if trn_perc < 100:
                X_cv, y_cv, mu, sd, ids = feature_prep(cv_data, heads, 'cv',
                                                       mu, sd, verbose, i)
 
                # cv accuracy
                cv_pred = clf.predict(X_cv)
                cv_accuracy = 1 - (sum(abs(y_cv - cv_pred)) / len(X_cv))
                cv.append(cv_accuracy)
 
        reg_rec.append(reg)
        trn_rec.append(np.mean(trn))
        if trn_perc < 100:
            cv_rec.append(np.mean(cv))
        else:
            cv_rec.append(0)
 
        if reg_loop == 0:
            print 'Loop  |  C param  |  Trn accuracy  |  CV accuracy   |  Dir'
            print '-----------------------------------------------------------'
 
        better = (reg_loop == 0 or cv_rec[reg_loop] > cv_rec[reg_loop-1])
 
        # switch direction & reduce adj_rate if not getting better
        if not better:
            adj_rate *= 0.95
            if reg_dir == 'up':
                reg_dir = 'down'
            else:
                reg_dir = 'up'
 
        try:
            print str(reg_loop) + ' ' * (6 - len(str(reg_loop))) + '|' + \
                  '  ' + str(round(reg, 3)) + \
                  ' ' * (9 - len(str(round(reg, 3)))) + '|' + \
                  '  ' + str(round(trn_rec[reg_loop], 9)) + \
                  ' ' * (14 - len(str(round(trn_rec[reg_loop], 9)))) + '|' + \
                  '  ' + str(round(cv_rec[reg_loop], 9)) + \
                  ' ' * (14 - len(str(round(cv_rec[reg_loop], 9)))) + '|' + \
                  '  ' + reg_dir
        except:
            print reg_loop, reg, trn_rec[reg_loop], cv_rec[reg_loop], reg_dir
            pass
 
        if reg_dir == 'up':
            reg *= adj_rate
        else:
            reg /= adj_rate
 
        reg_loop += 1
 
    # load in test data and run through the same prep / normalisation
    t_data, t_heads = import_data('test')
    X, tmp_y, mu, sd, ids = feature_prep(t_data, t_heads, 'test',
                                         mu, sd, verbose, i)
   
    # get predictions and make each item an int in a sublist (required format)
    y = clf.predict(X)
    print '\nFound', int(sum(y)), 'positive predictions out of', len(y)
    print '(iterations:', iterations, '| trn_perc:', trn_perc, ')'
 
    if tst_id_col >= 0:
            predictions = [[int(ids[i]), int(round(y[i],0))] for i in range(len(y))]
    else:
        predictions = [[int(round(y[i],0))] for i in range(len(y))]
    if heads_in_trn_file:
            predictions.insert(0, [t_heads[tst_id_col], t_heads[labels_col]])
 
    write_to_file(predictions,
                  os.path.join(default_dir, 'test_predictions.csv'))    
 
if __name__ == '__main__':
    main()
