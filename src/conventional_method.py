import os
import csv
import copy
import numpy as np
import argparse
import json
import configparser
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def annotation_dataset(image_path, f_list):
    x, y = [], []
    file_name = np.array(f_list)[:, 0]
    cls_name = ['0_upper_right', '1_upper_left', '2_lower_left', '3_lower_right']
    for cn in cls_name:
        dlist = os.listdir(os.path.join(image_path, cn))
        for dl in dlist:
            x.append(f_list[list(file_name).index(dl)][2:])
            y.append(cls_name.index(cn))
    return x, y

def mca_score(y, pred):
    assert np.shape(pred) == np.shape(y)
    cls_count = np.zeros((np.max(y) + 1))
    for num in range(len(pred)):
        if pred[num] == y[num]:
            cls_count[y[num]] += 1
    cls_acc = cls_count / np.unique(y, return_counts=True)[1]
    return np.mean(cls_acc), np.std(cls_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Test conventional method for
                                                    predicting the direction of
                                                    cell movement''')
    parser.add_argument('--conf_file', '-c', help='Feature data path')
    args = parser.parse_args()

    """ Config Parser """
    conf = configparser.ConfigParser()
    conf.read(args.conf_file)

    """ Load Feature of Image """
    with open(conf['RunTime']['feature_path'], 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        f_list = [row for row in reader]
    x_train, y_train = annotation_dataset(
            os.path.join(conf['RunTime']['image_path'], 'fold_{}'.format(conf['RunTime']['fold']), 'train'),
            f_list)
    x_val, y_val = annotation_dataset(
            os.path.join(conf['RunTime']['image_path'], 'fold_{}'.format(conf['RunTime']['fold']), 'val'),
            f_list)
    x_test, y_test = annotation_dataset(
            os.path.join(conf['RunTime']['image_path'], 'test'),
            f_list)


    """ Initialization of Model and Training Phase """
    if conf['RunTime']['model'] == 'SVM' or conf['RunTime']['model'] == 'NLSVM':
        model = svm.SVC(
                C=eval(conf['Param']['C']),
                kernel=eval(conf['Param']['kernel']),
                degree=eval(conf['Param']['degree']),
                gamma=eval(conf['Param']['gamma']),
                coef0=eval(conf['Param']['coef0']),
                shrinking=eval(conf['Param']['shrinking']),
                probability=eval(conf['Param']['probability']),
                tol=eval(conf['Param']['tol']),
                class_weight=eval(conf['Param']['class_weight']),
                verbose=eval(conf['Param']['verbose']),
                max_iter=eval(conf['Param']['max_iter']),
                #decision_function_shape=eval(conf['Param']['decision_function_shape']),
                random_state=eval(conf['Param']['random_state'])
            )
    elif conf['RunTime']['model'] == 'RF':
        model = RandomForestClassifier(
                n_estimators=eval(conf['Param']['n_estimators']),
                criterion=eval(conf['Param']['criterion']),
                max_depth=eval(conf['Param']['max_depth']),
                min_samples_split=eval(conf['Param']['min_samples_split']),
                min_samples_leaf=eval(conf['Param']['min_samples_leaf']),
                min_weight_fraction_leaf=eval(conf['Param']['min_weight_fraction_leaf']),
                max_features=eval(conf['Param']['max_features']),
                max_leaf_nodes=eval(conf['Param']['max_leaf_nodes']),
                min_impurity_decrease=eval(conf['Param']['min_impurity_decrease']),
                min_impurity_split=eval(conf['Param']['min_impurity_split']),
                bootstrap=eval(conf['Param']['bootstrap']),
                oob_score=eval(conf['Param']['oob_score']),
                n_jobs=eval(conf['Param']['n_jobs']),
                random_state=eval(conf['Param']['random_state']),
                verbose=eval(conf['Param']['verbose']),
                warm_start=eval(conf['Param']['warm_start']),
                class_weight=eval(conf['Param']['class_weight'])
            )
    else:
        sys.exit()
    model.fit(x_train, y_train)


    """ Prediction Phase for Validation """
    pred_val = model.predict(x_val)
    mca_mean_val, mca_std_val = mca_score(y_val, pred_val)


    """ Prediction Phase for Test """
    pred_test = model.predict(x_test)
    mca_mean_test, mca_std_test = mca_score(y_test, pred_test)

    if not os.path.exists(conf['RunTime']['out_dir']):
        os.mkdir(conf['RunTime']['out_dir'])
    with open(os.path.join(conf['RunTime']['out_dir'], 'mca_score.txt'), 'w') as f:
        f.write('MCA val: {} ({})\n'.format(mca_mean_val, mca_std_val))
        f.write('MCA test: {} ({})\n'.format(mca_mean_test, mca_std_test))
