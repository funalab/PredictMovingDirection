import os
import csv
import copy
import numpy as np
import argparse
import json
from sklearn.svm import SVC

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
    return np.mean(cls_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Test conventional method for
                                                    predicting the direction of
                                                    cell movement''')
    parser.add_argument('feature_path', help='Feature data path')
    parser.add_argument('image_path', help='Image data path')
    parser.add_argument('out_dir', help='Output directory path')
    parser.add_argument('--fold', '-f', type=int, default=1,
                        help='Specify fold')
    parser.add_argument('--model', '-m', default='SVM',
                        help='Conventional method: {SVM, NLSVM, RF}')
    args = parser.parse_args()


    """ Load Feature of Image """
    with open(args.feature_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        f_list = [row for row in reader]
    x_train, y_train = annotation_dataset(
            os.path.join(args.image_path, 'fold_{}'.format(args.fold), 'train'),
            f_list)
    x_val, y_val = annotation_dataset(
            os.path.join(args.image_path, 'fold_{}'.format(args.fold), 'val'),
            f_list)
    x_test, y_test = annotation_dataset(
            os.path.join(args.image_path, 'test'),
            f_list)

    """ Initialization of Model and Training Phase """
    if args.model == 'SVM':
        model = SVC(kernel='linear', random_state=None)
        model.fit(x_train, y_train)


    """ Prediction Phase for Validation """
    pred_val = model.predict(x_val)
    mca_val = mca_score(y_val, pred_val)

    """ Prediction Phase for Test """
    pred_test = model.predict(x_test)
    mca_test = mca_score(y_test, pred_test)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    with open(os.path.join(args.out_dir, 'mca_score.txt'), 'w') as f:
        f.write('MCA val: {}\n'.format(mca_val))
        f.write('MCA test: {}\n'.format(mca_test))
