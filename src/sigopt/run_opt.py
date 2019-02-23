import os
import time
import json
from sigopt import Connection
from argparse import ArgumentParser

api_token = "GXFCJRJGZTPTFZTKXYJKJCHIZGNJHVFGUOFMABOKUNSYXGAB"
dev_token = "MPTPMDWQKNCJFVWPZGIQTVWBYOGRVIVCEOKYUWOIHCPNQOOH"
my_token = dev_token


def main():
    ap = ArgumentParser(description='python run_opt.py')
    ap.add_argument('--iteration', type=int, default=100, help='Specify Number of Iteration')
    ap.add_argument('--outdir', '-o', nargs='?', default='opt_result', help='Specify output files directory for create segmentation image and save model file')
    ap.add_argument('--num_parallel', type=int, default=4, help='Specify Number of parallel jobs')
    args = ap.parse_args()

    opbase = create_opbase(args.outdir)

    conn = Connection(client_token=my_token)
    experiment = conn.experiments().create(
        name="NIH/3T3 model",
        parameters=[
            dict(name='ksize conv_layer0', type='int', bounds=dict(min=1, max=3)),
            dict(name='out_channel conv_layer0', type='int', bounds=dict(min=1, max=6)),
            dict(name='ksize conv_layer1', type='int', bounds=dict(min=1, max=3)),
            dict(name='out_channel conv_layer1', type='int', bounds=dict(min=1, max=6)),
            dict(name='ksize conv_layer2', type='int', bounds=dict(min=1, max=3)),
            dict(name='out_channel conv_layer2', type='int', bounds=dict(min=1, max=6)),
            dict(name='ksize conv_layer3', type='int', bounds=dict(min=1, max=3)),
            dict(name='out_channel conv_layer3', type='int', bounds=dict(min=1, max=6)),
            dict(name='ksize conv_layer4', type='int', bounds=dict(min=1, max=3)),
            dict(name='out_channel conv_layer4', type='int', bounds=dict(min=3, max=8)),
            dict(name='ksize conv_layer5', type='int', bounds=dict(min=1, max=3)),
            dict(name='out_channel conv_layer5', type='int', bounds=dict(min=3, max=8)),
            dict(name='ksize conv_layer6', type='int', bounds=dict(min=1, max=3)),
            dict(name='out_channel conv_layer6', type='int', bounds=dict(min=5, max=10)),
            dict(name='ksize conv_layer7', type='int', bounds=dict(min=1, max=3)),
            dict(name='out_channel conv_layer7', type='int', bounds=dict(min=5, max=10)),
            dict(name='out_size fc_layer0', type='int', bounds=dict(min=1, max=10)),
            dict(name='dr', type='int', bounds=dict(min=1, max=9)),
            dict(name='l2', type='int', bounds=dict(min=0, max=100)),
            dict(name='lr', type='int', bounds=dict(min=1, max=100)),
            dict(name='bsize', type='int', bounds=dict(min=1, max=32)),
            dict(name='momentum', type='int', bounds=dict(min=1, max=10)),
        ],
    )
    print("Created experiment: https://sigopt.com/experiment/" + experiment.id)

    for i in range(1, args.iteration+1):
        suggestion = conn.experiments(experiment.id).suggestions().create()
        arch_list, para_list = convert_arch_format(suggestion.assignments)
        with open(os.path.join(opbase, 'arch_{}.json'.format(i)), 'w') as f:
            json.dump(arch_list, f)
        with open(os.path.join(opbase, 'para_{}.json'.format(i)), 'w') as f:
            json.dump(para_list, f)

        for k in range(1, args.num_parallel+1):
            with open(opbase + '/{}-{}.ssh'.format(i, k), 'w') as f:
                f.write('#!/bin/sh\n')
                f.write('#PBS -q l-regular\n')
                f.write('#PBS -l select=1:mpiprocs=1:ompthreads=1\n')
                f.write('#PBS -W group_list=gi95\n')
                f.write('#PBS -l walltime=24:00:00\n')
                f.write('export PYTHONUSERBASE=/lustre/gi95/i95000/pmd\n')
                f.write('cd $PBS_O_WORKDIR\n')
                f.write('. /etc/profile.d/modules.sh\n')
                f.write('module load anaconda3/4.3.0\n')
                f.write('module load chainer/1.24.0\n')
                f.write('module load cuda\n')
                f.write('python src/train_test.py nishimoto_revise/train_val_test/NIH3T3/fold_1/train nishimoto_revise/train_val_test/NIH3T3/fold_1/val {}/{}-{} --arch_path {}/arch_{}.json --param_path {}/para_{}.json\n'.format(opbase, i, k, opbase, i, opbase, i))

            os.system('qsub {}/{}-{}.ssh'.format(opbase, i, k))
        for t in range(288):  # attack 5 min until 24 hour
            try:
                val = 0
                for k in range(1, args.num_parallel+1):
                    with open('opt_iteration_{}_{}/best_score.json'.format(i, k), 'r') as f:
                        val += json.load(f)['validation_in_mca/main/mca']
                val /= args.num_parallel
                print('Read ... {}/outval_{}.txt'.format(opbase, i))
                print('best MCA: {}'.format(val))
                break
            except:
                time.sleep(300)

        with open(os.path.join(opbase + psep + 'result.txt'), 'a') as f:
            f.write('=======================================\n')
            f.write('[iter: {}]\n'.format(i))
            f.write('Parameter: {}\n'.format(suggestion.assignments.items()))
            f.write('best MCA: {}\n'.format(val))

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=val,
        )


    best_assignments_list = (
        conn.experiments(experiment.id).best_assignments().fetch()
    )

    if best_assignments_list.data:
        best_assignments = best_assignments_list.data[0].assignments
        print(best_assignments)
        with open(os.path.join(opbase + psep + 'result.txt'), 'a') as f:
            f.write('=======================================\n')
            f.write('[Best Parameter]\n')
            f.write(best_assignments)


def generateParameterList(para_list, sug_key):
    for key in sug_key:
        para_list[key] = 0

def convertParameter(para_list, sug_list):
    for sl in sug_list:
        if sl[0] == 'init channels':
            para_list[str(sl[0])] = sl[1] * 4
        elif sl[0] == 'ap factor' and sl[1] == 1:
            para_list[str(sl[0])] = 1.5
        elif sl[0] == 'kernel size':
            para_list[str(sl[0])] = sl[1] * 2 + 1
        else:
            para_list[str(sl[0])] = sl[1]

def create_opbase(opbase):
    if (opbase[len(opbase) - 1] == '/'):
        opbase = opbase[:len(opbase) - 1]
    if not (opbase[0] == '/'):
        if (opbase.find('./') == -1):
            opbase = './' + opbase
    t = time.ctime().split(' ')
    if t.count('') == 1:
        t.pop(t.index(''))
    opbase = opbase + '_' + t[1] + t[2] + t[0] + '_' + t[4] + '_' + t[3].split(':')[0] + t[3].split(':')[1] + t[3].split(':')[2]
    os.makedirs(opbase, exist_ok=True)
    print('Output Directory: {}'.format(opbase))
    return opbase

def convert_arch_format(suggest):

    convert_arch = {"0": [
        {"ksize": 3, "in_channel": 1, "name": "conv_layer0", "out_channel": 16, "pad": 0},
        {"ksize": 3, "in_channel": None, "name": "conv_layer1", "out_channel": 16, "pad": 0},
        {"ksize": 2, "stride": 2, "name": "max_pool0"},
        {"ksize": 3, "in_channel": None, "name": "conv_layer2", "out_channel": 16, "pad": 0},
        {"ksize": 3, "in_channel": None, "name": "conv_layer3", "out_channel": 16, "pad": 0},
        {"ksize": 2, "stride": 2, "name": "max_pool1"},
        {"ksize": 3, "in_channel": None, "name": "conv_layer4", "out_channel": 32, "pad": 0},
        {"ksize": 3, "in_channel": None, "name": "conv_layer5", "out_channel": 32, "pad": 0},
        {"ksize": 2, "stride": 2, "name": "max_pool2"},
        {"ksize": 3, "in_channel": None, "name": "conv_layer6", "out_channel": 64, "pad": 0},
        {"ksize": 3, "in_channel": None, "name": "conv_layer7", "out_channel": 64, "pad": 0},
        {"ksize": 2, "stride": 2, "name": "max_pool3"},
        {"out_size": 100, "name": "fc_layer0", "in_size": None},
        {"out_size": 4, "name": "out_layer", "in_size": None}
    ]}

    convert_para = \
        {"weighting": True, "rate": 0.9, "bn": False, "dr": 0.5, "l2": 0, "lr": 0.01, "bsize": 32, "momentum": 0.9}

    for l in convert_arch['0']:
        if 'ksize {}'.format(l['name']) in suggest.keys():
            l['ksize'] = suggest['ksize {}'.format(l['name'])] * 2 + 1
        if 'out_channel {}'.format(l['name']) in suggest.keys():
            l['out_channel'] = suggest['out_channel {}'.format(l['name'])] * 8
        if 'out_size {}'.format(l['name']) in suggest.keys():
            l['out_size'] = suggest['out_size {}'.format(l['name'])] * 100

    convert_para['dr'] = suggest['dr'] * 0.1
    convert_para['l2'] = suggest['l2'] * 0.00001
    convert_para['lr'] = suggest['lr'] * 0.001
    convert_para['bsize'] = suggest['bsize'] * 2
    convert_para['momentum'] = suggest['momentum'] * 0.1

    return convert_arch, convert_para


if __name__ == '__main__':
    main()
