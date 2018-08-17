import os

from chainer.training import extension
from chainer import link
import chainer.training.trigger as trigger_module
from chainer.training.extensions import log_report as log_report_module
from chainer import serializers


class best_scoring(extension.Extension):
    trigger = 1, 'epoch'
    priority = extension.PRIORITY_READER

    def __init__(self, target, keys, s_keys=[], mname=None,
                 log_report='LogReport', trigger=(1, 'epoch')):
        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target
        self._keys = keys
        self._s_keys = s_keys
        self._mname = mname
        self._log_report = log_report
        self.trigger = trigger_module.get_trigger(trigger)

    def __call__(self, trainer):
        log_report = self._log_report

        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        current_ob = log[-1]
        epoch = current_ob['epoch']

        target = self._targets['main']

        for k in self._keys:
            updated = False
            if 'loss' in k:
                if epoch == 1:
                    target.predictor.best_score[k] = current_ob[k]
                elif current_ob[k] <= target.predictor.best_score[k]:
                    target.predictor.best_score[k] = current_ob[k]
                    updated = True
            else:
                if current_ob[k] >= target.predictor.best_score[k]:
                    target.predictor.best_score[k] = current_ob[k]
                    updated = True

            if updated and k in self._s_keys and self._mname is not None:
                out = trainer.out
                mpath = os.path.join(out, '{}.npz'.format(self._mname))
                serializers.save_npz(mpath, target.predictor)
