from chainer.training import extension


class scale_lr(extension.Extension):
    trigger = 1, 'epoch'
    priority = extension.PRIORITY_READER

    def __init__(self, attr='lr', rate=0.9, target=None, optimizer=None,
                 log_report='LogReport', key='validation_in_mca/main/mca'):
        self._attr = attr
        if rate < 0:
            raise ValueError('scale_lr does not support negative rate')
        self._rate = rate
        self._target = target
        self._optimizer = optimizer
        self._log_report = log_report
        self._key = key
        self._before_training = True

    def __call__(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')

        log_report = trainer.get_extension(self._log_report)
        log = log_report.log
        current_ob = log[-1][self._key]
        best_ob = optimizer.target.predictor.best_score[self._key]

        if not current_ob > best_ob:
            value = getattr(optimizer, self._attr) * self._rate
            if self._target is not None:
                if self._rate > 1:
                    if value / self._target > 1:
                        value = self._target
                else:
                    if value / self._target < 1:
                        value = self._target
            setattr(optimizer, self._attr, value)
