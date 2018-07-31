from chainer.training.extensions import Evaluator


class testmode_evaluator(Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        model.predictor.train = False

        ret = super(testmode_evaluator, self).evaluate()

        model.predictor.train = True

        return ret
