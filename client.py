
import warnings


class Client(object):
    r"""
    Implements a client

    Attributes
    ----------
    learner
    train_loader
    val_loader
    test_loader
    n_train_samples
    n_test_samples
    logger
    counter
    __save_path
    __id

    Methods
    ----------
    __init__
    step
    write_logs
    update_tuned_learners

    """
    def __init__(
            self,
            learner,
            train_loader,
            val_loader,
            test_loader,
            logger,
            save_path=None,
            id_=None,
            *args,
            **kwargs
    ):
        """

        :param learner:
        :param train_loader:
        :param val_loader:
        :param test_loader:
        :param logger:
        :param save_path:

        """
        self.learner = learner

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.n_train_samples = len(self.train_loader.dataset)
        self.n_test_samples = len(self.test_loader.dataset)

        self.save_path = save_path

        self.id = -1
        if id_ is not None:
            self.id = id_

        self.counter = 0
        self.logger = logger

    def is_ready(self):
        return self.learner.is_ready

    def step(self, dp_noise=None):
        self.counter += 1

        batch = next(iter(self.train_loader))

        self.learner.fit_batch(
            batch=batch,
            dp_noise=dp_noise
        )

    def write_logs(self, counter=None):
        if counter is None:
            counter = self.counter

        train_loss, train_acc = self.learner.evaluate_loader(self.val_loader)
        test_loss, test_acc = self.learner.evaluate_loader(self.test_loader)

        if self.logger is not None:
            self.logger.add_scalar("Train/Loss", train_loss, counter)
            self.logger.add_scalar("Train/Metric", train_acc, counter)
            self.logger.add_scalar("Test/Loss", test_loss, counter)
            self.logger.add_scalar("Test/Metric", test_acc, counter)
            self.logger.flush()

        return train_loss, train_acc, test_loss, test_acc

    def save_state(self, path=None):
        """

        :param path: expected to be a `.pt` file

        """
        if path is None:
            if self.save_path is None:
                warnings.warn("client state was not saved", RuntimeWarning)
                return
            else:
                self.learner.save_checkpoint(self.save_path)
                return

        self.learner.save_checkpoint(path)

    def load_state(self, path=None):
        if path is None:
            if self.save_path is None:
                warnings.warn("client state was not loaded", RuntimeWarning)
                return
            else:
                self.learner.load_checkpoint(self.save_path)
                return

        self.learner.load_checkpoint(path)

    def free_memory(self):
        self.learner.free_memory()
