import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class Learner:
    """
    Responsible for training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    model_name:

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    Methods
    ------
    compute_gradients_and_loss:

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    evaluate_loader: evaluate `model` on a dataloader

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_gradients:

    free_memory:

    """
    def __init__(
            self,
            model,
            criterion,
            metric,
            device,
            optimizer,
            model_name=None,
            lr_scheduler=None,
            is_binary_classification=False,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        if criterion is not None:
            self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.is_binary_classification = is_binary_classification

        self.is_ready = True

        self.n_modules = self.__get_num_modules()
        self.model_dim = int(self.get_param_tensor().shape[0])

    def zero_grad(self):
        """
        Sets the gradients of all optimized `torch.Tensor`s.
        """
        for param in self.model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)

    def __get_num_modules(self):
        """
        computes the number of modules in the model network;
        i.e., the size of `self.model.modules()`

        return:
            n_modules (int)
        """
        if not self.is_ready:
            return

        n_modules = 0
        for _ in self.model.modules():
            n_modules += 1

        return n_modules

    def compute_stochastic_gradient(self, batch, weights=None, frozen_modules=None):
        """
        compute the stochastic gradient on one batch, the result is stored in self.model.parameters()

        :param batch: (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param frozen_modules: list of frozen modules; default is None

        :return:
            None
        """
        if frozen_modules is None:
            frozen_modules = []

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        for frozen_module in frozen_modules:
            frozen_module.zero_grad()

        loss.backward()

    def fit_batch(self, batch, weights=None, frozen_modules=None, dp_noise=None):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param frozen_modules: list of frozen modules; default is None
        :param dp_noise:

        :return:
            loss.item()
            metric.item()

        """
        if frozen_modules is None:
            frozen_modules = []

        self.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y) / len(y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        for frozen_module in frozen_modules:
            frozen_module.zero_grad()

        loss.backward()

        # Apply optimizer step with or without noise
        if dp_noise is not None:
            self.optimizer.step(dp_noise)
        else:
            self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.item(), metric.item()

    def fit_epoch(self, loader, weights=None, frozen_modules=None, dp_noise=None):
        """
        perform several optimizer steps on all batches drawn from `loader`

        :param loader:
        :type loader: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param frozen_modules: list of frozen models; default is None
        :param dp_noise:

        :return:
            loss.item()
            metric.item()

        """
        if frozen_modules is None:
            frozen_modules = []

        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in loader:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            loss.backward()

            for frozen_module in frozen_modules:
                frozen_module.zero_grad()

            # Apply optimizer step with or without noise
            if dp_noise is not None:
                self.optimizer.step(dp_noise)
            else:
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            global_loss += loss.item() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).item()

        return global_loss / n_samples, global_metric / n_samples

    def evaluate_loader(self, loader):
        """
        evaluate learner on `loader`

        :param loader:
        :type loader: torch.utils.data.DataLoader

        :return:
            global_loss and  global_metric accumulated over the dataloader

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)

                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).sum().item()
                global_metric += self.metric(y_pred, y).item()

                n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def set_param_tensor(self, param_tensor):
        """
        sets the parameters of the model from `param_tensor`

        :param param_tensor: torch.tensor of shape (`self.model_dim`,)

        """
        param_tensor = param_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            param.data = \
                param_tensor[current_index: current_index + current_dimension].reshape(param_shape)

            current_index += current_dimension

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def set_grad_tensor(self, grad_tensor):
        """

        Parameters
        ----------
        grad_tensor: torch.tensor of shape (`self.model_dim`,)

        Returns
        -------
            None

        """
        grad_tensor = grad_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            if param.grad is None:
                param.grad = param.data.clone()

            param.grad.data = \
                deepcopy(grad_tensor[current_index: current_index + current_dimension].reshape(param_shape))

            current_index += current_dimension

    def free_gradients(self):
        """
        free memory allocated by gradients

        """

        self.optimizer.zero_grad(set_to_none=True)

    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        if not self.is_ready:
            return

        self.free_gradients()

        del self.lr_scheduler
        del self.optimizer
        del self.model

        self.is_ready = False

    def __sub__(self, other):
        """differentiate learners

        returns a Learner object with the same parameters
        as self and gradients equal to the difference with respect to the parameters of "other"

        Remark: returns a copy of self, and self is modified by this operation

        Parameters
        ----------
        other: Learner

        Returns
        -------
            Learner
        """
        params = self.get_param_tensor()
        other_params = other.get_param_tensor()

        self.set_grad_tensor(other_params - params)

        return self

    def save_checkpoint(self, path):
        """
        save the model, the optimizer and thr learning rate scheduler state dictionaries

        :param: expected to be the path to a .pt file
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        load the model, the optimizer and thr learning rate scheduler state dictionaries

        :param: expected to be the path to a .pt file storing the required data
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
