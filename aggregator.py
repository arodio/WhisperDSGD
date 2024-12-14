import os
import time
import random


from abc import ABC, abstractmethod

from utils.torch_utils import *

from tqdm import tqdm

import numpy as np

import torch


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients

    test_clients

    n_clients:

    n_test_clients

    clients_weights:

    model_dim: dimension if the used model

    device:

    global_train_logger:

    global_test_logger:

    dp_noise_generator:

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    rng: random number generator

    Methods
    ----------
    __init__

    mix

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients,
            log_freq,
            global_train_logger,
            global_test_logger,
            dp_noise_generator,
            test_clients=None,
            verbose=0,
            seed=None
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.model_dim = self.clients[0].learner.model_dim
        self.device = self.clients[0].learner.device

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)

        self.clients_ids = [client.id for client in self.clients]

        self.clients_weights = \
                torch.ones(self.n_clients, dtype=torch.float32, device=self.device) / self.n_clients

        self.dp_noise_generator = dp_noise_generator

        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger
        self.log_freq = log_freq
        self.verbose = verbose

    @abstractmethod
    def mix(self, c_round):
        r"""mix clients

        Parameters
        ----------
        c_round: int

        Returns
        -------
            None
        """
        pass

    @abstractmethod
    def toggle_client(self, client_id, mode):
        """
        toggle client at index `client_id`, if `mode=="train"`, `client_id` is selected in `self.clients`,
        otherwise it is selected in `self.test_clients`.

        :param client_id: (int)
        :param mode: possible are "train" and "test"

        """
        pass

    def toggle_clients(self):
        for client_id in range(self.n_clients):
            self.toggle_client(client_id, mode="train")

    def toggle_test_clients(self):
        for client_id in range(self.n_test_clients):
            self.toggle_client(client_id, mode="test")

    def write_logs(self, counter):
        self.toggle_test_clients()

        for global_logger, clients, mode in [
            (self.global_train_logger, self.clients, "train"),
            (self.global_test_logger, self.test_clients, "test")
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs(counter=counter)

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")
                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}% |", end=" ")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * self.clients_weights[client_id]
                global_train_acc += train_acc * self.clients_weights[client_id]
                global_test_loss += test_loss * self.clients_weights[client_id]
                global_test_acc += test_acc * self.clients_weights[client_id]

            if self.verbose > 0:
                print("+" * 30)
                print(f"Global.. {counter}")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end=" ")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, counter)
            global_logger.add_scalar("Train/Metric", global_train_acc, counter)
            global_logger.add_scalar("Test/Loss", global_test_loss, counter)
            global_logger.add_scalar("Test/Metric", global_test_acc, counter)
            global_logger.flush()

        if self.verbose > 0:
            print("#" * 80)

    def evaluate(self):
        """
        evaluate the aggregator, returns the performance of every client in the aggregator

        :return:
            clients_results: (np.array of size (self.n_clients, 2, 2))
                number of correct predictions and total number of samples per client both for train part and test part
            test_client_results: (np.array of size (self.n_test_clients))
                number of correct predictions and total number of samples per client both for train part and test part

        """

        clients_results = []
        test_client_results = []

        for results, clients, mode in [
            (clients_results, self.clients, "train"),
            (test_client_results, self.test_clients, "test")
        ]:
            if len(clients) == 0:
                continue

            print(f"evaluate {mode} clients..")
            for client_id, client in enumerate(tqdm(clients)):
                if not client.is_ready():
                    self.toggle_client(client_id, mode=mode)

                _, train_acc, _, test_acc = client.write_logs()

                results.append([
                    [train_acc * client.n_train_samples, client.n_train_samples],
                    [test_acc * client.n_test_samples, client.n_test_samples]
                ])

                client.free_memory()

        return np.array(clients_results, dtype=np.uint16), np.array(test_client_results, dtype=np.uint16)


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self, c_round):

        for client in self.clients:
            client.step()

    def toggle_client(self, client_id, mode):
        pass


class CentralizedAggregator(Aggregator):
    r""" Centralized Aggregator.
     Sampled clients get synchronized with the average client.

    """
    def __init__(
            self,
            clients,
            global_learner,
            dp_noise_generator,
            log_freq,
            global_train_logger,
            global_test_logger,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(CentralizedAggregator, self).__init__(
            clients,
            log_freq,
            global_train_logger,
            global_test_logger,
            dp_noise_generator,
            test_clients,
            verbose,
            seed
        )

        self.global_learner = global_learner

    def mix(self, c_round):

        for client in self.clients:
            client.step()

        learners = [client.learner for client in self.clients]

        self.global_learner.optimizer.zero_grad()

        average_learners(
            learners=learners,
            target_learner=self.global_learner,
            weights=self.clients_weights,
            average_params=False,
            average_gradients=True
        )

        # Get central differential privacy noise
        dp_noise = self.get_dp_noise(c_round)

        # Apply optimizer step with or without noise
        if dp_noise is not None:
            self.global_learner.optimizer.step(dp_noise)
        else:
            self.global_learner.optimizer.step()

        self.toggle_clients()

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.is_ready():
            copy_model(client.learner.model, self.global_learner.model)
        else:
            client.learner = deepcopy(self.global_learner)

        if callable(getattr(client.learner.optimizer, "set_initial_params", None)):
            client.learner.optimizer.set_initial_params(
                self.global_learner.model.parameters()
            )

    def save_state(self, dir_path, counter):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:
        :param counter:

        """
        save_path = os.path.join(dir_path, f"global_round_{counter}.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

        for client_id, client in enumerate(self.clients):
            save_path = os.path.join(dir_path, f"client_{client_id}_round_{counter}.pt")
            self.toggle_client(client_id, mode="train")
            client.save_state(save_path)
            client.free_memory()

    def load_state(self, dir_path, counter):
        """
        load the state of the aggregator

        :param dir_path:
        :param counter:

        """
        chkpts_path = os.path.join(dir_path, f"global_round_{counter}.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))
        for client_id, client in self.clients:
            chkpts_path = os.path.join(dir_path, f"client_{client_id}_round_{counter}.pt")
            self.toggle_client(client_id, mode="train")
            client.load_state(chkpts_path)
            client.free_memory()

    def get_dp_noise(self, c_round):
        if self.dp_noise_generator is None:
            return None

        # Get central differential privacy noise
        dp_noise = self.dp_noise_generator.get_dp_noise(c_round)  # Shape: (n_params)
        return torch.tensor(dp_noise, dtype=torch.float32, device=self.device)


class DecentralizedAggregator(Aggregator):
    r""" Decentralized Aggregator.
     Clients average models within their neighbors.

    """
    def __init__(
            self,
            clients,
            weight_matrix,
            log_freq,
            global_train_logger,
            global_test_logger,
            dp_noise_generator,
            test_clients=None,
            verbose=0,
            seed=None,
    ):
        super(DecentralizedAggregator, self).__init__(
            clients,
            log_freq,
            global_train_logger,
            global_test_logger,
            dp_noise_generator,
            test_clients,
            verbose,
            seed
        )

        self.weight_matrix = weight_matrix

        self.learners_temp = [deepcopy(client.learner) for client in self.clients]

    def mix(self, c_round):

        # Get differential privacy noise
        dp_noise = self.get_dp_noise(c_round)

        # Apply client step with or without noise
        for client_id, client in enumerate(self.clients):
            if dp_noise is not None:
                client_noise = dp_noise[client_id]
                client.step(client_noise)
            else:
                client.step()

        learners = [client.learner for client in self.clients]

        for client_id, client in enumerate(self.clients):
            neighbors_ids = [idx for idx, weight in enumerate(self.weight_matrix[client_id]) if weight > 0]
            neighbor_learners = [learners[neighbor_id] for neighbor_id in neighbors_ids]
            neighbor_weights = self.weight_matrix[client_id][neighbors_ids]

            average_learners(
                learners=neighbor_learners,
                target_learner=self.learners_temp[client_id],
                weights=torch.tensor(neighbor_weights, dtype=torch.float32, device=self.device),
                average_params=True,
                average_gradients=False
            )

        self.toggle_clients()

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.is_ready():
            copy_model(client.learner.model, self.learners_temp[client_id].model)
        else:
            client.learner = deepcopy(self.learners_temp[client_id])

        if callable(getattr(client.learner.optimizer, "set_initial_params", None)):
            client.learner.optimizer.set_initial_params(
                self.learners_temp[client_id].model.parameters()
            )

    def save_state(self, dir_path, counter):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:
        :param counter:

        """
        for client_id, client in enumerate(self.clients):
            save_path = os.path.join(dir_path, f"client_{client_id}_round_{counter}.pt")
            self.toggle_client(client_id, mode="train")
            client.save_state(save_path)
            client.free_memory()

    def load_state(self, dir_path, counter):
        """
        load the state of the aggregator

        :param dir_path:
        :param counter:

        """
        for client_id, client in self.clients:
            chkpts_path = os.path.join(dir_path, f"client_{client_id}_round_{counter}.pt")
            self.toggle_client(client_id, mode="train")
            client.load_state(chkpts_path)
            client.free_memory()

    def get_dp_noise(self, c_round):
        if self.dp_noise_generator is None:
            return None

        # Get local differential privacy noise
        dp_noise = self.dp_noise_generator.get_dp_noise(c_round).T  # Shape: (n_clients, n_params)
        return torch.tensor(dp_noise, dtype=torch.float32, device=self.device)
