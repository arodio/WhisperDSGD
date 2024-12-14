from aggregator import *
from client import *
from learner import *
from models import *
from datasets import *
from diff_privacy import *

from .constants import *
from .metrics import *
from .optim import *
from .graph import *

from torch.utils.data import DataLoader
import torch.optim as optim


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment
    :param experiment_name: name of the experiment
    :return: str
    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_loader(
        type_,
        path,
        batch_size,
        train,
        inputs=None,
        targets=None,
):
    """
    constructs a torch.utils.DataLoader object from the given path
    :param type_: type of the dataset; possible are `tabular`, `mnist`, `cifar10` and `cifar100`, `emnist`,
     `femnist` and `shakespeare`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None

    :return: torch.utils.DataLoader
    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "a9a":
        dataset = \
            SubA9A(
                path,
                a9a_data=inputs,
                a9a_targets=targets
            )
    elif type_ == "mnist":
        dataset = \
            SubMNIST(
                path,
                mnist_data=inputs,
                mnist_targets=targets
            )
    elif type_ == "cifar10":
        dataset = \
            SubCIFAR10(
                path,
                cifar10_data=inputs,
                cifar10_targets=targets
            )
    elif type_ == "cifar100":
        dataset = \
            SubCIFAR100(
                path,
                cifar100_data=inputs,
                cifar100_targets=targets
            )
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = \
            CharacterDataset(
                path,
                chunk_len=SHAKESPEARE_CONFIG["chunk_len"]
            )
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


def get_loaders(type_, data_dir, batch_size, is_validation):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_loader`, `val_loader` and `test_loader`;
     `val_loader` iterates on the same dataset as `train_loader`, the difference is only in drop_last
    :param type_: type of the dataset;
    :param data_dir: directory of the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test

    :return:
        train_loader, val_loader, test_loader
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])
    """
    if type_ == "a9a":
        inputs, targets = get_a9a()
    elif type_ == "mnist":
        inputs, targets = get_mnist()
    elif type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    else:
        inputs, targets = None, None

    train_loaders, val_loaders, test_loaders = [], [], []

    for task_id, task_dir in enumerate(os.listdir(data_dir)):
        task_data_path = os.path.join(data_dir, task_dir)

        train_loader = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True
            )

        val_loader = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_loader = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders


def get_model(name, model_name, device, input_dimension=None, hidden_dimension=None, chkpts_path=None):
    """
    create model and initialize it from checkpoints

    :param name: experiment's name

    :param model_name: the name of the model to be used, only used when experiment is CIFAR-10, CIFAR-100 or FEMNIST
            possible are mobilenet and resnet

    :param device: either cpu or cuda

    :param input_dimension:

    :param hidden_dimension:

    :param chkpts_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
                        otherwise the weights are initialized randomly; default is None.
    """
    if name == "synthetic":
        model = TwoLinearLayers(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=1
        )
    elif name == "a9a":
        model = LinearLayer(input_dimension=123, num_classes=1)
    elif name == "mnist":
        model = LinearLayer(input_dimension=28*28, num_classes=10)
    elif name == "cifar10":
        model = get_resnet18(num_classes=10)
    elif name == "cifar100":
        if model_name == "mobilenet":
            model = get_mobilenet(num_classes=100)
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "femnist":
        if model_name == "mobilenet":
            model = get_mobilenet(num_classes=62)
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "shakespeare":
        model = NextCharacterLSTM(
            input_size=SHAKESPEARE_CONFIG["input_size"],
            embed_size=SHAKESPEARE_CONFIG["embed_size"],
            hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
            output_size=SHAKESPEARE_CONFIG["output_size"],
            n_layers=SHAKESPEARE_CONFIG["n_layers"]
        )

    else:
        raise NotImplementedError(
            f"{name} is not available!"
            f" Possible are: `cifar10`, `cifar100`, `emnist`, `femnist` and `shakespeare`."
        )

    if chkpts_path is not None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.load_state_dict(torch.load(chkpts_path, map_location=map_location)['model_state_dict'])
        except KeyError:
            try:
                model.load_state_dict(torch.load(chkpts_path, map_location=map_location)['net'])
            except KeyError:
                model.load_state_dict(torch.load(chkpts_path, map_location=map_location))

    model = model.to(device)

    return model


def get_optimizer(
        optimizer_name,
        model,
        lr_initial,
        momentum,
        weight_decay,
        dp_mechanism,
        l2_norm_clip,
        minibatch_size,
        is_aggregator
):
    """
    Gets torch.optim.Optimizer given a DP mechanism and is_aggregator.

    Parameters:
    - optimizer_name: str, possible values are {"adam", "sgd"}.
    - model: nn.Module, the model to optimize.
    - lr_initial: float, initial learning rate.
    - dp_mechanism: str or None, differential privacy mechanism ("cdp", "ldp", "pairwise", "mixing").
    - momentum: float, default=0.0, momentum for SGD.
    - weight_decay: float, default=1e-4, weight decay regularization.
    - is_aggregator: bool, default=False, whether the optimizer is for the aggregator.

    Returns:
    - torch.optim.Optimizer
    """
    if optimizer_name == "adam":
        optimizer = optim.Adam
    elif optimizer_name == "sgd":
        optimizer = optim.SGD
    else:
        raise NotImplementedError("Other optimizer are not implemented")

    if dp_mechanism and (dp_mechanism != "cdp" or is_aggregator):
        dp_optimizer = make_dp_optimizer(optimizer)
        return dp_optimizer(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            momentum=momentum,
            weight_decay=weight_decay,
            l2_norm_clip=l2_norm_clip,
            minibatch_size=minibatch_size
        )
    else:
        return optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            momentum=0.0,
            weight_decay=0.0
        )


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None, steps_per_epoch=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer

    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :param steps_per_epoch: number of batches in one epoch, only used if `scheduler_name == one_cycle_lr`
    :type steps_per_epoch: int
    :return: torch.optim.lr_scheduler

    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds // 2, 3 * (n_rounds // 4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    elif scheduler_name == "one_cycle_lr":
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=steps_per_epoch, epochs=n_rounds)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")


def get_learner(
        name,
        model_name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        n_rounds,
        seed,
        momentum,
        weight_decay,
        dp_mechanism,
        l2_norm_clip,
        minibatch_size,
        is_aggregator,
        steps_per_epoch=None,
        input_dimension=None,
        hidden_dimension=None,
        chkpts_path=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
            {`synthetic`, `cifar10`, `emnist`, `shakespeare`}

    :param model_name: the name of the model to be used, only used when experiment is CIFAR-10, CIFAR-100 or FEMNIST
            possible are mobilenet and resnet

    :param device: used device; possible `cpu` and `cuda`

    :param optimizer_name: passed as argument to utils.optim.get_optimizer

    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler

    :param initial_lr: initial value of the learning rate

    :param momentum: momentum parameter

    :param weight_decay: weight decay parameter

    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;

    :param steps_per_epoch: number of batches in one epoch, only used if `scheduler_name == one_cycle_lr`;

    :param seed:

    :param input_dimension:

    :param hidden_dimension:

    :param dp_mechanism:

    :param l2_norm_clip:

    :param minibatch_size:

    :param is_aggregator:


    :param chkpts_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
            otherwise the weights are initialized randomly; default is None.

    :return: Learner


    """
    torch.manual_seed(seed)

    if name == "synthetic":
        criterion = nn.MSELoss(reduction="none").to(device)
        metric = mse
        is_binary_classification = True
    elif name == "a9a":
        criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
        metric = binary_accuracy
        is_binary_classification = True
    elif name == "mnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8
        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        is_binary_classification = False
    else:
        raise NotImplementedError

    model = \
        get_model(
            name=name,
            model_name=model_name,
            device=device,
            chkpts_path=chkpts_path,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension
        )

    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dp_mechanism=dp_mechanism,
            l2_norm_clip=l2_norm_clip,
            minibatch_size=minibatch_size,
            is_aggregator=is_aggregator
        )

    lr_scheduler = \
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds,
            steps_per_epoch=steps_per_epoch,
        )

    return Learner(
        model=model,
        model_name=model_name,
        criterion=criterion,
        metric=metric,
        device=device,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        is_binary_classification=is_binary_classification
    )


def get_client(
        learner,
        train_loader,
        val_loader,
        test_loader,
        logger,
        client_id=None,
        save_path=None,
):
    """

    :param learner:
    :param train_loader:
    :param val_loader:
    :param test_loader:
    :param logger:
    :param client_id:
    :param save_path:

    :return:
        Client

    """
    return Client(
        learner=learner,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        logger=logger,
        id_=client_id,
        save_path=save_path
    )


def get_dp_noise_generator(
        dp_mechanism,
        epsilon,
        delta,
        norm_clip,
        n_clients,
        n_rounds,
        model_dim,
        weight_matrix=None,
        seed=None
):
    """

    :param dp_mechanism: str, DP mechanism ('cdp', 'ldp', 'pairwise', 'mixing')
    :param epsilon: float, privacy budget
    :param delta: float, privacy parameter
    :param norm_clip: float, gradient clip constant
    :param n_clients: int
    :param n_rounds: int
    :param model_dim: int
    :param weight_matrix: np.ndarray
    :param seed: int

    :return: DPNoiseGenerator
    """
    if dp_mechanism is None:
        return None

    return DPNoiseGenerator(
        dp_mechanism=dp_mechanism,
        epsilon=epsilon,
        delta=delta,
        norm_clip=norm_clip,
        n_clients=n_clients,
        n_rounds=n_rounds,
        model_dim=model_dim,
        weight_matrix=weight_matrix,
        seed=seed,
    )


def get_aggregator(
        clients,
        test_clients,
        global_train_logger,
        global_test_logger,
        args
):
    """

    Parameters
    ----------

    :param clients:
    :param test_clients
    :param global_train_logger:
    :param global_test_logger:
    :param args:
    :return:

    """
    seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))

    if args.aggregator_type == "centralized":

        is_aggregator = True

        global_learner = \
            get_learner(
                name=args.experiment,
                model_name=args.model_name,
                device=args.device,
                optimizer_name=args.optimizer,
                scheduler_name=args.lr_scheduler,
                initial_lr=args.lr,
                n_rounds=args.n_rounds,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                input_dimension=args.input_dimension,
                hidden_dimension=args.hidden_dimension,
                dp_mechanism=args.dp_mechanism,
                l2_norm_clip=args.norm_clip,
                minibatch_size=args.bz,
                is_aggregator=is_aggregator,
                seed=seed
            )

        dp_noise_generator = \
            get_dp_noise_generator(
                dp_mechanism=args.dp_mechanism,
                epsilon=args.epsilon,
                delta=args.delta,
                norm_clip=args.norm_clip,
                n_clients=len(clients),
                n_rounds=args.n_rounds,
                model_dim=global_learner.model_dim,
                seed=seed
            )

        return CentralizedAggregator(
            clients=clients,
            test_clients=test_clients,
            global_learner=global_learner,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            dp_noise_generator=dp_noise_generator,
            log_freq=args.log_freq,
            verbose=args.verbose,
            seed=seed
        )

    elif args.aggregator_type == "decentralized":

        weight_matrix = \
            get_weight_matrix(
                n_nodes=len(clients),
                connectivity=args.connectivity,
                seed=seed
            )

        dp_noise_generator = \
            get_dp_noise_generator(
                dp_mechanism=args.dp_mechanism,
                epsilon=args.epsilon,
                delta=args.delta,
                norm_clip=args.norm_clip,
                n_clients=len(clients),
                n_rounds=args.n_rounds,
                model_dim=clients[0].learner.model_dim,
                weight_matrix=weight_matrix,
                seed=seed,
            )

        return DecentralizedAggregator(
            clients=clients,
            test_clients=test_clients,
            weight_matrix=weight_matrix,
            dp_noise_generator=dp_noise_generator,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            log_freq=args.log_freq,
            verbose=args.verbose,
            seed=seed
        )

    else:
        raise NotImplementedError(
            f"{args.aggregator_type} is not available!"
            f" Possible are: `centralized`, and `decentralized`."
        )
