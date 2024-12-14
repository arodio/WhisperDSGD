"""Simulate Federated Learning Training

This script allows to simulate federated learning; the experiment name, the method and  be precised along side with the
hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * train - simulate federated learning training

"""

from utils.utils import *
from utils.constants import *
from utils.args import TrainArgumentsManager

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, data_dir, logs_dir, chkpts_dir):
    """
    initialize clients from data folders

    :param args_:
    :param data_dir: path to directory containing data folders
    :param logs_dir: directory to save the logs
    :param chkpts_dir: directory to save chkpts
    :return: List[Client]

    """
    if chkpts_dir is not None:
        os.makedirs(chkpts_dir, exist_ok=True)

    if args_.verbose > 0:
        print("===> Building data loaders..")
    train_loaders, val_loaders, test_loaders = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=args_.validation
        )

    if args_.verbose > 0:
        print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_loader, val_loader, test_loader) in \
            enumerate(zip(train_loaders, val_loaders, test_loaders)):

        if train_loader is None or test_loader is None:
            continue

        # Print task targets with count
        task_target_counts = defaultdict(int)
        for _, targets, _ in train_loader:
            for target in targets:
                task_target_counts[target.item()] += 1
        if args_.verbose > 0:
            print(f"Task {task_id} has data {len(train_loader.dataset)}; targets: {dict(task_target_counts)}")

        learner =\
            get_learner(
                name=args_.experiment,
                model_name=args_.model_name,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                n_rounds=args_.n_rounds,
                steps_per_epoch=len(train_loaders[0]),
                seed=args_.seed,
                input_dimension=args_.input_dimension,
                hidden_dimension=args_.hidden_dimension,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                dp_mechanism=args_.dp_mechanism,
                l2_norm_clip=args_.norm_clip,
                minibatch_size=args_.bz,
                is_aggregator=False
            )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        save_path = os.path.join(chkpts_dir, "task_{}.pt".format(task_id)) if chkpts_dir else None
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            learner=learner,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            logger=logger,
            client_id=task_id,
            save_path=save_path
        )

        clients_.append(client)

    return clients_


def run_experiment(arguments_manager_):
    """

    :param arguments_manager_:
    :type arguments_manager_: ArgumentsManager

    """

    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()

    args_ = arguments_manager_.args

    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", arguments_manager_.args_to_string())

    if "chkpts_dir" in args_:
        chkpts_dir = args_.chkpts_dir
    else:
        chkpts_dir = None

    if args_.verbose > 0:
        print("==> Clients initialization..")

    clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "train"),
            logs_dir=os.path.join(logs_dir, "train"),
            chkpts_dir=os.path.join(chkpts_dir, "train") if chkpts_dir else None
        )

    if args_.verbose > 0:
        print("==> Test Clients initialization..")
    test_clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "test"),
            logs_dir=os.path.join(logs_dir, "test"),
            chkpts_dir=os.path.join(chkpts_dir, "test") if chkpts_dir else None
        )

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    aggregator = \
        get_aggregator(
            clients=clients,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            args=args_
        )

    aggregator.write_logs(counter=0)

    if args_.verbose > 0:
        print("Training..")
    for c_round in tqdm(range(1, args_.n_rounds+1)):

        if args_.verbose > 0:
            print(f"current round: {c_round} | {args_.aggregator_type}")
        aggregator.mix(c_round)

        if (c_round % args_.log_freq) == 0:
            if chkpts_dir is not None:
                aggregator.save_state(chkpts_dir, counter=c_round)
            aggregator.write_logs(counter=c_round)

    if chkpts_dir is not None:
        aggregator.save_state(chkpts_dir, counter=args_.n_rounds+1)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arguments_manager = TrainArgumentsManager()
    arguments_manager.parse_arguments()

    run_experiment(arguments_manager)
