import string

ERROR = 1e-10

ALL_STRATEGIES = {
    "random"
}

ALL_MODELS = {
    "mobilenet"
}

LOADER_TYPE = {
    "synthetic": "tabular",
    "titanic": "titanic",
    "a9a": "a9a",
    "mnist": "mnist",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "femnist": "femnist",
    "shakespeare": "shakespeare"
}

EXTENSIONS = {
    "tabular": ".pkl",
    "titanic": ".pkl",
    "a9a": ".pkl",
    "mnist": ".pkl",
    "cifar10": ".pkl",
    "cifar100": ".pkl",
    "femnist": ".pt",
    "shakespeare": ".txt"
}

N_CLASSES = {
    "synthetic": 1,
    "titanic": 1,
    "a9a": 1,
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "femnist": 62,
    "shakespeare": 100
}

SHAKESPEARE_CONFIG = {
    "input_size": len(string.printable),
    "embed_size": 8,
    "hidden_size": 256,
    "output_size": len(string.printable),
    "n_layers": 2,
    "chunk_len": 80
}

CHARACTERS_WEIGHTS = {
    '\n': 0.43795308843799086,
    ' ': 0.042500849608091536,
    ',': 0.6559597911540539,
    '.': 0.6987226398690805,
    'I': 0.9777491725556848,
    'a': 0.2226022051965085,
    'c': 0.813311655455682,
    'd': 0.4071860494572223,
    'e': 0.13455606165058104,
    'f': 0.7908671114133974,
    'g': 0.9532922255751889,
    'h': 0.2496906467588955,
    'i': 0.27444893060347214,
    'l': 0.37296488139109546,
    'm': 0.569937324017103,
    'n': 0.2520734570378263,
    'o': 0.1934141300462555,
    'r': 0.26035705948768273,
    's': 0.2534775933879391,
    't': 0.1876471355731429,
    'u': 0.47430062920373184,
    'w': 0.7470615815733715,
    'y': 0.6388302610200002
}

LOCAL_HEAD_UPDATES = 10  # number of epochs for local heads used in FedRep
