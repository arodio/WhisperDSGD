# _Whisper D-SGD_: Correlated Noise Across Agents for Differentially Private Decentralized Learning

This repository is the official implementation for  
[_Whisper D-SGD_: Correlated Noise Across Agents for Differentially Private Decentralized Learning](https://arxiv.org/abs/2501.14644).

Decentralized learning enables distributed agents to train a shared machine learning model through local computation and peer-to-peer communication. Although each agent retains its dataset locally, the communication of local models can still expose private information to adversaries. To mitigate these threats, local differential privacy (LDP) injects independent noise per agent, but it suffers a larger utility gap than central differential privacy (CDP).

We introduce **_Whisper D-SGD_**, a novel covariance-based approach that generates correlated privacy noise across agents, unifying several state-of-the-art methods as special cases. By leveraging network topology and mixing weights, _Whisper D-SGD_ optimizes the noise covariance to achieve network-wide noise cancellation. Experimental results show that _Whisper D-SGD_ cancels more noise than existing pairwise-correlation schemes, substantially narrowing the CDP-LDP gap and improving model performance under the same privacy guarantees.

---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```
---

## Overview 



### Code Structure

| File/Module                    | Content/Responsibility                                                                                                                             |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `train.py`                     | Main file to simulate both federated (centralized) and decentralized learning, as well as differentially private variants. Saves logs and checkpoints. |
| `aggregator.py`                | Contains aggregator classes that orchestrate the aggregation step:                                                                                 |
|                                | - `NoCommunicationAggregator`: No collaboration                                                                                                    |
|                                | - `CentralizedAggregator`: FedAvg / CDP                                                                                                            |
|                                | - `DecentralizedAggregator`: D-SGD / LDP / DECOR / Whisper-DSGD                                                                                    |
| `client.py`                    | Defines the `Client` class, handling local training step (`step()`), data loaders, logging                                                         |
| `diff_privacy.py`              | `DPNoiseGenerator` (`cdp`, `ldp`, DECOR: `pairwise`, Whisper D-SGD: `mixing`)                                                                      |
| `models.py`                    | Contains linear and NN models (e.g., `LinearLayer`, `TitanicNN`, `MnistCNN`)                                                                       |
| `utils/`                       | Helper modules:                                                                                                                                    |
|                                | - `utils.utils.py`: Instantiates learners, clients, aggregators, data loaders, etc                                                                 |
|                                | - `utils.graph.py`: Creates and manages Erdős–Rényi graph structures and weight matrices                                                         |
|                                | - `utils.optim.py`: Custom DP optimizer wrapper for gradient clipping and noise addition                                                           |
| `data/<dataset>/generate_data.py` | Scripts to generate local data partitions for each client (e.g., `data/a9a/generate_data.py`)                                                      |
| `paper_experiments/`           | Scripts to replicate paper experiments for each dataset (`a9a`, `titanic`, `mnist`)                                                                |

---

### Algorithms

By setting the aggregator and DP mechanism, you can replicate different standard or private decentralized/federated algorithms:

| Algorithm            | `aggregator_type` | `dp_mechanism` | Reference                                                                  |
|-----------------------|-------------------|----------------|----------------------------------------------------------------------------|
| FedAvg               | centralized       | None           | [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)                   |
| D-SGD (non-private)  | decentralized     | None           | [Koloskova et al., 2020](https://arxiv.org/abs/2003.10422)                 |
| Central DP (CDP)     | centralized       | cdp            | [Dwork et al., 2006](https://link.springer.com/chapter/10.1007/11787006_1) |
| Local DP (LDP)       | decentralized     | ldp            | [Kasiviswanathan et al., 2011](https://arxiv.org/abs/0803.0924)            |
| DECOR                | decentralized     | pairwise       | [Allouah et al., 2024](https://arxiv.org/abs/2405.01031)                   |
| Whisper D-SGD (ours) | decentralized     | mixing         | This repository / our paper                                                |

Use `--connectivity p` to define the Erdős–Rényi connectivity for decentralized topologies.

Use `--epsilon e`, `--delta d`, `--norm_clip c` to define DP parameters and norm clipping threshold.

---

### Datasets and Models

| Dataset                                                                           | Task                 | Model                                                                    |
|-----------------------------------------------------------------------------------|----------------------|--------------------------------------------------------------------------|
| [Titanic](https://www.kaggle.com/c/titanic/data)                                  | Binary classification| `LinearLayer(input_dimension=9, num_classes=1)`                          |
| [a9a LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) | Binary classification| `LinearLayer(input_dimension=123, num_classes=1)`                        |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                                                                         | Image classification | `MnistCNN` (two convolutional layers + fully connected layers) |

Scripts for generating partitions are under `data/<dataset>/generate_data.py`.  
Use `--n_tasks <num_clients>` to specify how many clients to create.

---

### Paper Experiments

Below are common steps and references for running experiments. 

We also provide shell scripts in `paper_experiments/<dataset>/` to reproduce our main paper results.

#### 1. Generate Data

Before training, prepare the partitions for your chosen dataset.  
For example, for `a9a`:

```bash
cd data/a9a
rm -rf all_data
python generate_data.py \
    --n_tasks 20 \
    --by_labels_split \
    --n_components -1 \
    --alpha 10 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345
cd ../..
```

#### 2. Run Training
Use `train.py` with the appropriate arguments. 

Example: run decentralized learning (`aggregator_type=decentralized`) with local DP (`dp_mechanism=ldp`),  
𝜖 = 10, connectivity 𝑝 = 1.0 (fully connected graph), and log the results to `logs/a9a_example/ldp/epsilon10/connectivity1.0/seed_12345`.

```bash
python train.py \
    a9a \
    --n_rounds 100 \
    --aggregator_type decentralized \
    --dp_mechanism ldp \
    --epsilon 10 \
    --norm_clip 0.1 \
    --connectivity 1.0 \
    --bz 128 \
    --lr 0.01 \
    --log_freq 1 \
    --device cuda \
    --optimizer sgd \
    --logs_dir logs/a9a_example/ldp/epsilon10/connectivity1.0/seed_12345 \
    --seed 12345 \
    --verbose 1
```

#### 3. Reproducible Experiments
To reproduce the full set of experiments, check the `paper_experiments/` directory. 

| Folder         | Files                                                                                                      | Description                                           |
|----------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `titanic`      | `titanic.sh`                                                                                              | Example script for the Titanic dataset                |
| `a9a_libsvm`   | `a9a_libsvm.sh`, `n20e10p.sh`, `n20e10p_lr.sh`, `n20p5e.sh`, `n20p5e_lr.sh`                                | Sweep parameters `ϵ`, `p`, `lr`, and `seed` |
| `mnist`        | `mnist.sh`, `n20e10p.sh`, `n20e10p_lr.sh`, `n20p5e.sh`, `n20p5e_lr.sh`                                     | Sweep parameters `ϵ`, `p`, `lr`, and `seed`      |


---

### Contributing

Pull requests and issues are welcome. If you find a bug or have a feature request, please open an issue.


---

### Citation

If you use our code or method in your work, please cite our paper:

```commandline
@misc{rodioWhisperDSGDCorrelated2025,
  title = {Whisper {{D-SGD}}: {{Correlated Noise Across Agents}} for {{Differentially Private Decentralized Learning}}},
  shorttitle = {Whisper {{D-SGD}}},
  author = {Rodio, Angelo and Chen, Zheng and Larsson, Erik G.},
  year = {2025},
  month = jan,
  number = {arXiv:2501.14644},
  eprint = {2501.14644},
  primaryclass = {cs},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2501.14644}
}
```

