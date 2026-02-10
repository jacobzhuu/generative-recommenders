# Generative Recommenders

Repository hosting code for ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations`` ([ICML'24 paper](https://proceedings.mlr.press/v235/zhai24a.html)) and related code, where we demonstrate that the ubiquitously used classical deep learning recommendation paradigm (DLRMs) can be reformulated as a generative modeling problem (Generative Recommenders or GRs) to overcome known compute scaling bottlenecks, propose efficient algorithms such as HSTU and M-FALCON to accelerate training and inference for large-scale sequential models by 10x-1000x, and demonstrate scaling law for the first-time in deployed, billion-user scale recommendation systems.

## Getting started

We recommend using `requirements.txt`. This has been tested with Ubuntu 22.04, CUDA 12.4, and Python 3.10.

```bash
pip3 install -r requirements.txt
```

Alternatively, you can manually install PyTorch based on official instructions. Then,

```bash
pip3 install gin-config pandas fbgemm_gpu torchrec tensorboard
```

## Experiments

### Public Experiments

To reproduce the public experiments in our paper (traditional sequential recommender setting, Section 4.1.1) on MovieLens and Amazon Reviews in the paper, please follow these steps:

#### Download and preprocess data.

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

A GPU with 24GB or more HBM should work for most datasets.

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
```

Other configurations are included in configs/ml-1m, configs/ml-20m, and configs/amzn-books to make reproducing these experiments easier.

#### Verify results.

By default we write experimental logs to exps/. We can launch tensorboard with something like the following:

```bash
tensorboard --logdir ~/generative-recommenders/exps/ml-1m-l200/ --port 24001 --bind_all
tensorboard --logdir ~/generative-recommenders/exps/ml-20m-l200/ --port 24001 --bind_all
tensorboard --logdir ~/generative-recommenders/exps/amzn-books-l50/ --port 24001 --bind_all
```

With the provided configuration (.gin) files, you should be able to reproduce the following results (verified as of 04/15/2024):

**MovieLens-1M (ML-1M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------| --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2853           | 0.1603          | 0.5474          | 0.2185          | 0.7528          | 0.2498          |
| BERT4Rec      | 0.2843 (-0.4%)   | 0.1537 (-4.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2811 (-1.5%)   | 0.1648 (+2.8%)  |                 |                 |                 |                 |
| HSTU          | 0.3097 (+8.6%)   | 0.1720 (+7.3%)  | 0.5754 (+5.1%)  | 0.2307 (+5.6%)  | 0.7716 (+2.5%)  | 0.2606 (+4.3%)  |
| HSTU-large    | **0.3294 (+15.5%)**  | **0.1893 (+18.1%)** | **0.5935 (+8.4%)**  | **0.2481 (+13.5%)** | **0.7839 (+4.1%)**  | **0.2771 (+10.9%)** |

**MovieLens-20M (ML-20M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2889           | 0.1621          | 0.5503          | 0.2199          | 0.7661          | 0.2527          |
| BERT4Rec      | 0.2816 (-2.5%)   | 0.1703 (+5.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2813 (-2.6%)   | 0.1730 (+6.7%)  |                 |                 |                 |                 |
| HSTU          | 0.3273 (+13.3%)  | 0.1895 (+16.9%) | 0.5889 (+7.0%)  | 0.2473 (+12.5%) | 0.7952 (+3.8%)  | 0.2787 (+10.3%) |
| HSTU-large    | **0.3556 (+23.1%)**  | **0.2098 (+29.4%)** | **0.6143 (+11.6%)** | **0.2671 (+21.5%)** | **0.8074 (+5.4%)**  | **0.2965 (+17.4%)** |

**Amazon Reviews (Books)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------|---------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.0306           | 0.0164          | 0.0754          | 0.0260          | 0.1431          | 0.0362          |
| HSTU          | 0.0416 (+36.4%)  | 0.0227 (+39.3%) | 0.0957 (+27.1%) | 0.0344 (+32.3%) | 0.1735 (+21.3%) | 0.0461 (+27.7%) |
| HSTU-large    | **0.0478 (+56.7%)**  | **0.0262 (+60.7%)** | **0.1082 (+43.7%)** | **0.0393 (+51.2%)** | **0.1908 (+33.4%)** | **0.0517 (+43.2%)** |

for all three tables above, the ``SASRec`` rows are based on [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) but with the original binary cross entropy loss
replaced with sampled softmax losses proposed in [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039). These rows are reproducible with ``configs/*/sasrec-*-final.gin``.
The ``BERT4Rec`` and ``GRU4Rec`` rows are based on results reported by [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602) -
note that the comparison slightly favors these two, due to them using full negatives whereas the other rows used 128/512 sampled negatives. The ``HSTU`` and ``HSTU-large`` rows are based on [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152); in particular, HSTU rows utilize identical configurations as SASRec. ``HSTU`` and ``HSTU-large`` results can be reproduced with ``configs/*/hstu-*-final.gin``.

### Synthetic Dataset / MovieLens-3B

We support generating synthetic dataset with fractal expansion introduced in https://arxiv.org/abs/1901.08910. This allows us to expand the current 20 million real-world ratings in ML-20M to 3 billion.

To download the pre-generated synthetic dataset:

```bash
pip3 install gdown
mkdir -p tmp/ && cd tmp/
gdown https://drive.google.com/uc?id=1-jZ6k0el7e7PyFnwqMLfqUTRh_Qdumt-
unzip ml-3b.zip && rm ml-3b.zip
```

To generate the synthetic dataset on your own:

```bash
python3 run_fractal_expansion.py --input-csv-file tmp/ml-20m/ratings.csv --write-dataset True --output-prefix tmp/ml-3b/
```

### Efficiency experiments

``ops/triton`` contains triton kernels needed for efficiency experiments. ``ops/cpp`` contains efficient CUDA kernels. In particular, ``ops/cpp/hstu_attention`` contains the attention implementation based on [FlashAttention V3](https://github.com/Dao-AILab/flash-attention) with state-of-the-art efficiency on H100 GPUs.

## DLRM-v3

We have created a DLRM model using HSTU and have developed benchmarks for both training and inference to faciliate production RecSys use cases.

#### Run model training with 4 GPUs

```bash
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/train/train_ranker.py --dataset debug --mode train
```

#### Run model inference with 4 GPUs

```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python -m pip install .

LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/inference/main.py --dataset debug
```

## License
This codebase is Apache 2.0 licensed, as found in the [LICENSE](LICENSE) file.

## Contributors
The overall project is made possible thanks to the joint work from many technical contributors (listed in alphabetical order):

Adnan Akhundov, Bugra Akyildiz, Shabab Ayub, Alex Bao, Renqin Cai, Jennifer Cao, Xuan Cao, Guoqiang Jerry Chen, Lei Chen, Li Chen, Sean Chen, Xianjie Chen, Huihui Cheng, Weiwei Chu, Ted Cui, Shiyan Deng, Nimit Desai, Fei Ding, Shilin Ding, Francois Fagan, Lu Fang, Leon Gao, Zhaojie Gong, Fangda Gu, Liang Guo, Liz Guo, Jeevan Gyawali, Yuchen Hao, Daisy Shi He, Michael Jiayuan He, Yu He, Samuel Hsia, Jie Hua, Yanzun Huang, Hongyi Jia, Rui Jian, Jian Jin, Rafay Khurram, Rahul Kindi, Changkyu Kim, Yejin Lee, Fu Li, Han Li, Hong Li, Shen Li, Rui Li, Wei Li, Zhijing Li, Lucy Liao, Xueting Liao, Emma Lin, Hao Lin, Chloe Liu, Jingzhou Liu, Xing Liu, Xingyu Liu, Kai Londenberg, Yinghai Lu, Liang Luo, Linjian Ma, Matt Ma, Yun Mao, Bert Maher, Ajit Mathews, Matthew Murphy, Satish Nadathur, Min Ni, Jongsoo Park, Colin Peppler, Jing Qian, Lijing Qin, Jing Shan, Alex Singh, Timothy Shi,  Yu Shi, Dennis van der Staay, Xiao Sun, Colin Taylor, Shin-Yeh Tsai, Rohan Varma, Omkar Vichare, Alyssa Wang, Pengchao Wang, Shengzhi Wang, Wenting Wang, Xiaolong Wang, Yueming Wang, Zhiyong Wang, Wei Wei, Bin Wen, Carole-Jean Wu, Yanhong Wu, Eric Xu, Bi Xue, Hong Yan, Zheng Yan, Chao Yang, Junjie Yang, Wen-Yun Yang, Ze Yang, Zimeng Yang, Yuanjun Yao, Chunxing Yin, Daniel Yin, Yiling You, Jiaqi Zhai, Keke Zhai, Yanli Zhao, Zhuoran Zhao, Hui Zhang, Jingjing Zhang, Lu Zhang, Lujia Zhang, Na Zhang, Rui Zhang, Xiong Zhang, Ying Zhang, Zhiyun Zhang, Charles Zheng, Erheng Zhong, Zhao Zhu, Xin Zhuang.

For the initial paper describing the Generative Recommender problem formulation and the algorithms used, including HSTU and M-FALCON, please refer to ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations``([ICML'24 paper](https://dl.acm.org/doi/10.5555/3692070.3694484), [slides](https://icml.cc/media/icml-2024/Slides/32684.pdf)).

## Yelp Experiment Retrospective (Run 1 vs Run 2)

This section documents two full Yelp experiment cycles in this repository, including code changes and result comparisons, to support reproducibility and auditing.

### 1) Runtime Environment and Data

- Environment: `conda` environment `gr26`
- Key dependencies: `torch`, `fbgemm_gpu`, `gin-config`
- Project directory: `/home/linjx/code/generative-recommenders`
- Yelp raw data directory: `/home/linjx/code/SELFRec-main/dataset/yelp`
- Config file: `configs/yelp/hstu-sampled-softmax-n512-final.gin`

### 2) First Model Run (Baseline) Details

#### 2.1 Training and evaluation logic at that time

- Preprocessing generated sequence files for `train/valid/test`, but without strict chronological filtering.
- During Yelp training, the default evaluation split was `test` (i.e., test performance was monitored during training).
- There was no best-checkpoint selection or early-stopping control based on `selection_score`.

#### 2.2 Typical command flow (Run 1)

```bash
conda activate gr26
python preprocess_yelp_data.py
python main.py --gin_config_file configs/yelp/hstu-sampled-softmax-n512-final.gin --master_port=12345
```

After training, the last checkpoint (for example `*_ep200`) was typically used for evaluation:

```bash
python evaluate_checkpoint.py \
  --gin_config_file configs/yelp/hstu-sampled-softmax-n512-final.gin \
  --checkpoint_path ckpts/yelp-l50/<YOUR_CKPT_EP200> \
  --device cuda
```

### 3) Code Changes in Run 2 (Relative to Run 1)

#### 3.1 Strict chronological filtering for evaluation preprocessing (key change)

- File: `generative_recommenders/research/data/preprocessor.py`
- Change: added strict chronological filtering so that a sample enters `valid/test` only if `target_ts > max(train_history_ts)`.
- Also added filtering statistics (kept/removed counts) in preprocessing logs.

Example logs after rerunning preprocessing:

- `train/valid/test = 166620/3540/3601`
- `valid/test = 55479/55436` before filtering

#### 3.2 Use `valid` for model selection during training, and `test` only for final reporting

- File: `generative_recommenders/research/data/reco_dataset.py`
- Change: added `eval_split` to `get_reco_dataset(...)` to support switching between `valid/test`.

- File: `generative_recommenders/research/trainer/train.py`
- Change: added `eval_split` to `train_fn(...)` and propagated it into dataset construction.

- File: `configs/yelp/hstu-sampled-softmax-n512-final.gin`
- Change: set `train_fn.eval_split = "valid"`.

#### 3.3 Added best-checkpoint selection and early stopping

- File: `generative_recommenders/research/trainer/train.py`
- Changes:
  - Added `model_selection_weights`
  - Added `save_best_checkpoint`
  - Added `early_stop_patience / early_stop_min_epochs / early_stop_min_delta / early_stop_full_eval_only`
  - Added `selection_score`, `NDCG@20`, and `HR@20` to logs

- Current Yelp config:
  - `train_fn.save_best_checkpoint = True`
  - `train_fn.model_selection_weights = {'ndcg@10': 1.0, 'ndcg@20': 2.0, 'hr@10': 1.0, 'hr@20': 2.0}`
  - `train_fn.early_stop_patience = 6`
  - `train_fn.early_stop_min_epochs = 10`
  - `train_fn.early_stop_min_delta = 1e-4`
  - `train_fn.early_stop_full_eval_only = True`

#### 3.4 Evaluation and logging persistence enhancements

- File: `evaluate_checkpoint.py`
- Changes:
  - Supports `--eval_split {valid,test}`
  - Supports `--metrics_out` and `--key_metrics_out`
  - Adds pre-run checks for CUDA/fbgemm
  - Outputs key metrics including `recall@10/20` and `ndcg@10/20`

- File: `run_yelp_train_eval.sh` (new)
- Changes:
  - One-command workflow for preprocessing (as needed) + training + final evaluation
  - With `RUN_PREPROCESS=auto`, automatically checks whether current cache is strict-chrono and reruns preprocessing if cache is stale
  - Final evaluation always uses `test`, with logs and metrics persisted to disk

### 4) Second Model Run (Current Recommended Workflow)

```bash
cd /home/linjx/code/generative-recommenders
conda activate gr26
RUN_PREPROCESS=auto bash run_yelp_train_eval.sh
```

Key artifacts are written to `logs/yelp/`:

- `preprocess_<ts>.log`
- `train_<ts>.log`
- `eval_<ts>.log`
- `metrics_<ts>.json`
- `metrics_<ts>.txt`
- `run_<ts>.summary`

### 5) Result Comparison Between the Two Runs

Definitions:

- `Δ = Run 2 - Run 1`
- `Relative Δ = Δ / Run 1`

| Metric | Run 1 | Run 2 | Δ | Relative Δ |
| --- | ---: | ---: | ---: | ---: |
| HR@10 | 0.034869 | 0.046654 | +0.011785 | +33.8% |
| NDCG@10 | 0.016894 | 0.023001 | +0.006107 | +36.1% |
| HR@20 | 0.063298 | 0.072202 | +0.008904 | +14.1% |
| NDCG@20 | 0.024014 | 0.029455 | +0.005441 | +22.7% |
| HR@50 | 0.129032 | 0.138017 | +0.008985 | +7.0% |
| NDCG@50 | 0.036896 | 0.042344 | +0.005448 | +14.8% |
| HR@100 | 0.212768 | 0.234102 | +0.021334 | +10.0% |
| NDCG@100 | 0.050411 | 0.057906 | +0.007495 | +14.9% |
| HR@200 | 0.339887 | 0.373785 | +0.033898 | +10.0% |
| NDCG@200 | 0.068124 | 0.077312 | +0.009188 | +13.5% |
| HR@500 | 0.581644 | 0.608442 | +0.026798 | +4.6% |
| HR@1000 | 0.766163 | 0.785615 | +0.019452 | +2.5% |
| MRR | 0.018700 | 0.023128 | +0.004428 | +23.7% |
| HR@10_>=4 | 0.036125 | 0.048111 | +0.011986 | +33.2% |
| NDCG@10_>=4 | 0.017439 | 0.023263 | +0.005824 | +33.4% |
| HR@50_>=4 | 0.132573 | 0.142073 | +0.009500 | +7.2% |
| MRR_>=4 | 0.019148 | 0.023080 | +0.003932 | +20.5% |

### 6) Conclusion

- After the Run 2 changes, the core target metrics `HR@10/20` and `NDCG@10/20` improved clearly.
- The training/evaluation protocol is now more rigorous: train-time model selection on `valid`, final reporting only on `test`.
- Logs, checkpoints, and key metrics are persistently saved, which improves future reproducibility and regression analysis.
