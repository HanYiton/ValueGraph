# ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation

Yitong Han¹, Wei Gao¹, Yi Zhao¹, Prasanta Bhattacharya², Fengzhu Zeng¹, Mohammad Amanlou¹

¹ [Affiliation] · ² [Affiliation]

This is the repository of **ValueGraph**, a two-stage graph pre-training framework for
social-media user representation learning.

![Framework](assets/framework.png)

- ValueGraph first pre-trains a [GraphMAE2](https://arxiv.org/abs/2304.04779)-style masked
  autoencoder on a Twitter/X + Reddit post-reply graph, then fine-tunes the encoder with
  user-level contrastive and clustering objectives so that posts of the same user are pulled
  together while similar and dissimilar users are separated. The learned embeddings are evaluated
  on bot detection (TwiBot-22) and stance detection (MT-CSD / RumourEval).

## Requirements

Our pre-training and user-representation code is developed based on the
[GraphMAE2](https://github.com/THUDM/GraphMAE2) codebase. Node features are produced with
[ModernBERT](https://github.com/AnswerDotAI/ModernBERT).

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Because DGL and `torch-scatter` must match your CUDA version, install them explicitly, e.g.:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install dglteam/label/th24_cu124::dgl
conda install conda-forge::transformers
pip install torch-scatter==2.1.2 torch-geometric==2.6.1
```

The downstream baselines use additional third-party code: [TwiBot-22](https://github.com/LuoUndergradX/TwiBot-22),
[MT-CSD](https://arxiv.org/abs/2403.11145), and the RumourEval 2019 baseline. See their own READMEs
under `downstreamtask/`.

## Data Preparation

Each social dataset (`twitter`, `reddit`) is stored as:

```
<data_dir>/<dataset>/
├── source.parquet   # post table (id, parent_id, text, split, ...)
└── feat.npy         # pre-computed node features (text embeddings)
```

1. Generate text features with ModernBERT:

   ```bash
   python gen_modernbert_embedding/genbymodernert.py
   ```

2. Place `feat.npy` and the corresponding `source.parquet` under `<data_dir>/<dataset>/`.

3. For Stage 2, prepare three JSON dictionaries: `user_map` (`{user_id: [post_ids]}`),
   `similar_users_dict`, and `dissimilar_users_dict`.

## Training & Evaluation

### Stage 1: pre-training

```bash
python GraphMae2/main_large.py \
  --dataset social --encoder gat --decoder gat \
  --config GraphMae2/configs/social.yaml \
  --data_dir /path/to/pretrain_data
```

### Stage 2: user representation

```bash
python GraphMae2/user_embedding.py
```

The script samples seed users, expands them with similar/dissimilar users, and trains with the
contrastive and clustering losses in `GraphMae2/losses.py`.

### Inference and fine-tuning

```bash
python GraphMae2/infer.py               # extract node embeddings
python GraphMae2/gen_userembedding.py   # batched embedding generation
python GraphMae2/main_finetune.py --dataset rumours_eval --pretrain_path <ckpt>
```

### Downstream tasks

```bash
# Bot detection (TwiBot-22)
python downstreamtask/botdetection/twibot_22/train.py
python downstreamtask/botdetection/GCN_GAT/train.py

# Stance detection
python downstreamtask/stancedetect/MT_CSD/train.py

# Significance testing
python p-value/train.py
```

## Citation

If you use this code in your research, please cite our paper.

```bibtex
@inproceedings{han2027valuegraph,
  title     = {ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation},
  author    = {Yitong Han and Wei Gao and Yi Zhao and Prasanta Bhattacharya and Fengzhu Zeng and Mohammad Amanlou},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027}
}
```

## Contact for issues

- Yitong Han, [email]
