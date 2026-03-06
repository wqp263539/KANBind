# KANBind — Full Reproducible Pipeline README

This README describes a **complete, end-to-end** workflow to reproduce the experiments:
1) Download the **two benchmark datasets** released by **Luo et al.** (BTD-Combo and HBTD)  
2) Generate **ProtT5** (ProtTrans) embedding features (`t5` vectors)  
3) Generate **PSSM** features (via PSI-BLAST) and convert them to fixed-length vectors  
4) Generate **NMBAC** features  
5) Run training + evaluation (`train_eval.py`) to output the **paper-consistent metrics**:  
**SN & SP & P\_{0.10} & FDR\_{0.10} & P\_{0.03} & FDR\_{0.03}**

> Notes:
> - ProtT5 feature extraction is **GPU-recommended**.
> - PSSM generation is **CPU-heavy** and depends on the BLAST database version.
> - Ensure you use the **same splits** as Luo et al. to match the benchmark protocol.

---

## 0. Repository Structure (recommended)

```text
KANBind/
  model.py
  train_eval.py
  README.md
  requirements.txt
  scripts/
    make_t5_features.py
    make_pssm_features.py
    pssm_to_vector.py
    make_nmbac_features.py
  data/
    BTD-Combo/
    HBTD/
  features/
    BTD-Combo/
    HBTD/

If your repository does not contain the scripts/ folder yet, you can still follow the same logic with your own feature-generation scripts; just ensure the outputs match the expected formats described below.

1. Environment Setup
1.1 Create a conda environment
conda create -n kanbind python=3.10 -y
conda activate kanbind
1.2 Install dependencies
pip install -r requirements.txt

Typical dependencies include:

torch

numpy, pandas

scikit-learn

transformers + sentencepiece (for ProtT5)

biopython (FASTA parsing)

BLAST+ tools (psiblast, makeblastdb) for PSSM

2. Step 1 — Download Luo et al. Datasets (BTD-Combo & HBTD)

Download the benchmark datasets introduced by Luo et al.:

BTD-Combo

HBTD

Place them under data/ and keep Luo’s original train/test splits.

A recommended layout is:

data/
  BTD-Combo/
    train_pos.fasta
    train_neg.fasta
    test_pos.fasta
    test_neg.fasta
  HBTD/
    train_pos.fasta
    train_neg.fasta
    test_pos.fasta
    test_neg.fasta

If Luo et al. provide additional metadata files, keep them together in the same dataset folder.

3. Step 2 — Generate ProtT5 (ProtTrans) Features

We encode each protein sequence into a fixed-length vector (1024-dim) using ProtT5.

3.1 Generate T5 vectors

Example commands (BTD-Combo):

python scripts/make_t5_features.py --fasta data/BTD-Combo/train_pos.fasta --out features/BTD-Combo/train_pos_t5.pkl
python scripts/make_t5_features.py --fasta data/BTD-Combo/train_neg.fasta --out features/BTD-Combo/train_neg_t5.pkl
python scripts/make_t5_features.py --fasta data/BTD-Combo/test_pos.fasta  --out features/BTD-Combo/test_pos_t5.pkl
python scripts/make_t5_features.py --fasta data/BTD-Combo/test_neg.fasta  --out features/BTD-Combo/test_neg_t5.pkl

Repeat the same for HBTD:

python scripts/make_t5_features.py --fasta data/HBTD/train_pos.fasta --out features/HBTD/train_pos_t5.pkl
python scripts/make_t5_features.py --fasta data/HBTD/train_neg.fasta --out features/HBTD/train_neg_t5.pkl
python scripts/make_t5_features.py --fasta data/HBTD/test_pos.fasta  --out features/HBTD/test_pos_t5.pkl
python scripts/make_t5_features.py --fasta data/HBTD/test_neg.fasta  --out features/HBTD/test_neg_t5.pkl
3.2 Output format requirements

The training pipeline expects .pkl storing vectors in a consistent order matching the FASTA input.

Each protein → one vector

Vector dimension: 1024

4. Step 3 — Generate PSSM Features

PSSM is computed using PSI-BLAST against a protein database (e.g., UniRef90 or NR).

4.1 Prepare a BLAST database

Download a FASTA database (example: UniRef90) and run:

makeblastdb -in uniref90.fasta -dbtype prot -out db/uniref90_db
4.2 Run PSI-BLAST to generate raw PSSM

Example (BTD-Combo train_pos):

python scripts/make_pssm_features.py \
  --fasta data/BTD-Combo/train_pos.fasta \
  --blast_db db/uniref90_db \
  --out_dir features/BTD-Combo/pssm_raw/train_pos \
  --num_iterations 3 \
  --evalue 0.001

Repeat for the other splits:

train_neg, test_pos, test_neg
and for HBTD too.

4.3 Convert raw PSSM to fixed-length PSSM vectors (e.g., 40-dim)

KANBind uses a fixed-length PSSM representation (commonly 40-dim).
Convert raw PSSM to a .csv file:

python scripts/pssm_to_vector.py \
  --pssm_dir features/BTD-Combo/pssm_raw/train_pos \
  --out_csv features/BTD-Combo/train_pos_pse_pssm.csv

Repeat for other splits and datasets:

python scripts/pssm_to_vector.py --pssm_dir features/BTD-Combo/pssm_raw/train_neg --out_csv features/BTD-Combo/train_neg_pse_pssm.csv
python scripts/pssm_to_vector.py --pssm_dir features/BTD-Combo/pssm_raw/test_pos  --out_csv features/BTD-Combo/test_pos_pse_pssm.csv
python scripts/pssm_to_vector.py --pssm_dir features/BTD-Combo/pssm_raw/test_neg  --out_csv features/BTD-Combo/test_neg_pse_pssm.csv

Expected output:

.csv with shape (N, 40)

row order must match FASTA order

5. Step 4 — Generate NMBAC Features

NMBAC features are fixed-length vectors (200-dim in this project).

Example (BTD-Combo):

python scripts/make_nmbac_features.py --fasta data/BTD-Combo/train_pos.fasta --out features/BTD-Combo/train_pos_NMBAC.txt
python scripts/make_nmbac_features.py --fasta data/BTD-Combo/train_neg.fasta --out features/BTD-Combo/train_neg_NMBAC.txt
python scripts/make_nmbac_features.py --fasta data/BTD-Combo/test_pos.fasta  --out features/BTD-Combo/test_pos_NMBAC.txt
python scripts/make_nmbac_features.py --fasta data/BTD-Combo/test_neg.fasta  --out features/BTD-Combo/test_neg_NMBAC.txt

Repeat for HBTD similarly.

Expected output:

.txt containing one 200-dim vector per protein (multiline format supported by our loader)

order must match FASTA order

6. Step 5 — Train and Evaluate
6.1 Prepare file paths

train_eval.py currently uses hard-coded file names (you can edit them).
You have two options:

Option A (recommended): edit train_eval.py and point to your generated files, e.g.:

train_files = {
  "t5_p_path": "features/BTD-Combo/train_pos_t5.pkl",
  "t5_n_path": "features/BTD-Combo/train_neg_t5.pkl",
  "pssm_p_path": "features/BTD-Combo/train_pos_pse_pssm.csv",
  "pssm_n_path": "features/BTD-Combo/train_neg_pse_pssm.csv",
  "nmbac_p_path": "features/BTD-Combo/train_pos_NMBAC.txt",
  "nmbac_n_path": "features/BTD-Combo/train_neg_NMBAC.txt",
}
test_files = {
  "t5_p_path": "features/BTD-Combo/test_pos_t5.pkl",
  "t5_n_path": "features/BTD-Combo/test_neg_t5.pkl",
  "pssm_p_path": "features/BTD-Combo/test_pos_pse_pssm.csv",
  "pssm_n_path": "features/BTD-Combo/test_neg_pse_pssm.csv",
  "nmbac_p_path": "features/BTD-Combo/test_pos_NMBAC.txt",
  "nmbac_n_path": "features/BTD-Combo/test_neg_NMBAC.txt",
}

Option B: rename your feature files to match the default names used inside train_eval.py.

6.2 Run training
python train_eval.py
6.3 Output metrics (paper-consistent)

The script prints metrics in the same format as the paper:

SN

SP

P_{0.10}, FDR_{0.10}

P_{0.03}, FDR_{0.03}

It also writes:

cv_paper_metrics_summary.csv
7. Important Notes (to match the paper)

Sequence order consistency is critical
All generated features (T5, PSSM, NMBAC) must correspond to the same protein order as in the FASTA file.

PSSM reproducibility depends on the database
Different BLAST database versions can slightly change PSSM features.

Threshold and prevalence settings
By default, train_eval.py computes paper metrics at:

threshold = 0.5

prevalence φ = 0.10 and 0.03
These match the table fields:
SN & SP & P_{0.10} & FDR_{0.10} & P_{0.03} & FDR_{0.03}

8. Citation

If you use this pipeline, please cite:

Luo et al. (for the BTD-Combo and HBTD benchmark datasets)

ProtTrans / ProtT5 (for protein language model embeddings)

KAN / efficient-kan (for the Kolmogorov–Arnold Network classifier head)
