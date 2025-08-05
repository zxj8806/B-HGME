# Bayesian Hyperspherical Graph Mixture-of-Experts Deciphers Cell–Cell Interaction in Spatial Transcriptomics

This repository contains the official implementation of the paper  
**“Bayesian Hyperspherical Graph Mixture-of-Experts Deciphers Cell–Cell Interaction in Spatial Transcriptomics.”**

---

## Scope (Under Review)

This is a **scoped, minimal release** to run demos.  

## Computing Infrastructure <a name="infrastructure"></a>

The experiments reported in the paper were run on the following stack.  

| Component | Specification |
|---|---|
| **CPU** | 2 × Intel Xeon Gold 6430 (32 cores each, 2.8 GHz) |
| **GPU** | 2 × NVIDIA A100 80 GB (PCIe, 700 W cap) |
| **System Memory** | 512 GB DDR4-3200 |
| **Storage** | 2 TB NVMe SSD (Samsung PM9A3) |
| **Operating System** | Ubuntu 22.04.4 LTS, Linux 5.15 |
| **CUDA Driver** | 12.1 |
| **cuDNN** | 9.0 |
| **Python Environment** | Conda 23.7 |
| **Other Libraries** | GCC 11.4, CMake 3.29, OpenMPI 4.1 |

---

## Getting Started

### 1) Create a Conda environment (Python 3.8)
```bash
conda create --name BHGME python=3.8
conda activate BHGME
````

### 2) Install non-PyTorch dependencies

```bash
pip install -r requirements.txt
```

### 3) Pin NumPy version

Ensure `numpy==1.24.1` is installed (reinstall if needed):

```bash
pip install --upgrade --force-reinstall numpy==1.24.1
```

### 4) Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### 5) Install PyTorch Geometric and extensions

```bash
pip install torch-geometric==2.6.1
pip install torch-cluster==1.6.3+pt24cu121 \
            torch-scatter==2.1.2+pt24cu121 \
            torch-sparse==0.6.18+pt24cu121 \
            torch-spline-conv==1.2.2+pt24cu121
```

---

## Run

Change to the source directory:

```bash
cd src
```

### Demo 1: `seqFISH`, `scMultiSim`, or `MERFISH`

```bash
python3 main.py -m train -o ../out/seqFISH/ -s B-HGME -t 0.3
```

### Demo 2: V1 dataset (`.h5ad`)

```bash
python3 BHGME4h5ad.py \
  --adata_file ../data/V1/raw/adata.h5ad \
  --output_dir ./result/V1 \
  --n_clusters 7 --adj_weight 0.3 --k 25 \
  --clu_model kmeans --n_experts 100
```

## Notes

* This is an initial, scoped release. We welcome reports of inefficiencies or minor issues.
* Features not included here are planned to be released after acceptance.

---

