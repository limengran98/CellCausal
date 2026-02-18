# CellCausal

CellCausal is an autonomous AI agent framework designed for Virtual Cell Modeling (VCM).


## 🛠️ Installation

🎉 欢迎使用 AI 工作站，limengran！
💡 常用命令: workspace, myenvs, shared, sharedcode
📊 查看空间: diskusage | myspace
🎮 GPU状态: gpu | gpuwatch
⚠️ 请在/data/your_user_name下存放代码，大数据和模型，不要在/home空间！
Channels:
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
 - defaults
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
Platform: linux-64
Collecting package metadata (repodata.json): - \ | / - \ | done
Solving environment: - \ done

## Package Plan ##

  environment location: /data/conda-envs/limengran/CellCausal

  added / updated specs:
    - python=3.11.14


The following NEW packages will be INSTALLED:

  _libgcc_mutex      anaconda/cloud/conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge 
  _openmp_mutex      anaconda/cloud/conda-forge/linux-64::_openmp_mutex-4.5-2_gnu 
  bzip2              anaconda/cloud/conda-forge/linux-64::bzip2-1.0.8-hda65f42_8 
  ca-certificates    anaconda/cloud/conda-forge/noarch::ca-certificates-2026.1.4-hbd8a1cb_0 
  icu                anaconda/cloud/conda-forge/linux-64::icu-78.2-h33c6efd_0 
  ld_impl_linux-64   anaconda/cloud/conda-forge/linux-64::ld_impl_linux-64-2.45.1-default_hbd61a6d_101 
  libexpat           anaconda/cloud/conda-forge/linux-64::libexpat-2.7.3-hecca717_0 
  libffi             anaconda/cloud/conda-forge/linux-64::libffi-3.5.2-h3435931_0 
  libgcc             anaconda/cloud/conda-forge/linux-64::libgcc-15.2.0-he0feb66_17 
  libgcc-ng          anaconda/cloud/conda-forge/linux-64::libgcc-ng-15.2.0-h69a702a_17 
  libgomp            anaconda/cloud/conda-forge/linux-64::libgomp-15.2.0-he0feb66_17 
  liblzma            anaconda/cloud/conda-forge/linux-64::liblzma-5.8.2-hb03c661_0 
  libnsl             anaconda/cloud/conda-forge/linux-64::libnsl-2.0.1-hb9d3cd8_1 
  libsqlite          anaconda/cloud/conda-forge/linux-64::libsqlite-3.51.2-hf4e2dac_0 
  libstdcxx          anaconda/cloud/conda-forge/linux-64::libstdcxx-15.2.0-h934c35e_17 
  libuuid            anaconda/cloud/conda-forge/linux-64::libuuid-2.41.3-h5347b49_0 
  libxcrypt          anaconda/cloud/conda-forge/linux-64::libxcrypt-4.4.36-hd590300_1 
  libzlib            anaconda/cloud/conda-forge/linux-64::libzlib-1.3.1-hb9d3cd8_2 
  ncurses            anaconda/cloud/conda-forge/linux-64::ncurses-6.5-h2d0b736_3 
  openssl            anaconda/cloud/conda-forge/linux-64::openssl-3.6.1-h35e630c_1 
  packaging          anaconda/cloud/conda-forge/noarch::packaging-26.0-pyhcf101f3_0 
  pip                anaconda/cloud/conda-forge/noarch::pip-26.0.1-pyh8b19718_0 
  python             anaconda/cloud/conda-forge/linux-64::python-3.11.14-hd63d673_3_cpython 
  readline           anaconda/cloud/conda-forge/linux-64::readline-8.3-h853b02a_0 
  setuptools         anaconda/cloud/conda-forge/noarch::setuptools-81.0.0-pyh332efcf_0 
  tk                 anaconda/cloud/conda-forge/linux-64::tk-8.6.13-noxft_h366c992_103 
  tzdata             anaconda/cloud/conda-forge/noarch::tzdata-2025c-hc9c84f9_1 
  wheel              anaconda/cloud/conda-forge/noarch::wheel-0.46.3-pyhd8ed1ab_0 
  zstd               anaconda/cloud/conda-forge/linux-64::zstd-1.5.7-hb78ec9c_6 


Proceed ([y]/n)? 

Downloading and Extracting Packages: ...working... done
Preparing transaction: / - done
Verifying transaction: | / - \ | / - \ | done
Executing transaction: - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ done
#
# To activate this environment, use
#
#     $ conda activate CellCausal
#
# To deactivate an active environment, use
#
#     $ conda deactivate

Looking in indexes: https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
Looking in links: https://download.pytorch.org/whl/cu118/torch_stable.html
Collecting torch==2.0.1+cu118
  Using cached https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp311-cp311-linux_x86_64.whl (2267.3 MB)
Collecting torchvision==0.15.2+cu118
  Using cached https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp311-cp311-linux_x86_64.whl (6.1 MB)
Collecting torchaudio==2.0.2+cu118
  Using cached https://download.pytorch.org/whl/cu118/torchaudio-2.0.2%2Bcu118-cp311-cp311-linux_x86_64.whl (4.4 MB)
Collecting filelock (from torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/b5/36/7fb70f04bf00bc646cd5bb45aa9eddb15e19437a28b8fb2b4a5249fac770/filelock-3.20.3-py3-none-any.whl (16 kB)
Collecting typing-extensions (from torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Collecting sympy (from torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl (6.3 MB)
Collecting networkx (from torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl (2.1 MB)
Collecting jinja2 (from torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl (134 kB)
Collecting triton==2.0.0 (from torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/b7/cd/4aa0179919306f9c2e3e5308f269d20c094b2a4e2963b656e9405172763f/triton-2.0.0-1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (63.3 MB)
Collecting numpy (from torchvision==0.15.2+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/1b/46/6fa4ea94f1ddf969b2ee941290cca6f1bfac92b53c76ae5f44afe17ceb69/numpy-2.4.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.9 MB)
Collecting requests (from torchvision==0.15.2+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/1e/db/4254e3eabe8020b458f1a747140d32277ec7a271daf1d235b70dc0b4e6e3/requests-2.32.5-py3-none-any.whl (64 kB)
Collecting pillow!=8.3.*,>=5.3.0 (from torchvision==0.15.2+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/5c/1f/8e66ab9be3aaf1435bc03edd1ebdf58ffcd17f7349c1d970cafe87af27d9/pillow-12.1.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (7.0 MB)
Collecting cmake (from triton==2.0.0->torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/28/19/b54ff2e03946beeef785e6407d965a9493d26c50dd1aa09ffc7b53fbf9a5/cmake-4.2.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (28.9 MB)
Collecting lit (from triton==2.0.0->torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/96/06/b36f150fa7c5bcc96a31a4d19a20fddbd1d965b6f02510b57a3bb8d4b930/lit-18.1.8-py3-none-any.whl (96 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/30/ac/0273f6fcb5f42e314c6d8cd99effae6a5354604d461b8d392b5ec9530a54/markupsafe-3.0.3-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (22 kB)
Collecting charset_normalizer<4,>=2 (from requests->torchvision==0.15.2+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/6d/fc/de9cce525b2c5b94b47c70a4b4fb19f871b24995c728e957ee68ab1671ea/charset_normalizer-3.4.4-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (151 kB)
Collecting idna<4,>=2.5 (from requests->torchvision==0.15.2+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/0e/61/66938bbb5fc52dbdf84594873d5b51fb1f7c7794e9c0f5bd885f30bc507b/idna-3.11-py3-none-any.whl (71 kB)
Collecting urllib3<3,>=1.21.1 (from requests->torchvision==0.15.2+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/39/08/aaaad47bc4e9dc8c725e68f9d04865dbcb2052843ff09c97b08904852d84/urllib3-2.6.3-py3-none-any.whl (131 kB)
Collecting certifi>=2017.4.17 (from requests->torchvision==0.15.2+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/e6/ad/3cc14f097111b4de0040c83a525973216457bbeeb63739ef1ed275c1c021/certifi-2026.1.4-py3-none-any.whl (152 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy->torch==2.0.1+cu118)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl (536 kB)
Installing collected packages: mpmath, lit, urllib3, typing-extensions, sympy, pillow, numpy, networkx, MarkupSafe, idna, filelock, cmake, charset_normalizer, certifi, requests, jinja2, triton, torch, torchvision, torchaudio

Successfully installed MarkupSafe-3.0.3 certifi-2026.1.4 charset_normalizer-3.4.4 cmake-4.2.1 filelock-3.20.3 idna-3.11 jinja2-3.1.6 lit-18.1.8 mpmath-1.3.0 networkx-3.6.1 numpy-2.4.2 pillow-12.1.0 requests-2.32.5 sympy-1.14.0 torch-2.0.1+cu118 torchaudio-2.0.2+cu118 torchvision-0.15.2+cu118 triton-2.0.0 typing-extensions-4.15.0 urllib3-2.6.3
Looking in indexes: https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
Looking in links: https://data.pyg.org/whl/torch-2.0.1+cu118.html
Collecting torch-scatter
  Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.2%2Bpt20cu118-cp311-cp311-linux_x86_64.whl (10.2 MB)
Collecting torch-sparse
  Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.18%2Bpt20cu118-cp311-cp311-linux_x86_64.whl (4.9 MB)
Collecting torch-cluster
  Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_cluster-1.6.3%2Bpt20cu118-cp311-cp311-linux_x86_64.whl (3.3 MB)
Collecting torch-spline-conv
  Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt20cu118-cp311-cp311-linux_x86_64.whl (886 kB)
Collecting scipy (from torch-sparse)
  Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/ef/df/df1457c4df3826e908879fe3d76bc5b6e60aae45f4ee42539512438cfd5d/scipy-1.17.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (35.1 MB)
Requirement already satisfied: numpy<2.7,>=1.26.4 in /data/conda-envs/limengran/CellCausal/lib/python3.11/site-packages (from scipy->torch-sparse) (2.4.2)
Installing collected packages: torch-spline-conv, torch-scatter, scipy, torch-sparse, torch-cluster

Successfully installed scipy-1.17.0 torch-cluster-1.6.3+pt20cu118 torch-scatter-2.1.2+pt20cu118 torch-sparse-0.6.18+pt20cu118 torch-spline-conv-1.2.2+pt20cu118

## 📂 Project Structure



## ⚙️ Experiment Settings & Environment

### Hardware & Software Infrastructure

Experiments are conducted on high-performance nodes tailored.

* **CPU:** Dual Intel Xeon Platinum 8336C @ 2.30GHz
* **GPU:** NVIDIA RTX 5880 Ada Generation (48GB VRAM)
* **Memory:** 512 GB DDR4 ECC
* **Software:** Python 3.11.14, PyTorch 2.0.1+cu118, PyG 2.3.0, CUDA 11.8

### Hyperparameters (Key Configurations)

The Dual-Space Bilevel Optimization is controlled via hierarchical configs:

* **LLM Engine:** Gemini 3 Pro (Temp: 0.5 - 0.7)
* **Design Phase:** 4 parallel hypothesis branches; Max 3 self-correction fix rounds.
* **Execution Phase:** Global timeout 100h; Step timeout 5h; Max 5 debugging rounds.
* **Review Phase:** Max 10 optimization iterations; Optimized via Pearson Correlation Coefficient (PCC).

### Cost Efficiency

CellCausal minimizes cost through a **Contextual Memory** mechanism that reduces token load by ~60% in later iterations.

* **Average Run (3-5 iterations):** .00 - .00 USD
* **Complex Run (10 iterations):** < .00 USD

## 🚀 Usage


               🧬 CellScientist Pipeline Configuration (BBBC036)                
┏━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃            ┃         ┃         ┃         ┃        ┃         ┃  Max   ┃       ┃
┃ Stage      ┃ Model   ┃ Litera… ┃  BioKB  ┃  GPU   ┃ Timeout ┃ Iters  ┃ Cache ┃
┡━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ Experiment │ gemini… │ ✓ (N/A) │ ✓ (N/A) │ GPUN/A │     N/A │   3    │   ✓   │
│ Review     │ gemini… │ ✓ (N/A) │ ✓ (N/A) │ GPUN/A │     N/A │   5    │   ✓   │
└────────────┴─────────┴─────────┴─────────┴────────┴─────────┴────────┴───────┘


🔄 EXPERIMENT STAGE
