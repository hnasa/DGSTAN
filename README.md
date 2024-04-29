
##Table of Contents

```text
basicts   --> The BasicTS, which provides standard pipelines for training MTS forecasting models. 

datasets  --> Raw datasets and preprocessed data

scripts   --> Data preprocessing scripts.

basicts/archs/arch_zoo/dgstan_arch/      --> The implementation of DGSTAN.

examples/DGSTAN/ DGSTAN_${DATASET_NAME}.py    --> Training configs.
```

Replace `${DATASET_NAME}` with one of `PEMS04`, `PEMS07`, `PEMS08` and `PEMS03`.


### Python

Python >= 3.6 (recommended >= 3.9).

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) are recommended to create a virtual python environment.

### Other Dependencies

BasicTS is built based on PyTorch and [EasyTorch](https://github.com/cnstark/easytorch).
You can install PyTorch following the instruction in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

After ensuring that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

### Preparing Data

- **Download Raw Data**

    You can download all the raw datasets at [Google Drive](https://drive.google.com/drive/folders/1vu59RzLFqPli1yXFpP4wISRYOYa8scyt?usp=drive_link) , and unzip them to `datasets/raw_data/`.

- **Pre-process Data**

    ```bash
    cd /path/to/your/project
    python scripts/data_preparation/${DATASET_NAME}/generate_training_data.py
    ```

    Replace `${DATASET_NAME}` with one of  `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`, or any other supported dataset. The processed data will be placed in `datasets/${DATASET_NAME}`.



    ```bash
    cd /path/to/your/project
    bash scripts/data_preparation/all.sh
    ```


### Run It!


```bash
python examples/run.py -c examples/DGSTAN/DGSTAN_PEMS08.py --gpus '0'
```
```or bash
bash train.sh
```





