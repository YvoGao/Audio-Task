# TAPE: Task-Adapt and Prototype Evolution in Audio-Language Models

## quick start

create ENV
```
conda create -n audio python=3.7
source activate audio
pip install -r requirements.txt
```
run

Noting: before you start, you should download bert-base-uncased from https://huggingface.co/google-bert/bert-base-uncased, and change the path in the ./pengi/configs/base.yml file to your own file path. and you can download the pengi cheakpoint by wget `https://zenodo.org/records/8387083/files/base.pth` put at the path ./pengi/configs/

The specific parameters per dataset in the paper are consistent with run.sh.
```
sh run.sh
```
