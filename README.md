# Personal Knowledge Graph Embedding

## About

Generate embeddings for personal knowledge graph.

## Installation

1. Install TensorFlow and Keras

2. In this directory, run:

	$ bash make.sh


## Usage

```
python pkg_embedding.py [-h] -openke OPENKE_DIR -pkg PKG_DIR -output
                        OUTPUT_DIR [-model MODEL_NAME] [-ndays NDAYS]
                        [-weight_threshold WEIGHT_THRESHOLD] [-day_rel]
                        [-phases PHASES] [-alpha ALPHA] [-nbatches NBATCHES]
                        [-epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -pkg PKG_DIR          directory of pkg dir
  -openke OPENKE_DIR    directory of openke format graph
  -output OUTPUT_DIR    directory of output dir
  -model MODEL_NAME     model to generate the embeddings, could be 1hot, TransE, TransH, HolE, RESCAL, etc.
  -ndays NDAYS          number of days of knowledge to include
  -weight_threshold WEIGHT_THRESHOLD
                        weight threshold of knowledge to include
  -day_rel              whether to represent days with different relations
  -phases PHASES        phases to run, could be one or more of gen_kg, train, evaluate and visualize.
  -alpha ALPHA          hyper parameter: alpha.
  -nbatches NBATCHES    hyper parameter: nbatches.
  -epochs EPOCHS        hyper parameter: epochs.
```

## Data

初始时需要准备个人知识图谱数据，数据样例在`data/sample_pkg`目录下，其中需要包含三个文件：

- user_id_imei_birth_gender.txt 用户相关信息，每一行是一个user，其id，imeimd5，birthday，gender以空格分隔。
- app_id_package_usercount.txt 实体（应用）相关信息，每一行是一个app，其id，package，usercount以空格分隔。
- user_app_day_duration.txt 用户与实体关系信息，每一行是一个user与app的使用关系，分别是user_id, app_id, day, duration以空格分隔。

数据处理逻辑在pkg/pkg.py文件中，如果需要修改数据格式，可能需要修改对应文件读写代码。

## Example

假设知识图谱数据在`data/sample_pkg`目录下，生成embedding可以使用如下代码：

```
python pkg_embedding.py -pkg data/sample_pkg/ -openke data/openke/ -output data/embedding_TransE/ -model TransE -phases gen_kg,train
```

执行完毕之后即可在`data/embedding_TransE/`目录找到使用TransE模型生成的embedding文件。
