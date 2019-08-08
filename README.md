**Status:** Archive (code is provided as-is, no updates expected)

DeepType: Multilingual Entity Linking through Neural Type System Evolution
--------------------------------------------------------------------------

This repository contains code necessary for designing, evolving type systems, and training neural type systems. To read more about this technique and our results [see this blog post](https://blog.openai.com/discovering-types-for-entity-disambiguation/) or [read the paper](https://arxiv.org/abs/1802.01021).

Authors: Jonathan Raiman & Olivier Raiman

Steps Rewrite Author: Yifan Ding yding4@nd.edu 
### Notice that the in the package install, it requires tensorflow_gpu==1.4 and Cython== 0.26
to set up tensorflow_gpu==1.4, needs CUDA8 and cuDNN 6, see [stack overflow](https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible)

### Data collection

Get wikiarticle -> wikidata mapping (all languages) + Get anchor tags, redirections, category links, statistics (per language). To store all wikidata ids, their key properties (`instance of`, `part of`, etc..), and
a mapping from all wikipedia article names to a wikidata id do as follows,
along with wikipedia anchor tags and links, in three languages: English (en), French (fr), and Spanish (es):

#### original

```bash
export DATA_DIR=data/
./extraction/full_preprocess.sh ${DATA_DIR} en fr es
```

#### revised
```bash
./extraction/full_preprocess.sh /data/datasets/wikipedia/ en
```

### Create a type system manually and check oracle accuracy:

To build a graph projection using a set of rules inside `type_classifier.py`
(or any Python file containing a `classify` method), and a set of nodes
that should not be traversed in `blacklist.json`:

**Note** Some ID of The blacklist.json is out of date
To save a graph projection as a numpy array along with a list of classes to a
directory stored in `CLASSIFICATION_DIR`:
**note**
the default classifier has four different types, time, location, type and location

#### original
```bash
export LANGUAGE=fr
export DATA_DIR=data/
export CLASSIFICATION_DIR=data/type_classification
python3 extraction/project_graph.py ${DATA_DIR}wikidata/ extraction/classifiers/type_classifier.py  --export_classification ${CLASSIFICATION_DIR}
```
#### revised
```bash
export LANGUAGE=en
export DATA_DIR=/data/datasets/wikipedia
export CLASSIFICATION_DIR=/data/datasets/wikipedia/type_classification
python3 extraction/project_graph.py ${DATA_DIR}wikidata/ extraction/classifiers/type_classifier.py  --export_classification ${CLASSIFICATION_DIR}
```

**can be skipped** To use the saved graph projection on wikipedia data to test out how discriminative this
classification is (Oracle performance) (edit the config file to make changes to the classification used):

#### original
```bash
export DATA_DIR=data/
python3 extraction/evaluate_type_system.py extraction/configs/en_disambiguator_config_export_median.json --relative_to ${DATA_DIR}
```
#### revised
#### evaluate on 100000 samples 
#### create a new file extraction/configs/en_disambiguator_config_export_median.json and change the sample_number to 100000
```bash
python3 extraction/evaluate_type_system.py extraction/configs/en_disambiguator_config_export_median.json --relative_t /data/datasets/wikipedia
```

## Paper steps start
### Obtain learnability scores for types
#### original
```bash
export DATA_DIR=data/
python3 extraction/produce_wikidata_tsv.py extraction/configs/en_disambiguator_config_export_small.json --relative_to ${DATA_DIR} sample_data.tsv
python3 learning/evaluate_learnability.py sample_data.tsv --out report.json --wikidata ${DATA_DIR}wikidata/
```
See `learning/LearnabilityStudy.ipynb` for a visual analysis of the AUC scores.

#### revised
**note** second step needs to run on tensorflow_gpu. To run on cpu, you can set "device" to cpu in the evaluate_learnability.py

```bash
python3 extraction/produce_wikidata_tsv.py extraction/configs/en_disambiguator_config_export_median.json --relative_to /data/datasets/wikipedia/ sample_data.tsv

python3 learning/evaluate_learnability.py --dataset sample_data.tsv --out report.json --wikidata /data/datasets/wikipedia/wikidata/
```



### Evolve a type system(create type_system by methods in the paper)
#### original
```bash
python3 extraction/evolve_type_system.py extraction/configs/en_disambiguator_config_export_small.json --relative_to ${DATA_DIR}  --method cem  --penalty 0.00007
```

#### revised
**note** need to add output file name after configure file <br>
**note** need to change the evolve_type_system.py to add report file gained from learning score step, named "report.json" to select type systems.<br>

```bash
python3 extraction/evolve_type_system.py extraction/configs/en_disambiguator_config_export_median.json  cem_100000.json --relative_to /data/datasets/wikipedia  --method cem  --penalty 0.00007
```

Method can be `cem`, `greedy`, `beam`, or `ga`, and penalty is the soft constraint on the size of the type system (lambda in the paper).

#### Convert a type system solution into a trainable type classifier
**note** this step is neccessary to create envolving type system to compare with the manual type system.
The output of `evolve_type_system.py` is a set of types (root + relation, in a json file) that can be used to build a type system. To create a config file that can be used to train an LSTM use the jupyter notebook `extraction/TypeSystemToNeuralTypeSystem.ipynb`.

### Train a type classifier using a type system
For each language create a training file:
#### original
```bash
export LANGUAGE=en
python3 extraction/produce_wikidata_tsv.py extraction/configs/${LANGUAGE}_disambiguator_config_export.json /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.tsv  --relative_to /Volumes/Samsung_T3/tahiti/2017-12/
```

#### revised 
**note** need to change the input file name somewhere
```bash
python3 extraction/produce_wikidata_tsv.py extraction/configs/en_disambiguator_config_export_median.json /data/datasets/wikipedia/en_train.tsv  --relative_to /data/datasets/wikipedia/
```

Then create an H5 file from each language containing the mapping from tokens to their entity ids in Wikidata:

#### original
```bash
export LANGUAGE=en
python3 extraction/produce_windowed_h5_tsv.py  /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.tsv /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.h5 /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_dev.h5 --window_size 10  --validation_start 1000000 --total_size 200500000
```

#### revised
```bash
python3 extraction/produce_windowed_h5_tsv.py  /data/datasets/wikipedia/en_train.tsv /data/datasets/wikipedia/en_train.h5 /data/datasets/wikipedia/en_dev.h5 --window_size 10  --validation_start 1000000 --total_size 200500000
```


Create a training config with all languages, `my_config.json`. Paths to the datasets is relative to config file (e.g. you can place it in the same directory as the dataset h5 files):
[Note: set `wikidata_path` to where you extracted wikidata information, and `classification_path` to where you exported the classifications with `project_graph.py`]. See learning/configs for a pre written config covering English, French, Spanish, German, and Portuguese.


Launch training on a single gpu:
#### original
```bash
CUDA_VISIBLE_DEVICES=0 python3 learning/train_type.py my_config.json --cudnn --fused --hidden_sizes 200 200 --batch_size 256 --max_epochs 10000  --name TypeClassifier --weight_noise 1e-6  --save_dir my_great_model  --anneal_rate 0.9999
```

#### revised CPU version
```bash
python3 learning/train_type.py my_config.json --cudnn --fused --hidden_sizes 200 200 --batch_size 256 --max_epochs 10000  --name TypeClassifier --weight_noise 1e-6  --save_dir my_great_model  --anneal_rate 0.9999 --device cpu --faux_cudnn
```
Several key parameters:

- `name`: main scope for model variables, avoids name clashing when multiple classifiers are loaded
- `batch_size`: how many examples are used for training simultaneously, can cause out of memory issues
- `max_epochs`: length of training before auto-stopping. In practice this number should be larger than needed.
- `fused`: glue all output layers into one, and do a single matrix multiply (recommended).
- `hidden_sizes`: how many stacks of LSTMs are used, and their sizes (here 2, each with 200 dimensions).
- `cudnn`: use faster CuDNN kernels for training
- `anneal_rate`: shrink the learning rate by this amount every 33000 training steps
- `weight_noise`: sprinkle Gaussian noise with this standard deviation on the weights of the LSTM (regularizer, recommended).


### To test that training works:

You can test that training works as expected using the dummy training set containing a Part of Speech CRF objective and cat vs dogs log likelihood objective is contained under learning/test:

#### orginal
```bash
python3 learning/train_type.py learning/test/config.json
```

#### revised cpu version 
```bash
python3 learning/train_type.py learning/test/config.json --device cpu --faux_cudnn
```

### Installation

#### Mac OSX

```
pip3 install -r requirements.txt
pip3 install wikidata_linker_utils_src/
```

