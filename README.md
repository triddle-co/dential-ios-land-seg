# ToothInstanceNet

Welcome to the repository of our solution to the [3DTeethLand challenge](https://www.synapse.org/Synapse:syn57400900/wiki/).

![alt text](docs/method.png "ToothInstanceNet")


## Install

First, setup a Conda environment.

``` bash
conda create -n 3dteethland python=3.10
conda activate 3dteethland
```

Then, install the Pip requirements.

``` bash
pip install -r requirements.txt
```

Lastly, compile and install the Cuda kernels.

``` bash
pip install -v -e .
```


## Inference

Please specify the root directory where your scans are stored using the `root` keyword in `teethland/config/config.yaml`. The file names of the scans are expected to contain whether they are of the lower or upper jaw as `STEM_(lower|upper).(stl|ply|obj)`. Please rename the scans using this format to get the correct FDI labels.

Then, download the checkpoints from [here](https://drive.google.com/drive/folders/1MIPNtsM3rW_VAUtD8RBPOso1IxyJZgdF?usp=sharing) and specify their paths using the `checkpoint_path` keywords in `teethland/config/config.yaml`.

Finally, run the model with

``` bash
python infer.py landmarks --devices DEVICES
```

where `DEVICES` can be set to use multiple GPUs for inference. The tooth instance segmentations will be saved next to the scan as `STEM_(lower|upper).json` and the detected landmarks will be saved next to the scan as `STEM_(lower|upper)__kpt.json`.


## Cite

```
@inproceedings{toothinstancenet,
  title = {ToothInstanceNet: Comprehensive Information from Intra-oral Scans by Integration of Large-Context and High-Resolution Predictions},
  booktitle = {Supervised and Semi-supervised Multi-structure Segmentation and Landmark Detection in Dental Data},
  pages = {229--240},
  year = {2025},
  month = {05},
  doi = {10.1007/978-3-031-88977-6_21},
  author = {Niels {van Nistelrooij} and Shankeeth Vinayahalingam}
}
```

```
@article{mixed_ios,
  title = {Automated detection and numbering of primary and permanent teeth in digital impressions of children using artificial intelligence},
  journal = {Journal of Dentistry},
  volume = {161},
  pages = {105976},
  year = {2025},
  month = {07},
  doi = {10.1016/j.jdent.2025.105976},
  author = {Niels {van Nistelrooij} and Haline Cunha de Medeiros Maia and Lingyun Cao and Shankeeth Vinayahalingam and Bas Loomans and Maximiliano Sergio Cenci and Fausto Medeiros Mendes}
}
```

```
@article{partial_ios,
  title = {Fully Automated Tooth Segmentation and Labeling for Both Full- and Partial-Arch Intraoral Scans Using Deep Learning},
  journal = {International Dental Journal},
  volume = {75},
  number = {5},
  pages = {100950},
  year = {2025},
  month = {08},
  doi = {10.1016/j.identj.2025.100950},
  author = {Lingyun Cao and Niels {van Nistelrooij} and Jiaqi Liu and Shankeeth Vinayahalingam and Maximiliano Sergio Cenci and Tong Xi and Bas A.C. Loomans}
}
```
