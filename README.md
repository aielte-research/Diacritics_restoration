<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->
<br />
<div align="center">
    <h2 align="center">Lightweight Diacritics Restoration<br />with<br />Dilated Convolutional Neural Networks</h2>
    <h3 align="center">
        <br />
            <a href="https://ai.elte.hu/csbalint/diacritics/demo.html?lang=en&model_lang=HU" target="_blank"><strong>Try the model »</strong></a>
        <br />
    </h3>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains code for training, evaluation, and inference of our lightweight model for diacritics restoration, which employs a character-level 1D convolutional architecture.
We demonstrate that solutions based on 1D dilated convolutional neural networks are competitive alternatives to models utilizing recurrent neural networks or linguistic modeling for diacritics restoration.
Our proposed solution outperforms models of comparable size and demonstrates competitiveness with larger models.
An additional advantage of our solution is its ability to run locally in a web browser, demonstrated in a functional implementation.
We evaluate our model on various corpora, with an emphasis on the Hungarian language.
We conducted comparative analyses to assess the generalization capabilities of the model across three Hungarian corpora.
Additionally, we analyzed the errors to understand the limitations of corpus-based self-supervised training.
More information can be found in our <a href="https://arxiv.org/abs/2201.06757" target="_blank"><strong>paper</strong></a> presented at LREC2022.

### Model architecture visualization
<p align="center">
  <img src="./img/A-TCN_4225_zoomed_txt.png" alt="Model Architecture">
</p>

### Built With

* [PyTorch](https://pytorch.org/)
* [ONNX.js](https://github.com/microsoft/onnxjs)
* [Hungarian Webcorpus 2.0](https://hlt.bme.hu/en/resources/webcorpus2)
* [Hunaccent](https://github.com/juditacs/hunaccent)
* [neptune.ai](https://neptune.ai/)

<!-- GETTING STARTED -->
## Getting Started

If you want to try out the model, the demo is available <a href="https://web.cs.elte.hu/~csbalint/diacritics/demo.html?lang=en&model_lang=HU"><strong>here</strong></a>.

For training the model:

### Prerequisites

The project logs both locally and to <a href="https://neptune.ai/"><strong>neptune.ai</strong></a>,
to use a neptune.ai an account is neeeded. Logging to neptune can be disabled for individual experiments in the experiment's config, or globally by not providing an api token in the `neptune_cfg.yaml`.

Copy `neptune_cfg_template.yaml` to `neptune_cfg.yaml`, and fill out the appropriate details:
```yaml
project_qualified_name: 
api_token: 
offline_logging_dir: 
```

### Installation

Refer to the example below and install the missing packages manually,

or use the `environment.yml` file: `conda env create -f environment.yml`.

<!-- USAGE EXAMPLES -->
## Usage

The following command should work with the small example data provided in this repository. 

```sh
CUDA_VISIBLE_DEVICES=0,1 python run.py -c conf/example.yaml
```

## How to Cite
If you use this code in your research, please cite the corresponding paper:
```bibtex
@inproceedings{csanady-lukacs-2022-dilated,
    title = "Dilated Convolutional Neural Networks for Lightweight Diacritics Restoration",
    author = "Csan{\'a}dy, B{\'a}lint  and
      Luk{\'a}cs, Andr{\'a}s",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.452/",
    pages = "4253--4259",
}
```

### Contributors
- Bálint Csanády (csbalint@protonmail.ch)
- András Lukács (andras.lukacs@ttk.elte.hu)
