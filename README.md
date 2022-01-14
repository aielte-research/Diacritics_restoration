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
    <h3 align="center">Lightweight Diacritics Restoration for V4 Languages</h3>

  <p align="center">
    <br />
    <a href="https://web.cs.elte.hu/~csbalint/diacritics/demo.html?lang=en&model_lang=HU"><strong>Try the model »</strong></a>
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Diacritics restoration became a ubiquitous task in the Latin-alphabet-based English-dominated Internet language environment.
In this repository, we provide a model built with small footprint 1D convolution-based approach, which works on character-level.
The model even runs locally in a web browser, and surpasses the performance of similarly sized models.

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

The project logs both locally and to <a href="https://neptune.ai/"><strong>neptune.ai</strong></a>, a neptune.ai account is neeeded at the moment.

Copy `neptune_cfg_template.yaml` to `neptune_cfg.yaml`, and fill out the appropriate details:
```yaml
project_qualified_name: 
api_token: 
offline_logging_dir: 
```

### Installation

For the moment refer to the example below and install the missing packages manually.

<!-- USAGE EXAMPLES -->
## Usage

The following command should work with the small example data provided in this repository. 

```sh
CUDA_VISIBLE_DEVICES=0 python run.py -c conf/example.yaml
```

