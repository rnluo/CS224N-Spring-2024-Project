# Multitask Training with BERT

***Note:** This repository is a fork of the original Stanford CS224N Spring 2024 Default Final Project starter code. The original README information is preserved below.*

## Description

This repository documents my work on the final project of CS224N (Spring 2024). The goal of my project was to utilize multitask training on a BERT model to improve its performance on three sentence-level tasks.

### Features

* Explored multitask training heuristics including undersampling and uncertainty weighted loss.
* Applied gradient techniques including PCGrad and gradient clipping.
* The best performing model scores 17% higher compared to the baseline.
* For more implementation details and experimental results, please refer to the [project report](./Report/report.pdf).

### Usage

* Clone the repository and set up the environment using Conda:
```
conda env create -f environment.yml
```
* [Download](https://huggingface.co/google-bert/bert-base-uncased) the `bert-base-uncased` model into the working directory, or configure the code to use the model remotely.
* Run `multitask_classifier.py` to start finetuning. The list of command line arguments can be seen in `get_args()`. 
---

# Original README

# CS 224N Default Final Project - Multitask BERT

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the BERT model to better understand its architecture. 
In the second part, you will use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection, and semantic similarity. You will implement extensions to improve your model's performance on the three downstream tasks.

In broad strokes, Part 1 of this project targets:
* bert.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

And Part 2 targets:
* multitask_classifier.py: Missing code blocks.
* datasets.py: Possibly useful functions/classes for extensions.
* evaluation.py: Possibly useful functions/classes for extensions.

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).