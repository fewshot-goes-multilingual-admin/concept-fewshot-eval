## Conceptual Few-shot evaluation

This repository contains scripts for evaluating 
few-shot in-context learners on sensitivity to the 
shared demonstrations concepts.

Contrary to the previous few-shot evaluation, we choose
the in-context priming demonstrations that share a specific
_concept_ with the predicted sample, that the model can exploit
and improve prediction quality. See the referencing paper for terminology and details.


### Evaluation

A few-shot learner can be evaluated on supported datasets 
by running the following commands:

```shell
git clone {this repo}
cd concept-fewshot-eval
export PYTHONPATH=$pwd:{$PYTHONPATH}

pip install -r evaluation/requirements.txt
[CUDA_VISIBLE_DEVICES={...}] python evaluation/evaluate.py --dataset_ids glue/mnli,openbookqa/additional,hotpot_qa/fullwiki,worldtree 
                                                           --metric ROUGE 
                                                           --model_names_or_paths allenai/tk-instruct-large-def-pos
```
where:
* `dataset_ids` is coma-separated list of evaluation datasets, for which we implement concept extraction (implemented in `evaluation/tasks/en`).
  Note that the names correspond to Promptsource's dataset identifiers; see [all templates](https://github.com/bigscience-workshop/promptsource/tree/main/promptsource/templates).
* `metric` is the evaluation metric for which we report results
* `model_names_or_paths` is coma-separated list of evaluated HuggingFace models with `generate()` method; 
   Either local directory paths, or model identifiers from https://huggingface.co/models
* `template_names`: If specified, a subset of Promptsource template ids for given `dataset_ids` 
  is evaluated.
* `bootstrap`: Whether to perform bootstrapped, or conventional evaluation, Defaults to `True`.
* `max_input_length`: If given, will exclude the inputs longer than the given number of 
  space-separated words from the evaluation, this is to avoid memory overflow on very large contexts of HotpotQA, 
  ranging up to 5000 sentencepieces. 

### Training experiments

We measure models' inherent ability to learn demonstrations' conceptual sensitivity
by clustering QA dataset samples by a question-word of samples' question 
(i.e. putting together all questions starting with "Who", or "How")
and fine-tune mT5 models with contexts of demonstrations sharing a question-word (i.e. a concept)
with the sample to be predicted (see the referencing Paper for more details).

We run two experiments comparing the impact of concepts' sharing, as follows:

```shell
pip install -r training/requirements.txt
```

1. Random selection (baseline):
```shell
[CUDA_VISIBLE_DEVICES={...}] python training/train_mt5_qa_random_demos.py
```
2. Concept-sharing selection (ours):
```shell
[CUDA_VISIBLE_DEVICES={...}] python training/train_mt5_qa_informative_demos.py
```

The model of a chosen checkpoint (persisted in `train_dir_{random/info}`) 
can be directly evaluated using the evaluate script above.
We pick our checkpoints by the largest BLEU reported on a validation set. 
