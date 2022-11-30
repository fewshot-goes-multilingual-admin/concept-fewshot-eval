import abc
import random
from typing import Optional, Tuple, List, Union

from transformers import PreTrainedTokenizer, PreTrainedModel

from evaluation.evaluator import Evaluator
from evaluation.tasks.task import Task


class InfoDiffEvaluatorBase(abc.ABC):

    def __init__(self,
                 task: Task,
                 num_demonstrations: int = 3,
                 firstn: Optional[int] = None,
                 bootstrap: bool = False,
                 max_input_length: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.num_demonstrations = num_demonstrations
        self.firstn = firstn
        self.bootstrap = bootstrap
        self.max_input_length = max_input_length

    @abc.abstractmethod
    def _compute(self, expected: List[str], actual: List[str]) -> float:
        pass

    def _compute_bootstrapped(self,
                              expected_all: List[str],
                              actual_all: List[str],
                              per_round_samples: int = 50,
                              repeats: int = 200) -> List[float]:
        assert len(expected_all) == len(actual_all), "Prediction lists' length do not match"

        evals = []
        while len(evals) < repeats:
            subset_idx = [random.randrange(len(expected_all)) for _ in range(per_round_samples)]
            expected_subset = [expected_all[idx] for idx in subset_idx]
            actual_subset = [actual_all[idx] for idx in subset_idx]

            evals.append(self._compute(expected_subset, actual_subset))

        return evals

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, _) -> Tuple[Union[float, List[float]],
                                                                                           Union[float, List[float]]]:
        # print("Model's performance in random selection: %s" % random_performance)
        # there's always less samples in 'informative' group
        expected, actual_informative, eval_set = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                               self.num_demonstrations, self.firstn,
                                                                               demo_selection_strategy="informative",
                                                                               max_input_length=self.max_input_length)
        expected, actual_random, _ = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                   self.num_demonstrations, self.firstn,
                                                                   demo_selection_strategy="random", eval_set=eval_set)
        if self.bootstrap:
            informative_performance = self._compute_bootstrapped(expected, actual_informative)
            random_performance = self._compute_bootstrapped(expected, actual_informative)
        else:
            informative_performance = self._compute(expected, actual_informative)
            random_performance = self._compute(expected, actual_random)

        # print("Model's performance in informative selection: %s" % informative_performance)

        return random_performance, informative_performance

    def __str__(self):
        return "%s_%s" % (self.task, super().__str__())


class RougeInfoDIff(InfoDiffEvaluatorBase):

    def _compute(self, expected: List[str], actual: List[str]) -> Union[float, List[float]]:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        all_scores = [scorer.score(expected, actual)['rougeL'].recall
                      for expected, actual in zip(expected, actual)]
        return sum(all_scores) / len(expected)


class AccuracyInfoDIff(InfoDiffEvaluatorBase):

    def _compute(self, expected: List[str], actual: List[str]) -> Union[float, List[float]]:
        num_correct = sum([exp == act for exp, act in zip(expected, actual)])
        return num_correct / len(expected)
