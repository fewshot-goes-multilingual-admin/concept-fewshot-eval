from typing import List

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU, ROUGE
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from priming_objective import Priming

training_arguments = AdaptationArguments(output_dir="train_dir_info",
                                         learning_rate=5e-5,
                                         # stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=30,  # TODO: set
                                         eval_steps=100,  # TODO: set
                                         logging_steps=10,
                                         save_steps=1000,
                                         num_train_epochs=50,
                                         evaluation_strategy="steps",
                                         save_total_limit=10,
                                         stopping_patience=30)
eval_examples = 200  # TODO set

# priming
num_demonstrations = 3

val_metrics = [BLEU(decides_convergence=True), ROUGE()]


def _construct_priming_prompt(previous_examples: List[str], current_example: str) -> str:
    return " ".join(previous_examples + [current_example])


# lang_module = LangModule("google/mt5-small")  # TODO set
# lang_module = LangModule("google/mt5-base")
lang_module = LangModule("google/mt5-large")

# priming
per_type_examples = {}

qa_en = load_dataset("adversarial_qa", "adversarialQA")
qa_train = qa_en["train"].filter(lambda entry: len(entry["context"]) < 2000)


def _get_en_squad_categories(data) -> List[str]:
    return [question.split()[0] if not question.startswith("To")
            else " ".join(question.split()[:2])
            for question in data["question"]]


q_answering_en = Priming(lang_module,
                         difficulty_sample=64,  # TODO set
                         demos_selection_strategy="hard",  # TODO set
                         texts_or_path=qa_train["question"],
                         text_pair_or_path=qa_train["context"],
                         val_texts_or_path=qa_en["validation"]["question"][-eval_examples:],
                         val_text_pair_or_path=qa_en["validation"]["context"][-eval_examples:],
                         labels_or_path=[a["text"][0] for a in qa_train["answers"]],
                         val_labels_or_path=[a["text"][0] for a in qa_en["validation"]["answers"]][-eval_examples:],
                         train_question_categories=_get_en_squad_categories(qa_train),
                         val_question_categories=_get_en_squad_categories(qa_en["validation"])[-eval_examples:],
                         batch_size=1,
                         val_evaluators=val_metrics,
                         source_lang_id="en",
                         objective_id="AQA-en")

schedule = ParallelSchedule(objectives=[q_answering_en],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
