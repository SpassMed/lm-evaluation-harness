"""
What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
https://arxiv.org/abs/2009.13081

This contains the English portion of the full MedQA dataset, containing 12,723 multiple (4) choice questions from the US medical licensing exam.

Homepage: None
Credit to: https://github.com/TimD1 for the pending PR in the lm-evaluation harness repo

"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@misc{jin2020disease,
    title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams}, 
    author={Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
    year={2020},
    eprint={2009.13081},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class MedQA_USMLE(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "GBaker/MedQA-USMLE-4-options-hf-MiniLM-IR-cs"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return []

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])


    def _process_doc(self, doc):
        _option_dict = {
            "A": doc['ending0'],
            "B": doc['ending1'],
            "C": doc['ending2'],
            "D": doc['ending3']
            }

        _query = doc["sent2"] + "".join([f" {k}. {v}" if i else f"{k}. {v}" \
                    for i, (k, v) in enumerate(_option_dict.items())])
        _choices = list(_option_dict.values())
        _gold = doc['label']

        return {
            "query": _query,
            "choices": _choices,
            "gold": _gold,
            "context": doc['sent1']
        }

    # def _process_doc(self, doc):
    #     return {
    #         "query": doc["question"] + "\n" + \
    #                 "".join([f" ({k}) {v}" if i else f"({k}) {v}" \
    #                 for i, (k, v) in enumerate(doc["options"].items())]),
    #         "choices": list(doc["options"].values()),
    #         "gold": ord(doc["answer_idx"])-ord("A"),
    #     }

    def doc_to_text(self, doc):
        return f"Context: {doc['context']}\nQuestion: {doc['query']}\nAnswer:"