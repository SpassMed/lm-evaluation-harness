# https://huggingface.co/wanglab/ClinicalCamel-70B/blob/main/TaskFiles/usmle_self_eval_step1.py

from lm_eval.base import MultipleChoiceTask
from datasets import concatenate_datasets

class JMLESingleAnswer(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "SpassMedAI/japanse_books_medical_qna_single_answer"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return []

    def validation_docs(self):
        return []

    def test_docs(self):
        return map(self._process_doc, concatenate_datasets([self.dataset["test"], self.dataset["train"]]))

    def _process_doc(self, doc):
        filtered_options = {key: value for key, value in doc['options_dict'].items() if value is not None}
        answer = doc["Answer 1"].lower()

        return {
            "query": doc["question"] + "\n" + \
                    "".join([f" ({k}) {v}" if i else f"({k}) {v}" \
                    for i, (k, v) in enumerate(filtered_options.items())]),
            "choices": list(filtered_options.values()),
            "gold": ord(answer)-ord("a"),
        }

    def doc_to_text(self, doc):
        return f"Question: {doc['query']}\nAnswer:"