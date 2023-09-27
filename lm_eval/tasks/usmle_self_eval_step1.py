# https://huggingface.co/wanglab/ClinicalCamel-70B/blob/main/TaskFiles/usmle_self_eval_step1.py

from lm_eval.base import MultipleChoiceTask


class usmle_self_eval_step1(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "augtoma/usmle_self_eval_step1"
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
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        filtered_options = {key: value for key, value in doc['options'].items() if value is not None}

        return {
            "query": doc["question"] + "\n"
            + "".join([f" ({k}) {v}" if i else f"({k}) {v}"
                       for i, (k, v) in enumerate(filtered_options.items())]),
            "choices": list(filtered_options.values()),
            "gold": ord(doc["answer_idx"]) - ord("A"),
        }

    def doc_to_text(self, doc):
        """<s> [INST] Context: {}. Question: {}.[/INST]\nAnswer: {}</s>"""
        #return f"<s> [INST] Context: {doc['context']}\nQuestion: {doc['query']}\nAnswer:"
        # separate context and question
        _arr = doc["query"].split(".")
        _context = ".".join(_arr[:-1]) + "." # add back the period
        _question = _arr[-1] # last element
        return f"<s> [INST] Context: {_context}\nQuestion: {_question}\n[/INST]</s>Answer:"
        #return f"Question: {doc['query']}\nAnswer:"
