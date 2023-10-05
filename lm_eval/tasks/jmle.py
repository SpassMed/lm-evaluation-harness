# https://huggingface.co/wanglab/ClinicalCamel-70B/blob/main/TaskFiles/usmle_self_eval_step1.py

from lm_eval.base import MultipleChoiceTask
from datasets import concatenate_datasets
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.base import Task, rf, mean

class JMLESingleAnswer(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "SpassMedAI/jpn-med-exam-IgakuQA"
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
        return map(self._process_doc, self.dataset["train"])

    @staticmethod
    def assign_alphabet(choices):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        return dict(zip(alphabet, choices))

    def _process_doc(self, doc):

        choices = doc["choices"]
        choices = self.assign_alphabet(choices)
        filtered_options = {key: value for key, value in choices.items() if (value is not None) and (len(value) > 0)}

        # doc["answer"] = [e, f] format, can be a number as well; need to make it into a single string separated by comma, if number, then just convert to string
        answer = doc["answer"]
        answer = ",".join([str(a) for a in answer]) if isinstance(answer, list) else str(answer)
        _book = doc["book_name"]
        _section = doc["section"]

        return {
            "query": doc["problem_text"] + "\n" + \
                    "".join([f" ({k}) {v}" if i else f"({k}) {v}" \
                    for i, (k, v) in enumerate(filtered_options.items())]),
            "choices": list(filtered_options.values()),
            "gold": answer,
            "points": int(doc["points"]),
            "section": _section,
            "book": _book,
        }

    def doc_to_target(self, doc):
        return doc["gold"]

    @staticmethod
    def compute_scores(gold_list, pred):
        """
        There are multiple correct answers and all of them must be selected.
        So take top-k answers based on length od gold answers and compute exact match.
        """
        print(gold, pred)
        acc_list = []
        for gold, pred in zip(gold_list, pred):
            gold = gold.split(",")
            pred = pred.split(",")
            acc = 0
            for g in gold:
                if g in pred:
                    acc += 1
            acc_list.append(acc/len(gold))
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """

        gold_list = doc["gold"]
        points = doc["points"]

        # separate alphabets from gold_list
        gold_list = gold_list.split(",")

        # print(results)
        # assign ordered alphabet to results
        results = self.assign_alphabet(results)
        print(results)

        # select top-k index based on value
        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
        
        print(results)
        result_topk = list(results.keys())[:len(gold_list)]
        print(result_topk)

        # intersection between gold_list and result_topk is acc
        _common_options = set(gold_list).intersection(result_topk)
        acc = int(len(_common_options) == len(gold_list))

        _section = doc["section"]
        _book = doc["book"]

        return {
            f"acc_{_section}": acc,
            "acc": acc
            # "points": sum(points_list)/len(points_list),
        }

    def doc_to_text(self, doc):
        return f"Question: {doc['query']}\nAnswer:"

    def higher_is_better(self):
        return {
            "acc_A": True,
            "acc_B": True,
            "acc_C": True,
            "acc_D": True,
            "acc_E": True,
            "acc_F": True,
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc_A": mean,
            "acc_B": mean,
            "acc_C": mean,
            "acc_D": mean,
            "acc_E": mean,
            "acc_F": mean,
            "acc": mean,
        }