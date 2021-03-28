import pycountry
from pprint import pprint
from sacrebleu import sacrebleu
from lm_eval import metrics
from lm_eval.base import Task, rf

"""
This file implements translation tasks using datasets from WMT conferences, provided by sacrebleu.
Traditionally they are evaluated with BLEU scores. TER and CHRF are other options.

See sacrebleu.DATASETS for all available datasets. There are a lot!
"""
sacrebleu_datasets = sacrebleu.DATASETS


def create_tasks_from_benchmarks(benchmark_dict):
    """Creates a dictionary of tasks from a dict
    :param benchmark_dict: { dataset: [lang_pair, ...], }
    :return: {task_name: task}
        e.g. {wmt14-fr-en: Task, wmt16-de-en: Task}
    """
    return {
        f"{dataset}-{language_pair}": create_translation_task(dataset, language_pair)
        for dataset, language_pairs in benchmark_dict.items()
        for language_pair in language_pairs
    }

########################################
# Tasks
########################################

def create_translation_task(dataset, language_pair):
    class TranslationTask(GeneralTranslationTask):
        def __init__(self):
            super().__init__(dataset, language_pair)
    return TranslationTask

class GeneralTranslationTask(Task):

    # e.g. ("wmt14", "fr-en")
    def __init__(self, sacrebleu_dataset, sacrebleu_language_pair=None):
        self.sacrebleu_dataset = sacrebleu_dataset
        self.sacrebleu_language_pair = sacrebleu_language_pair
        self.src_file = self.ref_file = self.src_data = self.ref_data = None

        super().__init__()

    def download(self):
        # This caches in the users home dir automatically
        self.src_file, self.ref_file = \
            sacrebleu.download_test_set(self.sacrebleu_dataset, self.sacrebleu_language_pair)
        self.src_data, self.ref_data = [
            [line.rstrip() for line in sacrebleu.smart_open(file)]
            for file in (self.src_file, self.ref_file)
        ]

    def has_training_docs(self):
        """Whether the task has a training set"""
        # TODO In the future we could be more discerning. Some more recent tests have train and dev sets
        return False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return [{
            "src": src,
            "ref": ref
        } for src, ref in zip(self.src_data, self.ref_data)]

    def doc_to_text(self, doc):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])
        return f"{src_lang} phrase: " + doc["src"] + f"\n{tar_lang} phrase:"

    def doc_to_target(self, doc):
        # This shows a single target, though there may be multiple targets in a lang test
        return " " + doc["ref"] if isinstance(doc["ref"], str) else doc["ref"][0]

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        return rf.greedy_until(ctx, ["\n"])

    def process_results(self, doc, results):
        # These metrics are corpus-level not sentence level, so we'll hide the
        # results in this dict and compute the corpus score in the aggregate method
        ref_pred = (doc["ref"], results)
        return {
            "bleu": ref_pred,
            "chrf": ref_pred,
            "ter": ref_pred,
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "bleu": metrics.bleu,
            "chrf": metrics.chrf,
            "ter": metrics.ter,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "bleu": True,
            "chrf": True,
            "ter": False,
        }

    def fewshot_description(self):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])
        return f"Translate these {src_lang} phrases to {tar_lang}."

    def __str__(self):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])
        return f"{self.sacrebleu_dataset.upper()} {src_lang} to {tar_lang} Task"


########################################
# Util
########################################


def code_to_language(code):
    # key is alpha_2 or alpha_3 depending on the code length
    language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
    return language_tuple.name

def print_available_tests():
    pprint({ts: sacrebleu.get_langpairs_for_testset(ts) for ts in sacrebleu.get_available_testsets()})


def print_available_pairs():
    list_of_pairs = [sacrebleu.get_langpairs_for_testset(ts) for ts in sacrebleu.get_available_testsets()]
    pairs = set([item for sublist in list_of_pairs for item in sublist])
    pairs = sorted(["-".join(map(code_to_language, pair.split("-"))) for pair in pairs])
    pprint(pairs)
    print(len(pairs))


def main():
    # print(sacrebleu.download_test_set("wmt14", "en-fr"))
    # print_available_tests()
    # sacrebleu.print_test_set("wmt14", "fr-en", "src")

    # # Print number of benchmarks
    # print(sum([
    #     len(sacrebleu.get_langpairs_for_testset(ts))
    #     for ts in sacrebleu.get_available_testsets()
    # ]))

    # Test task dictionary
    # for task, task_class in create_tasks_from_benchmarks(selected_benchmarks).items():
    #     print(task, task_class())
    print_available_pairs()
    pass


if __name__ == "__main__":
    main()


########################################
# Don't mind me...!
########################################

# Available tests as of 2020/02/11
"""
{'iwslt17': ['en-fr',
             'fr-en',
             'en-de',
             'de-en',
             'en-zh',
             'zh-en',
             'en-ar',
             'ar-en',
             'en-ja',
             'ja-en',
             'en-ko',
             'ko-en'],
 'iwslt17/dev2010': ['en-fr', 'fr-en', 'en-de', 'de-en', 'en-zh', 'zh-en'],
 'iwslt17/tst2010': ['en-fr', 'fr-en', 'en-de', 'de-en', 'en-zh', 'zh-en'],
 'iwslt17/tst2011': ['en-fr', 'fr-en', 'en-de', 'de-en', 'en-zh', 'zh-en'],
 'iwslt17/tst2012': ['en-fr', 'fr-en', 'en-de', 'de-en', 'en-zh', 'zh-en'],
 'iwslt17/tst2013': ['en-fr', 'fr-en', 'en-de', 'de-en', 'en-zh', 'zh-en'],
 'iwslt17/tst2014': ['en-fr', 'fr-en', 'en-de', 'de-en', 'en-zh', 'zh-en'],
 'iwslt17/tst2015': ['en-fr', 'fr-en', 'en-de', 'de-en', 'en-zh', 'zh-en'],
 'iwslt17/tst2016': ['en-fr', 'fr-en', 'en-de', 'de-en', 'en-zh', 'zh-en'],
 'mtnt1.1/test': ['en-fr', 'fr-en', 'en-ja', 'ja-en'],
 'mtnt1.1/train': ['en-fr', 'fr-en', 'en-ja', 'ja-en'],
 'mtnt1.1/valid': ['en-fr', 'fr-en', 'en-ja', 'ja-en'],
 'mtnt2019': ['en-fr', 'fr-en', 'en-ja', 'ja-en'],
 'multi30k/2016': ['en-fr', 'en-de', 'en-cs'],
 'multi30k/2017': ['en-fr', 'en-de'],
 'multi30k/2018': ['en-fr', 'en-de'],
 'wmt08': ['cs-en',
           'en-cs',
           'de-en',
           'en-de',
           'es-en',
           'en-es',
           'fr-en',
           'en-fr',
           'hu-en',
           'en-hu'],
 'wmt08/europarl': ['de-en', 'en-de', 'es-en', 'en-es', 'fr-en', 'en-fr'],
 'wmt08/nc': ['cs-en', 'en-cs'],
 'wmt09': ['cs-en',
           'en-cs',
           'de-en',
           'en-de',
           'es-en',
           'en-es',
           'fr-en',
           'en-fr',
           'hu-en',
           'en-hu',
           'it-en',
           'en-it'],
 'wmt10': ['cs-en',
           'en-cs',
           'de-en',
           'en-de',
           'es-en',
           'en-es',
           'fr-en',
           'en-fr'],
 'wmt11': ['cs-en',
           'en-cs',
           'de-en',
           'en-de',
           'fr-en',
           'en-fr',
           'es-en',
           'en-es'],
 'wmt12': ['cs-en',
           'en-cs',
           'de-en',
           'en-de',
           'es-en',
           'en-es',
           'fr-en',
           'en-fr'],
 'wmt13': ['cs-en',
           'en-cs',
           'de-en',
           'en-de',
           'es-en',
           'en-es',
           'fr-en',
           'en-fr',
           'ru-en',
           'en-ru'],
 'wmt14': ['cs-en',
           'en-cs',
           'de-en',
           'en-de',
           'en-fr',
           'fr-en',
           'en-hi',
           'hi-en',
           'en-ru',
           'ru-en'],
 'wmt14/full': ['cs-en',
                'en-cs',
                'de-en',
                'en-de',
                'en-fr',
                'fr-en',
                'en-hi',
                'hi-en',
                'en-ru',
                'ru-en'],
 'wmt15': ['en-fr',
           'fr-en',
           'cs-en',
           'de-en',
           'en-cs',
           'en-de',
           'en-fi',
           'en-ru',
           'fi-en',
           'ru-en'],
 'wmt16': ['cs-en',
           'de-en',
           'en-cs',
           'en-de',
           'en-fi',
           'en-ro',
           'en-ru',
           'en-tr',
           'fi-en',
           'ro-en',
           'ru-en',
           'tr-en'],
 'wmt16/B': ['en-fi'],
 'wmt16/dev': ['en-ro', 'en-tr', 'ro-en', 'tr-en'],
 'wmt16/tworefs': ['en-fi'],
 'wmt17': ['cs-en',
           'de-en',
           'en-cs',
           'en-de',
           'en-fi',
           'en-lv',
           'en-ru',
           'en-tr',
           'en-zh',
           'fi-en',
           'lv-en',
           'ru-en',
           'tr-en',
           'zh-en'],
 'wmt17/B': ['en-fi'],
 'wmt17/dev': ['en-lv', 'en-zh', 'lv-en', 'zh-en'],
 'wmt17/improved': ['en-zh', 'zh-en'],
 'wmt17/ms': ['zh-en'],
 'wmt17/tworefs': ['en-fi'],
 'wmt18': ['cs-en',
           'de-en',
           'en-cs',
           'en-de',
           'en-et',
           'en-fi',
           'en-ru',
           'et-en',
           'fi-en',
           'ru-en',
           'en-tr',
           'tr-en',
           'en-zh',
           'zh-en'],
 'wmt18/dev': ['et-en', 'en-et'],
 'wmt18/test-ts': ['cs-en',
                   'de-en',
                   'en-cs',
                   'en-de',
                   'en-et',
                   'en-fi',
                   'en-ru',
                   'et-en',
                   'fi-en',
                   'ru-en',
                   'en-tr',
                   'tr-en',
                   'en-zh',
                   'zh-en'],
 'wmt19': ['cs-de',
           'de-cs',
           'de-en',
           'de-fr',
           'en-cs',
           'en-de',
           'en-fi',
           'en-gu',
           'en-kk',
           'en-lt',
           'en-ru',
           'en-zh',
           'fi-en',
           'fr-de',
           'gu-en',
           'kk-en',
           'lt-en',
           'ru-en',
           'zh-en'],
 'wmt19/dev': ['lt-en', 'en-lt', 'gu-en', 'en-gu', 'kk-en', 'en-kk'],
 'wmt19/google/ar': ['en-de'],
 'wmt19/google/arp': ['en-de'],
 'wmt19/google/hqall': ['en-de'],
 'wmt19/google/hqp': ['en-de'],
 'wmt19/google/hqr': ['en-de'],
 'wmt19/google/wmtp': ['en-de'],
 'wmt20': ['cs-en',
           'de-en',
           'de-fr',
           'en-cs',
           'en-de',
           'en-iu',
           'en-ja',
           'en-km',
           'en-pl',
           'en-ps',
           'en-ru',
           'en-ta',
           'en-zh',
           'fr-de',
           'iu-en',
           'ja-en',
           'km-en',
           'pl-en',
           'ps-en',
           'ru-en',
           'ta-en',
           'zh-en'],
 'wmt20/dev': ['iu-en',
               'en-iu',
               'ja-en',
               'en-ja',
               'pl-en',
               'en-pl',
               'ta-en',
               'en-ta'],
 'wmt20/robust/set1': ['en-ja', 'en-de'],
 'wmt20/robust/set2': ['en-ja', 'ja-en'],
 'wmt20/robust/set3': ['de-en'],
 'wmt20/tworefs': ['de-en', 'en-de', 'en-zh', 'ru-en', 'zh-en']}
"""