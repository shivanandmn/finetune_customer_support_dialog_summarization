import evaluate


def mean(x):
    return sum(x) / len(x)


class Metrics:
    def __init__(self):
        self.rouge_scorer = evaluate.load("rouge")
        self.bleu_scorer = evaluate.load("bleu")
        self.bertscore_scorer = evaluate.load("bertscore")

    def eval_bertscore(self, predictions, references):
        result = self.bertscore_scorer.compute(
            predictions=predictions, references=references, lang="en"
        )
        result = {
            "precision": mean(result["precision"]),
            "recall": mean(result["recall"]),
            "f1": mean(result["f1"]),
        }
        return {"bertscore__" + k: v for k, v in result.items()}

    def eval_rouge(self, predictions, references):
        result = self.rouge_scorer.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True,
            use_stemmer=True,
        )
        return {"rouge__" + k: v for k, v in result.items()}

    def eval_bleu(self, predictions, references):
        result = self.bleu_scorer.compute(
            predictions=predictions, references=[[x] for x in references]
        )
        return {
            "bleu__" + k: (v if not isinstance(v, list) else mean(v))
            for k, v in result.items()
        }

    def score(self, prediction, reference):
        methods_starting_with_eval = [
            getattr(self, method)
            for method in dir(self)
            if method.startswith("eval") and callable(getattr(self, method)) #Had fun here
        ]
        results = {}
        for method in methods_starting_with_eval:
            res = method(predictions=prediction, references=reference)
            results.update(res)
        return results
