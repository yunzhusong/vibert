
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name in ["mnli", "mnli-mm", "rte", "snli",\
       "addonerte", "dpr", "spr", "fnplus", "joci", "mpe",\
       "scitail", "sick",  "QQP", "snlihard", "imdb", "yelp"]:
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def compute_metrics(args, task, preds, out_label_ids):
    return glue_compute_metrics(task, preds, out_label_ids)

