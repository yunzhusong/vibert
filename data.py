import json
import copy
import csv
import os
from os.path import join
import logging
import numpy as np
import pdb


logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, encoding="utf-8-sig"):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding=encoding) as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))



class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def __init__(self, spurious_correlation=None):
        self.num_classes = 2  
        self.spurious_correlation = spurious_correlation

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    def __init__(self, spurious_correlation=None):
        # It joins the other two label to one label.
        self.num_classes = 3 
        self.spurious_correlation = spurious_correlation

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_mismatched")

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "dev_matched.tsv"))
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                   InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv"))
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)


class ImdbProcessor(DataProcessor):
    """Processor for the IMDB dataset."""
    def __init__(self, spurious_correlation=None):
        self.num_classes = 2
        self.spurious_correlation = spurious_correlation

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class YelpProcessor(DataProcessor):
    """Processor for the Yelp dataset."""
    def __init__(self, spurious_correlation=None):
        self.num_classes = 5
        self.spurious_correlation = spurious_correlation

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label = line[0]
            text_a = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    def __init__(self, spurious_correlation=None):
        self.num_classes = 1 
        self.spurious_correlation = spurious_correlation

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = -1 if set_type=="test" else line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""
    def __init__(self, spurious_correlation=None):
        self.num_classes = 2 
        self.spurious_correlation = spurious_correlation

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_validation_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "validation.tsv")), "validation")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
    def __init__(self, spurious_correlation=None):
        self.num_classes = 2 
        self.spurious_correlation = spurious_correlation
        print("Number of class: {}".format(self.num_classes))
        print("How to build spurious correlation: {}".format(spurious_correlation))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")


    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if self.spurious_correlation=='hypo_only':
                text_a = line[2]
                text_b = None
            elif self.spurious_correlation=='premise_only':
                text_a = line[1]
                text_b = None
            elif self.spurious_correlation=='exchange':
                text_a = line[2]
                text_b = line[1]
            else:
                text_a = line[1]
                text_b = line[2]
            label = -1 if set_type =="test" else line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""
    def __init__(self, spurious_correlation=None):
        self.num_classes = 3
        self.spurious_correlation = spurious_correlation

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        print("test set")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_validation_examples(self, data_dir):
        print("dev set")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class NliProcessor(DataProcessor):
    """Processor for the dataset of the format of SNLI
    (InferSent version), could be 2 or 3 classes."""
    # We use get_labels() class to convert the labels to indices,
    # later during the transfer it will be problematic if the labels
    # are not the same order as the SNLI/MNLI so we return the whole
    # 3 labels, but for getting the actual number of classes, we use
    # self.num_classes.

    def __init__(self, data_dir):
        # We assume there is a training file there and we read labels from there.
        labels = [line.rstrip() for line in open(join(data_dir, 'labels.train'))]
        self.labels = list(set(labels))
        labels = ["contradiction", "entailment", "neutral"]
        ordered_labels = [] 
        for l in labels:
            if l in self.labels:
                ordered_labels.append(l)
        self.labels = ordered_labels
        self.num_classes = len(self.labels)

    def get_dev_labels(self, data_dir):
        labels = [line.rstrip() for line in open(join(data_dir, 'labels.test'))]
        return np.array(labels)

    def get_validation_examples(self, data_dir):
        return self._create_examples(data_dir, "dev")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return  ["contradiction", "entailment", "neutral"] #self.labels

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        s1s = [line.rstrip() for line in open(join(data_dir, 's1.'+set_type))]
        s2s = [line.rstrip() for line in open(join(data_dir, 's2.'+set_type))]
        labels = [line.rstrip() for line in open(join(data_dir, 'labels.'+set_type))]
 
        examples = []
        for (i, line) in enumerate(s1s):
            guid = "%s-%s" % (set_type, i)
            text_a = s1s[i]
            text_b = s2s[i]
            label = labels[i]
            # In case of hidden labels, changes it with entailment.
            if label == "hidden":
               label = "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


processors = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "rte": RteProcessor,
    "snli": SnliProcessor,
    "addonerte": NliProcessor,
    "dpr": NliProcessor,
    "spr": NliProcessor,
    "fnplus": NliProcessor,
    "joci": NliProcessor,
    "mpe": NliProcessor,
    "scitail": NliProcessor,
    "sick": NliProcessor,
    "QQP": NliProcessor,
    "snlihard": NliProcessor,
    "imdb": ImdbProcessor,
    "yelp": YelpProcessor,
}

output_modes = {
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "rte": "classification",
    "snli":"classification",
    "addonerte": "classification",
    "dpr": "classification", 
    "spr":"classification", 
    "fnplus": "classification", 
    "joci": "classification", 
    "mpe": "classification", 
    "scitail": "classification", 
    "sick": "classification", 
    "QQP": "classification",
    "snlihard": "classification", 
    "imdb": "classification",
    "yelp": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "mnli": 3,
    "mnli-mm": 3,
    "mrpc": 2,
    "sts-b": 1,
    "qqp": 2,
    "rte": 2,
    "snli": 3,
    "imdb": 2,
    "yelp": 5,
 }

