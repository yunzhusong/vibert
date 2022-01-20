import os
from os.path import join
import torch
from torch.utils.data import TensorDataset
from data import processors, output_modes
from data import InputExample, InputFeatures


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    output_mode = None,
    mask_padding_with_zero=True,
    no_label = False):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    def get_padded(input_ids, token_type_ids, attention_mask, max_length, pad_token,
            pad_token_segment_id, mask_padding_with_zero):
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        return input_ids, attention_mask, token_type_ids


    if task is not None:
        processor = processors[task](args.spurious_correlation)
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}
 
    def label_from_example(example: InputExample):
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_ids, attention_mask, token_type_ids = get_padded(input_ids, token_type_ids,\
          attention_mask, max_length, pad_token, pad_token_segment_id, mask_padding_with_zero)

        label = label_from_example(example) if not no_label else -1

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    return features



def load_and_cache_examples(args, task, tokenizer, eval_type):
    data_dir = args.task_to_data_dir[task]
    #if args.local_rank not in [-1, 0] and not evaluate:
    #    torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if task in args.nli_tasks:
        processor = processors[task](args.task_to_data_dir[task])
    else:
        processor = processors[task](args.spurious_correlation)

    output_mode = output_modes[task]

    label_list = processor.get_labels()
    if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]


    if eval_type == "train":
        if args.sample_train:
            cached_features_file = join(data_dir, 'cached_type_{}_task_{}_sample_train_{}_num_samples_{}_model_{}_data_seed_{}'.\
                                format(eval_type, task, args.sample_train, args.num_samples,
                                       list(filter(None, args.model_name_or_path.split('/'))).pop(), args.data_seed))
        else:
            # here data_seed has no impact.
            cached_features_file = join(data_dir,
                                        'cached_type_{}_task_{}_sample_train_{}_num_samples_{}_model_{}'. \
                                        format(eval_type, task, args.sample_train, args.num_samples,
                                               list(filter(None, args.model_name_or_path.split('/'))).pop()))
    else:
        cached_features_file = join(data_dir, 'cached_type_{}_task_{}_model_{}'. \
                                    format(eval_type, task, list(filter(None, args.model_name_or_path.split('/'))).pop()))

    #if os.path.exists(cached_features_file):
    #    #logger.info("Loading features from cached file %s", cached_features_file)
    #    features = torch.load(cached_features_file)
    #else:
    if eval_type == "train":
        if args.sample_train:
            data_dir = join(data_dir, "sampled_datasets", "seed_"+str(args.data_seed), str(args.num_samples)) # sampled: for old version.
        examples = (processor.get_train_examples(data_dir))
    elif eval_type == "test":
        examples = (processor.get_dev_examples(data_dir))
    elif eval_type == "dev":
        examples = (processor.get_validation_examples(data_dir))

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        output_mode=output_mode,
        no_label=True if (eval_type == "test" and task in args.glue_tasks) else False
    )
    #print("Saving features into cached file %s", cached_features_file)
    #torch.save(features, cached_features_file)


    if args.local_rank == 0 and eval_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset, processor.num_classes

