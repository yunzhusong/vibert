import os
from os.path import join
import csv
import torch
import random
import numpy as np
from data import processors, output_modes

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

def binarize_preds(preds):
    # maps the third label (neutral one) to first, which is contradiction.
    preds[preds == 2] = 0
    return preds

def write_to_csv(scores, params, outputfile):
    """This function writes the parameters and the scores with their names in a
    csv file."""
    # creates the file if not existing.
    file = open(outputfile, 'a')
    # If file is empty writes the keys to the file.
    params_dict = vars(params)
    if os.stat(outputfile).st_size == 0:
        # Writes the configuration parameters
        for key in params_dict.keys():
            file.write(key+";")
        for i, key in enumerate(scores.keys()):
            ending = ";" if i < len(scores.keys())-1 else ""
            file.write(key+ending)
        file.write("\n")
    file.close()

    # Writes the values to each corresponding column.
    with open(outputfile, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)

    # Iterates over the header names and write the corresponding values.
    with open(outputfile, 'a') as f:
        for i, key in enumerate(headers):
            ending = ";" if i < len(headers)-1 else ""
            if key in params_dict:
                f.write(str(params_dict[key])+ending)
            elif key in scores:
                f.write(str(scores[key])+ending)
            else:
                raise AssertionError("Key not found in the given dictionary")
        f.write("\n")

def write_in_glue_format(args, preds, eval_type, epoch):
    def label_from_example(label, output_mode, label_map):
        if output_mode == "classification":
            return label_map[label]
        elif output_mode == "regression":
            return float(label)
        raise KeyError(output_mode)

    def write_labels(labels, outpath):
        with open(outpath, 'wt') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['index', 'prediction'])
            for i, label in enumerate(labels):
                tsv_writer.writerow([i, label])

    task_to_filename={"rte": "RTE", "sts-b":"STS-B", "mrpc": "MRPC"}
    task = args.eval_tasks[0]
    preds = preds[task+"_"+eval_type]
    processor = processors[task]()
    label_list = processor.get_labels()
    label_map = {i: label for i, label in enumerate(label_list)}
    output_mode = output_modes[task]
    labels = [label_from_example(label, output_mode, label_map) for label in preds]
    write_labels(labels, join(args.output_dir, task_to_filename[task]+"_"+eval_type+".tsv"))
    #write_labels(labels, join(args.output_dir, task_to_filename[task]+"_"+eval_type+"_epoch_"+str(epoch)+".tsv"))

