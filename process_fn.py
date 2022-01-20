import pandas as pd
import seaborn as sns
from sklearn import manifold
from matplotlib import pyplot as plt

import pdb
import logging
import os
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from prior_wd_optim import PriorWD

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

from data_builder import load_and_cache_examples
from evaluation_metric import compute_metrics

#from others import _unfreeze_specified_params, _freeze_specified_params
from others import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def _unfreeze_specified_params(model, train_only=None):
    if train_only is not None:
        for name, sub_module in model.named_modules():
            if train_only in name:
                for param in sub_module.parameters():
                    param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

def _freeze_specified_params(model, fix_model=None):
    if fix_model is not None:
        fix_layers = fix_model.split(" ")

        for fix_layer in fix_layers:
            for name, sub_module in model.named_modules():
                if fix_layer in name:
                    for param in sub_module:
                        param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False

def _freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False

def evaluate(args, model, tokenizer, prefix="", sampling_type="argmax", save_results=True, epoch=0):
    results = {}
    all_preds = {}
    all_zs = {}
    all_labels = {}
    for eval_task in args.eval_tasks:
        for eval_type in args.eval_types:
            #print("Evaluating on "+eval_task+" with eval_type ", eval_type)
            if eval_type=="test" and epoch < args.num_train_epochs-1:
                continue

            eval_dataset, num_classes = load_and_cache_examples(args, eval_task, tokenizer, eval_type)
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # multi-gpu eval
            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            zs = []
            #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            X = []
            Y = []
            for batch in eval_dataloader:
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                    no_label = True if (eval_type == "test" and eval_task in args.glue_tasks) else False
                    if no_label:
                        inputs["labels"] = None
                    outputs = model(**inputs, sampling_type=sampling_type)
                    tmp_eval_loss, logits = outputs["loss"], outputs["logits"]
                    X.append(outputs["z"])
                    Y.append(batch[3])

                    #eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = None if no_label else inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = None if no_label else np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

                zs.append(outputs["z"])

            if args.plotting_tsne:
                X = torch.cat(X).cpu()
                Y = torch.cat(Y).cpu()

                X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)
                x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
                #plt.figure(figsize=(8, 8))
                #plt.figure()

                #df = pandas.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=Y))
                #df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')

                #for i in range(X_norm.shape[0]):
                #    plt.plot(X_norm[i, 0], X_norm[i, 1], '*', color=plt.cm.Set1([Y[i]])) #, fontdict={'weight': 'bold', 'size': 9})

                #plt.xticks([])
                #plt.yticks([])
                #plt.savefig('{}/tSNE.png'.format(args.output_dir))

                # Visualization
                df = pd.DataFrame()
                df["y"] = Y
                df["comp-1"] = X_norm[:,0]
                df["comp-2"] = X_norm[:,1]

                sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),palette=sns.color_palette("hls", 2),data=df).set(title="T-SNE projection")
                sns_plot[0].figure.savefig("{}/sns.png".format(args.output_dir))


            #eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)

                # binarize the labels and predictions if needed.
                if num_classes == 2 and args.binarize_eval:
                    preds = binarize_preds(preds)
                    out_label_ids = binarize_preds(out_label_ids)

            elif args.output_mode == "regression":
                preds = np.squeeze(preds)

            all_preds[eval_task + "_" + eval_type] = preds
            all_zs[eval_task+"_"+eval_type] = torch.cat(zs)
            all_labels[eval_task+"_"+eval_type] = out_label_ids

            no_label = True if (eval_type == "test" and eval_task in args.glue_tasks) else False
            if not no_label:
                temp = compute_metrics(args, eval_task, preds, out_label_ids)
                if len(args.eval_tasks) > 1:
                    # then this is for transfer and we need to know the name of the datasets.
                    temp = {eval_task+"_"+k + '_' + eval_type: v for k, v in temp.items()}
                else:
                    temp = {k + '_' + eval_type: v for k, v in temp.items()}
                results.update(temp)
                print(results)
            else:
                write_in_glue_format(args, all_preds, eval_type, epoch=epoch)

    # In case of glue, results is empty.
    #if args.outputfile is not None and save_results and results:
    #    write_to_csv(results, args, args.outputfile)
    return results, all_preds, all_zs, all_labels

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    def save_model(args, global_step, model, optimizer, scheduler, tokenizer):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.fix_model is not None:
        """ Adjsut the trainable parameters by train_only"""
        _freeze_specified_params(model, args.fix_model)
        #_unfreeze_all_params(model, args.train_only)
        
    if args.train_only is not None:
        _freeze_all_params(model)
        _unfreeze_specified_params(model, args.train_only)

    # Show number of parameters
    all_param_num = sum([p.nelement() for p in model.parameters()])
    trainable_param_num = sum([
        p.nelement()
        for p in model.parameters()
        if p.requires_grad == True
    ])
    print(f"All parameters : {all_param_num}")
    print(f"Trainable parameters : {trainable_param_num}")

    tb_writer = SummaryWriter(args.output_dir+'/log')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.prior_weight_decay: # I am just addding this because revisiting bert few-sample added it. should be checked.
       optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,\
            correct_bias=not args.use_bertadam, weight_decay=args.weight_decay)
    else:
       optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.prior_weight_decay:
       optimizer = PriorWD(optimizer, use_prior_wd=args.prior_weight_decay) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,\
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        #epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False,
    )
    stop_training = 0
    set_seed(args)  # Added here for reproductibility
    for epoch in train_iterator:
        if stop_training == 2:
            break;
        epoch = epoch + 1
        args.epoch = epoch
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        #for step, batch in enumerate(epoch_iterator):
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs, epoch=epoch)
            loss = outputs["loss"]["loss"]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 1:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print("loss: {:.4f}".format(loss.item()))
                    tb_writer.add_scalar("train_loss", loss.item(), global_step)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #if args.evaluate_during_training:
                    results, _, _, _ = evaluate(args, model, tokenizer)

                    for k, v in results.items():
                        tb_writer.add_scalar(k, v, global_step)

                #if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(args, global_step, model, optimizer, scheduler, tokenizer)

            #if args.max_steps > 0 and global_step > args.max_steps:
            #    epoch_iterator.close()
            #    break
        #if args.max_steps > 0 and global_step > args.max_steps:
        #    train_iterator.close()
        #    break

        # Evaluates the model after each epoch.
        if args.evaluate_after_each_epoch:
            results, _, _, _ = evaluate(args, model, tokenizer, epoch=epoch)
            save_model(args, global_step, model, optimizer, scheduler, tokenizer)
    #if args.local_rank in [-1, 0]:
    #    tb_writer.close()

    return global_step, tr_loss / global_step

