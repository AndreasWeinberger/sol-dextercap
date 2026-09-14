import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from ignite.handlers import ModelCheckpoint, Checkpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from torcheval.metrics.functional import binary_f1_score, binary_recall

from .models import UNet
from .datasets import MarkerDataset

import numpy as np
import os
import matplotlib.pyplot as plt

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default=None, type=str, required=True, help='Path to the labels.json file containing "image" fields and annotated data')
    parser.add_argument('--out', default='./ckpts/0001/', type=str)
    parser.add_argument('--lr', default=0.001, type=float, required=True)
    parser.add_argument('--batch-size', default=1024, type=int, required=True)
    parser.add_argument('--block-size', default=96, type=int)
    parser.add_argument('--margin', default=50, type=int)
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('--max-epochs', default=400, type=int)
    parser.add_argument('--max-num-markers', default=100, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--debug', default=False, type=bool)

    args = parser.parse_args()

    model = UNet(output_channel=2).to(device)

    args.dataset_folder, args.labels = os.path.split(args.labels)

    train_loader = DataLoader(
        MarkerDataset(args.dataset_folder, args.labels, block_size=args.block_size, margin=args.margin, size=128*500, train=True, max_num_markers=args.max_num_markers, 
                      output_mask_image=True, output_line_mask_image=True, augment_image=args.augment), 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=args.num_workers > 0
    )

    val_loader = DataLoader(
        MarkerDataset(args.dataset_folder, args.labels, block_size=args.block_size, margin=args.margin, size=1280, train=False, max_num_markers=args.max_num_markers, 
                      output_mask_image=True, output_line_mask_image=True, augment_image=False), 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=args.num_workers > 0
    )

    # Test model with 0.74% acc trained with eps 1e-7
    # , weight_decay=0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    def loss_func(input: torch.Tensor, target: torch.Tensor):
        heatmap_loss = nn.functional.mse_loss(input, target*10)
        return heatmap_loss

    trainer = create_supervised_trainer(model, optimizer, loss_func, device)

    def accuracy_func(input: torch.Tensor, target: torch.Tensor):
        pred = input > 5
        targ = target > 0.5
        f1 = binary_f1_score(pred.view(-1), targ.view(-1))        
        return f1

    def recall_func(input: torch.Tensor, target: torch.Tensor):
        pred = input > 5
        targ = target > 0.5
        
        return binary_recall(pred.view(-1), targ.view(-1))
    
    val_metrics = {
        "accuracy": Loss(accuracy_func),
        "recall": Loss(recall_func),
        "loss": Loss(loss_func)
    }

    # train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    log_interval = 1
    tb_logger = TensorboardLogger(log_dir="logger/conv")
        
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f} Avg recall: {metrics['recall']:.2f}")
        
        tb_logger.writer.flush()


    def score_function(engine):
        return engine.state.metrics["accuracy"]


    model_checkpoint = ModelCheckpoint(
        os.path.join(args.out,'conv'),
        n_saved=1,
        filename_prefix=f'lr_{args.lr}_bs_{args.batch_size}',
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    best_checkpoint = ModelCheckpoint(
        os.path.join(args.out,'conv'),
        n_saved=3,
        filename_prefix=f'best_lr_{args.lr}_bs_{args.batch_size}',
        score_function=score_function,
        score_name="acc",
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )
    
    val_evaluator.add_event_handler(Events.COMPLETED, best_checkpoint, {"model": model})
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, 
                                    {'model': model, 'optimizer': optimizer, 'trainer': trainer})


    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    for tag, evaluator in [
        # ("training", train_evaluator), 
        ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )
        
    # resume
    if not args.checkpoint is None:
        checkpoint_fp = args.checkpoint
        checkpoint = torch.load(checkpoint_fp, map_location=device) 
        Checkpoint.load_objects(to_load={
            'model': model, 
            'optimizer': optimizer, 
            'trainer': trainer
            }, checkpoint=checkpoint) 

    trainer.run(train_loader, max_epochs=args.max_epochs)

    tb_logger.close()