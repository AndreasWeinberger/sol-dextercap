import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from ignite.handlers import ModelCheckpoint, Checkpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from .models import BlockNet
from .datasets import BlockCodeDataset

from torcheval.metrics.functional import binary_f1_score, binary_recall, binary_precision

import numpy as np

from typing import Tuple

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default=None, type=str, required=True, help='Path to the labels.json file containing "image" fields and annotated data')
    parser.add_argument('--out', default='./ckpts/0001/', type=str)
    parser.add_argument('--lr', default=0.001, type=float, required=True)
    parser.add_argument('--batch-size', default=512, type=int, required=True)
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('--max-epochs', default=400, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--debug', default=False, type=bool)

    args = parser.parse_args()
    args.dataset_folder, args.labels = os.path.split(args.labels)

    dataset = BlockCodeDataset(args.dataset_folder, args.labels, size=128*500, train=True, augment_image=args.augment, debugging=args.debug)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    val_loader = DataLoader(
        BlockCodeDataset(args.dataset_folder, args.labels, size=1280, train=False, augment_image=False, debugging=args.debug),
        batch_size=args.batch_size, shuffle=False, num_workers=int(args.num_workers / 2), persistent_workers=args.num_workers > 0
    )

    model = BlockNet(label0_chars=len(dataset.label_characters_0), label1_chars=len(dataset.label_characters_1), blk_dirs=5).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    def loss_func(input: Tuple[torch.Tensor], target: Tuple[torch.Tensor]):
        label_0, label_1, blk_dir = input
        label_0_gt, label_1_gt, blk_dir_gt = target
        label_0_loss = nn.functional.cross_entropy(label_0.sigmoid(), label_0_gt)
        label_1_loss = nn.functional.cross_entropy(label_1.sigmoid(), label_1_gt)
        blk_dir_loss = nn.functional.cross_entropy(blk_dir.sigmoid(), blk_dir_gt)

        l = label_0_loss + label_1_loss + blk_dir_loss
        return l

    def accuracy_func(input: torch.Tensor, target: torch.Tensor):
        pred = input.argmax(dim=-1)
        targ = target.argmax(dim=-1)

        a = (pred == targ).sum() / pred.numel()

        return a

    def f1_score_func(input: torch.Tensor, target: torch.Tensor):
        pred = input > 0.5
        targ = target > 0.5

        f1 = binary_f1_score(pred.view(-1), targ.view(-1))

        return f1

    def recall_func(input: torch.Tensor, target: torch.Tensor):
        pred = input > 0.5
        targ = target > 0.5

        r = binary_recall(pred.view(-1), targ.view(-1))

        return r

    def prec_func(input: torch.Tensor, target: torch.Tensor):
        pred = input > 0.5
        targ = target > 0.5

        p = binary_precision(pred.view(-1), targ.view(-1))

        return p

    def valid_loss(input: torch.Tensor, target: torch.Tensor):
        l = nn.functional.cross_entropy(input.sigmoid(), target)
        return l

    # criterion = nn.MSELoss()
    criterion = loss_func

    trainer = create_supervised_trainer(model, optimizer, criterion, device)

    val_metrics = {
        "accuracy": Loss(accuracy_func),
        "loss": Loss(valid_loss),
        "f1_score": Loss(f1_score_func),
        "recall": Loss(recall_func),
        "precision": Loss(prec_func)
    }

    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    # train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    log_interval = 1
    tb_logger = TensorboardLogger(log_dir="logger/block")

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")
        # train_evaluator.run(val_loader)
        # metrics = train_evaluator.state.metrics
        # print(f"Train Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f} Avg f1_score: {metrics['f1_score']:.2f} Avg recall: {metrics['recall']:.2f} Avg precision: {metrics['precision']:.2f}")

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(trainer):
    #     train_evaluator.run(train_loader)
    #     metrics = train_evaluator.state.metrics
    #     print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f} Avg f1_score: {metrics['f1_score']:.2f} Avg recall: {metrics['recall']:.2f}")

    #     tb_logger.writer.flush()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        # print(f'Scheduler LR: {scheduler.get_last_lr()}')
        # scheduler.step(val_evaluator.state.metrics['loss'])
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f} Avg f1_score: {metrics['f1_score']:.2f} Avg recall: {metrics['recall']:.2f} Avg precision: {metrics['precision']:.2f}")

        tb_logger.writer.flush()

    def score_function(engine):
        return engine.state.metrics["f1_score"]

    model_checkpoint = ModelCheckpoint(
        os.path.join(args.out, 'block'),
        n_saved=1,
        filename_prefix="lr_{}_bs_{}".format(args.lr, args.batch_size),
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    best_checkpoint = ModelCheckpoint(
        os.path.join(args.out, 'block'),
        n_saved=3,
        filename_prefix="best_lr_{}_bs_{}".format(args.lr, args.batch_size),
        score_function=score_function,
        score_name="f1_score",
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
