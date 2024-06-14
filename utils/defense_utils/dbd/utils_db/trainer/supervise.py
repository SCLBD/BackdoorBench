import time

import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from .log import AverageMeter, tabulate_epoch_meter, tabulate_step_meter


def poison_train(model, loader, criterion, optimizer, logger, amp=False):
    loss_meter = AverageMeter("loss")
    poison_loss_meter = AverageMeter("poison loss")
    clean_loss_meter = AverageMeter("clean loss")
    acc_meter = AverageMeter("acc")
    poison_acc_meter = AverageMeter("poison acc")
    clean_acc_meter = AverageMeter("clean acc")
    meter_list = [
        loss_meter,
        poison_loss_meter,
        clean_loss_meter,
        acc_meter,
        poison_acc_meter,
        clean_acc_meter,
    ]

    model.train()
    gpu = next(model.parameters()).device
    if amp:
        scaler = GradScaler()
    else:
        scaler = None
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        data = batch["img"].cuda(gpu, non_blocking=True)
        target = batch["target"].cuda(gpu, non_blocking=True)

        optimizer.zero_grad()
        if amp:
            with autocast():
                output = model(data)
                criterion.reduction = "none"
                raw_loss = criterion(output, target)
                criterion.reduction = "mean"
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            criterion.reduction = "none"
            raw_loss = criterion(output, target)
            criterion.reduction = "mean"
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((torch.sum(truth).float() / len(truth)).item())
        poison_idx = torch.nonzero(batch["poison"], as_tuple=True)
        clean_idx = torch.nonzero(batch["poison"] - 1, as_tuple=True)
        # Not every batch contains poison data.
        if len(poison_idx[0]) != 0:
            poison_loss_meter.update(torch.mean(raw_loss[poison_idx]).item())
            poison_acc_meter.update(
                (torch.sum(truth[poison_idx]).float() / len(truth[poison_idx])).item()
            )
        clean_loss_meter.update(torch.mean(raw_loss[clean_idx]).item())
        clean_acc_meter.update(
            (torch.sum(truth[clean_idx]).float() / len(truth[clean_idx])).item()
        )

        tabulate_step_meter(batch_idx, len(loader), 3, meter_list, logger)

    logger.info("Poison training summary:")
    tabulate_epoch_meter(time.time() - start_time, meter_list, logger)
    result = {m.name: m.total_avg for m in meter_list}

    return result


def test(model, loader, criterion, logger):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    meter_list = [loss_meter, acc_meter]

    model.eval()
    gpu = next(model.parameters()).device
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        data = batch["img"].cuda(gpu, non_blocking=True)
        target = batch["target"].cuda(gpu, non_blocking=True)
        with torch.no_grad():
            output = model(data)
        criterion.reduction = "mean"
        loss = criterion(output, target)

        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((torch.sum(truth).float() / len(truth)).item())

        tabulate_step_meter(batch_idx, len(loader), 2, meter_list, logger)

    logger.info("Test summary:")
    tabulate_epoch_meter(time.time() - start_time, meter_list, logger)
    result = {m.name: m.total_avg for m in meter_list}

    return result
