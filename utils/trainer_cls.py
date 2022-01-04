import sys, logging
sys.path.append('../')
import random
from pprint import pformat
from collections import deque
from typing import *
import numpy as np
import torch

try:
    import wandb
except:
    pass


class MyModelTrainerCLS():
    def __init__(self, model):
        self.model = model

    def init_or_continue_train(self,
                               train_data,
                               end_epoch_num,
                               criterion,
                               optimizer,
                               scheduler,
                               device,
                               continue_training_path: Optional[str] = None,
                               only_load_model: bool = False,
                               ) -> None:
        # 这里traindata只是为了CyclicLR所以才设置的
        model = self.model

        model.to(device)
        model.train()

        # train and update

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if continue_training_path is not None:
            start_epoch, start_batch = self.load_from_path(continue_training_path, device, only_load_model)
            if (start_epoch is None) or (start_batch is None):
                self.start_epochs, self.end_epochs = 0, end_epoch_num
                self.start_batch = 0
            else:
                batch_num = len(train_data)
                self.start_epochs, self.end_epochs = start_epoch + ((start_batch + 1)//batch_num), end_epoch_num
                self.start_batch = (start_batch + 1) % batch_num
        else:
            self.start_epochs, self.end_epochs = 0, end_epoch_num
            self.start_batch = 0

        logging.info(f'All setting done, train from epoch {self.start_epochs} batch {self.start_batch} to epoch {self.end_epochs}')

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def save_all_state_to_path(self,
                               path: str,
                               epoch: Optional[int] = None,
                               batch: Optional[int] = None,
                               only_model_state_dict: bool = False) -> None:

        save_dict = {
            'epoch_num_when_save': epoch,
            'batch_num_when_save': batch,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'model_state_dict': self.get_model_params(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'criterion_state_dict': self.criterion.state_dict(),
        } \
            if only_model_state_dict == False else self.get_model_params()

        torch.save(
            save_dict,
            path,
        )

    def load_from_path(self,
                       path: str,
                       device,
                       only_load_model: bool = False
                       ) -> [Optional[int], Optional[int]]:
        'all move to cpu first, then consider to move to GPU again'

        self.model = self.model.to(device)

        load_dict = torch.load(
            path,
        )

        logging.info(f"loading... keys:{load_dict.keys()}, only_load_model:{only_load_model}")

        attr_list = [
            'epoch_num_when_save',
            'batch_num_when_save',
            'random_state',
            'np_random_state',
            'torch_random_state',
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'criterion_state_dict',
        ]

        if all([key_name in load_dict for key_name in attr_list]) :
            # all required key can find in load dict
            # AND only_load_model == False
            if only_load_model == False:
                random.setstate(load_dict['random_state'])
                np.random.set_state(load_dict['np_random_state'])
                torch.random.set_rng_state(load_dict['torch_random_state'].cpu()) # since may map to cuda

                self.model.load_state_dict(
                    load_dict['model_state_dict']
                )
                self.optimizer.load_state_dict(
                    load_dict['optimizer_state_dict']
                )
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(
                        load_dict['scheduler_state_dict']
                    )
                self.criterion.load_state_dict(
                    load_dict['criterion_state_dict']
                )
                logging.info('all state load successful')
                return load_dict['epoch_num_when_save'], load_dict['batch_num_when_save']
            else:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None

        else:  # only state_dict

            if 'model_state_dict' in load_dict:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None
            else:
                self.model.load_state_dict(
                    load_dict,
                )
                logging.info('only model state_dict load')
                return None, None

    def test(self, test_data, device):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            # 'detail_list' : [],
        }

        criterion = self.criterion.to(device)

        with torch.no_grad():
            for batch_idx, (x, target, *additional_info) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                # logging.info(list(zip(additional_info[0].cpu().numpy(), pred.detach().cpu().numpy(),
                #                target.detach().cpu().numpy(), )))

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                # metrics['detail_list'] += list(zip(
                #     additional_info[0].cpu().numpy(),
                #     pred.detach().cpu().numpy(),
                #     predicted.detach().cpu().numpy(),
                #     target.detach().cpu().numpy(),
                # ))
                # logging.info(f"testing, batch_idx:{batch_idx}, acc:{metrics['test_correct']}/{metrics['test_total']}")

        return metrics

    #@resource_check
    def train_one_batch(self, x, labels, device):

        self.model.train()
        self.model.to(device)

        x, labels = x.to(device), labels.to(device)
        self.model.zero_grad()
        log_probs = self.model(x)
        loss = self.criterion(log_probs, labels.long())
        loss.backward()

        # Uncommet this following line to avoid nan loss
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # logging.info(list(zip(additional_info[0].cpu().numpy(), log_probs.detach().cpu().numpy(), labels.detach().cpu().numpy(), )))

        self.optimizer.step()

        batch_loss = (loss.item())

        return batch_loss
        # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")

    def train_one_epoch(self, train_data, device):

        batch_loss = []
        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):
            batch_loss.append(self.train_one_batch(x, labels, device))
        one_epoch_loss = sum(batch_loss) / len(batch_loss)

        if self.scheduler is not None:
            self.scheduler.step()

        return one_epoch_loss

    def train(self, train_data, end_epoch_num,
                               criterion,
                               optimizer,
                               scheduler, device,  frequency_save, save_folder_path,
              save_prefix,
              continue_training_path: Optional[str] = None,
              only_load_model: bool = False, ):

        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train, epoch_loss: {epoch_loss[-1]}')
            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")

    #@resource_check
    def action_in_eval(self,
                       action_dl_list : list,
                       device,
                       control_metrics_deque: deque,
                       epoch_idx :int ,
                       batch_idx : int,
                       batch_total:int
                       ) -> (dict, bool, dict):

        save_indicator_all = False

        additional_save_file_dict = {} # action name : file

        metrics_afterward_control_dict = {}  # do NEVER put tensor on cuda in this !!!! ONLY for control further action and save func, do log in metrics_to_log !!!

        for i, (rule, action, dl) in enumerate(action_dl_list): # i is to distinguish save files amd metrics from same action on different inputs

            if rule(epoch_idx, batch_idx, batch_total, control_metrics_deque):

                metrics_to_log, metrics_afterward_control, save_indicator_once, *additional_save_file = action(self.model, dl, device, control_metrics_deque)

                metrics_to_log.update(
                    {
                        'epoch':epoch_idx,
                        'epoch_batch_idx': epoch_idx + batch_idx/batch_total,
                     }
                )
                logging.info(pformat(metrics_to_log))
                try:
                    wandb.log(
                        metrics_to_log
                    )
                except:
                    pass

                save_indicator_all = save_indicator_all or save_indicator_once # once decide to save, save_indicator change to True

                if len(additional_save_file) != 0:

                    additional_save_file_dict[f'{i}_{action.__name__}'] = additional_save_file # a list

                if metrics_afterward_control is not None:

                    metrics_afterward_control_dict[f'{i}_{action.__name__}'] = metrics_afterward_control

        return metrics_afterward_control_dict, save_indicator_all, additional_save_file_dict

    def general_train_with_eval_function_in_epoch_and_batch(self,
                                                            end_epoch_num,
                                                            criterion,
                                                            optimizer,
                                                            scheduler,
                                                            device,
                                                            train_data,
                                                            save_folder_path,
                                                            save_prefix,
                                                            rule_save_for_batch_level = (lambda epoch_idx, batch_idx, batch_total, control_metrics_deque : True) ,
                                                            action_dl_list_for_batch_level: list = (),
                                                            continue_training_path: Optional[str] = None,
                                                            only_load_model: bool = False,
                                                            sliding_window_batch_len : int = 20,
                                                            ):

        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )

        epoch_loss = []

        batch_total = len(train_data)

        control_metrics_deque = deque(maxlen= sliding_window_batch_len)

        batch_already_start_indicator = False

        if self.start_batch == 0:

            batch_already_start_indicator = True

            metrics_afterward_control_dict, save_indicator, additional_save_file_dict = self.action_in_eval(
                action_dl_list=action_dl_list_for_batch_level,
                device=device,
                control_metrics_deque = control_metrics_deque,
                epoch_idx=self.start_epochs,
                batch_idx=0,
                batch_total = batch_total,
            )

            control_metrics_deque.append(metrics_afterward_control_dict)

            if rule_save_for_batch_level(0, 0, batch_total, control_metrics_deque) or save_indicator: # satisfiy one condition then save
                # if frequency_save_for_epoch != 0 and epoch % frequency_save_for_epoch == frequency_save_for_epoch - 1:
                self.save_all_state_to_path(
                    epoch=0,
                    batch=0,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{0}.pt"
                )
                logging.info(f'saved. epoch:{0}, at {save_folder_path}/{save_prefix}_epoch_{0}.pt')

            if bool(additional_save_file_dict): #check if empty, bool({}) is False
                torch.save(additional_save_file_dict,
                           f"{save_folder_path}/{save_prefix}_epoch_{0}_additional_file.pt")
                logging.info(f'additional save at {save_folder_path}/{save_prefix}_epoch_{0}_additional_file.pt')

        for epoch in range(self.start_epochs, self.end_epochs):

            batch_loss = []

            for batch_idx, (x, labels, *additional_info) in enumerate(train_data):

                if self.start_batch <= batch_idx + 1 or batch_already_start_indicator == True:

                    batch_already_start_indicator = True

                    one_batch_loss = self.train_one_batch(x, labels, device)

                    batch_loss.append(one_batch_loss)

                    batch_train_info = {
                        'train_batch_loss': one_batch_loss,
                    }

                    batch_train_info.update(
                        {
                            'epoch': epoch,
                            'epoch_batch_idx': epoch + batch_idx / batch_total,
                        }
                    )

                    logging.info(pformat(batch_train_info))
                    try:
                        wandb.log(batch_train_info)
                    except:
                        pass

                    metrics_afterward_control_dict, save_indicator, additional_save_file_dict = self.action_in_eval(
                        action_dl_list=action_dl_list_for_batch_level,
                        device=device,
                        control_metrics_deque = control_metrics_deque,
                        epoch_idx=epoch,
                        batch_idx = batch_idx + 1,
                        batch_total=batch_total,
                    )

                    control_metrics_deque.append(metrics_afterward_control_dict)

                    if rule_save_for_batch_level(epoch, (batch_idx + 1), batch_total, control_metrics_deque) or save_indicator:
                    # if frequency_save_for_batch != 0 and batch_idx % frequency_save_for_batch == frequency_save_for_batch - 1:
                        self.save_all_state_to_path(
                            epoch=epoch,
                            batch = batch_idx + 1,
                            path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}_batch_{(batch_idx + 1)}.pt")
                        logging.info(f'saved. epoch:{epoch}, (batch_idx + 1):{(batch_idx + 1)}, (batch_idx + 1)_in_percent = {(batch_idx + 1) / len(train_data)}')
                        logging.info(f'at {save_folder_path}/{save_prefix}_epoch_{epoch}_batch_{(batch_idx + 1)}.pt')

                    if bool(additional_save_file_dict):  # check if empty, bool({}) is False
                        torch.save(additional_save_file_dict,
                                   f"{save_folder_path}/{save_prefix}_epoch_{epoch}_batch_{(batch_idx + 1)}_additional_file.pt")
                        logging.info(f'additional save at {save_folder_path}/{save_prefix}_epoch_{epoch}_batch_{(batch_idx + 1)}_additional_file.pt')

            one_epoch_loss = sum(batch_loss) / len(batch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_batch, epoch:{epoch} epoch_loss: {epoch_loss[-1]}')

    def train_with_test_each_epoch(self,
                                   train_data,
                                   test_data,
                                   adv_test_data,
                                   end_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   continue_training_path: Optional[str] = None,
                                   only_load_model: bool = False,
                                   ):

        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            metrics = self.test(test_data, device)
            metric_info = {
                'epoch': epoch,
                'benign acc': metrics['test_correct'] / metrics['test_total'],
                'benign loss': metrics['test_loss'],
            }
            logging.info(pformat(metric_info))
            try:
                wandb.log(metric_info)
            except:
                pass

            adv_metrics = self.test(adv_test_data, device)
            adv_metric_info = {
                'epoch': epoch,
                'ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
                'backdoor loss': adv_metrics['test_loss'],
            }
            logging.info(pformat(adv_metric_info))
            try:
                wandb.log(adv_metric_info)
            except:
                pass

            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")

    def train_with_test_each_batch(self,
                                   train_data,
                                   test_data,
                                   adv_test_data,
                                   end_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   continue_training_path: Optional[str] = None,
                                   only_load_model: bool = False,
                                   ):

        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):

            batch_loss = []
            for batch_idx, (x, labels, *additional_info) in enumerate(train_data):

                batch_loss.append(self.train_one_batch(x, labels, device))

                metrics = self.test(test_data, device)
                metric_info = {
                    'epoch': epoch,
                    'epoch_batch_idx': epoch + batch_idx / len(train_data),
                    'benign acc': metrics['test_correct'] / metrics['test_total'],
                    'benign loss': metrics['test_loss'],
                }
                logging.info(pformat(metric_info))
                try:
                    wandb.log(
                        metric_info
                    )
                except:
                    pass

                adv_metrics = self.test(adv_test_data, device)
                adv_metric_info = {
                    'epoch': epoch,
                    'epoch_batch_idx': epoch + batch_idx / len(train_data),
                    'ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
                    'backdoor loss': adv_metrics['test_loss'],
                }
                logging.info(pformat(adv_metric_info))
                try:
                    wandb.log(adv_metric_info)
                except:
                    pass

            one_epoch_loss = sum(batch_loss) / len(batch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_batch, epoch:{epoch} epoch_loss: {epoch_loss[-1]}')

            # still save for epoch
            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")

    def train_in_one_epoch_and_save_batch(self,
                           train_data,
                           test_data,
                           adv_test_data,
                           end_epoch_num,
                           criterion,
                           optimizer,
                           scheduler,
                           device,
                           frequency_batch_save,
                           save_folder_path,
                           save_prefix,
                           continue_training_path: str,
                           ):
        '''
        load specific epoch and train for some batch to extract specific batch model result
        '''
        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model = False,
        )

        batch_loss = []
        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):

            batch_loss.append(self.train_one_batch(x, labels, device))

            metrics = self.test(test_data, device)
            metric_info = {
                'epoch': self.start_epochs,
                'epoch_batch_idx': self.start_epochs + batch_idx / len(train_data),
                'benign acc': metrics['test_correct'] / metrics['test_total'],
                'benign loss': metrics['test_loss'],
            }
            logging.info(pformat(metric_info))
            try:
                wandb.log(
                    metric_info
                )
            except:
                pass

            adv_metrics = self.test(adv_test_data, device)
            adv_metric_info = {
                'epoch': self.start_epochs,
                'epoch_batch_idx': self.start_epochs + batch_idx / len(train_data),
                'ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
                'backdoor loss': adv_metrics['test_loss'],
            }

            logging.info(pformat(adv_metric_info))
            try:
                wandb.log(adv_metric_info)
            except:
                pass

            if frequency_batch_save != 0 and batch_idx % frequency_batch_save == frequency_batch_save - 1:
                logging.info(f'saved. batch:{batch_idx}')
                self.save_all_state_to_path(
                    path=f"{save_folder_path}/{save_prefix}_batch_{batch_idx}.pt",
                    only_model_state_dict=True,
                )

# if __name__ == '__main__':
#
#     class Args():
#         def __init__(self, settings_dict):
#             self.__dict__ = settings_dict
#
#
#     train_args = Args({
#
#         'client_optimizer': 'sgd',  # 'sgd',
#         'epochs': 100,
#         'batch_size': 128,
#         'lr': 0.01,
#         'wd': 5e-4,  # 5e-4,
#         'lr_scheduler': 'CosineAnnealingLR',  # 'CosineAnnealingLR', #'StepLR',
#
#         # 'flooding_scalar':0.5,
#
#         # 'steplr_stepsize':20,
#         # 'steplr_gamma':0.5,
#         "sgd_momentum": 0.9,
#         # "adam_betas": (0.999, 0.999),
#     })
#
#     from test_examples.pytorch_train_a_classifier_eg import *
#     trainer = MyModelTrainerCLS(net)
#
#     from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
#
#     optimizer, scheduler = argparser_opt_scheduler(net,train_args)
#     criterion = argparser_criterion(train_args)
#     #
#     # trainer.init_or_continue_train(
#     #             train_data,
#     #             end_epoch_num,
#     #             criterion,
#     #             optimizer,
#     #             scheduler,
#     #             device,
#     #             continue_training_path,
#     #             only_load_model
#     #         )
#     #
#     # trainer.save_all_state_to_path(
#     #     path = '../record/1.pth',
#     #     epoch = 2,
#     #     batch = 3,
#     # )
#
#     # batch_loss = []
#     # for batch_idx, (x, labels, *additional_info) in enumerate(trainloader):
#     #     batch_loss.append(trainer.train_one_batch(x, labels, device))
#     #     print(batch_loss[-1])
#     # one_epoch_loss = sum(batch_loss) / len(batch_loss)
#     #
#     # if trainer.scheduler is not None:
#     #     trainer.scheduler.step()
#
#     def rule_save_for_batch_train(epoch, batch_idx, batch_total, control_metrics_deque):
#         if batch_idx == batch_total:
#             return True
#         else:
#             return False
#
#
#     def action_rule_for_batch_train(epoch, batch_idx, batch_total, control_metrics_deque):
#         if batch_idx == batch_total:
#             return True
#         else:
#             return False
#
#
#     def testAccAsrAndCalculate(net, dls, device, control_metrics_deque):
#         return {}, None, False,
#
#
#     action_dl_list_per_batch = [
#         (
#             action_rule_for_batch_train,  # same rule, so no further define
#             testAccAsrAndCalculate,
#             {
#
#             },
#         )
#     ]
#
#
#
#     trainer.general_train_with_eval_function_in_epoch_and_batch(
#         end_epoch_num = 10,
#         criterion = criterion,
#         optimizer = optimizer,
#         scheduler = scheduler,
#         device = device,
#         train_data = trainloader,
#         save_folder_path = '../record',
#         save_prefix = '1',
#         rule_save_for_batch_level=rule_save_for_batch_train,
#         action_dl_list_for_batch_level= action_dl_list_per_batch,
#         continue_training_path = '../record/1.pth',
#         only_load_model = False,
#         sliding_window_batch_len = 20,
#     )

