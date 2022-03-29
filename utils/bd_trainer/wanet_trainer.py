
'''

Notice that in train mode I adopt almost the same way in original code (but for all2one part I do not choose those img from target class)
but for test part for all2one case I do not choose those img from target class

rewrite from
    @inproceedings{
    nguyen2021wanet,
    title={WaNet - Imperceptible Warping-based Backdoor Attack},
    author={Tuan Anh Nguyen and Anh Tuan Tran},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=eEn8KTtJOx}
    }
    code : https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
'''


import sys, logging
sys.path.append('../')
import random
from pprint import pformat
from collections import deque
from typing import *
import numpy as np
import torch


from utils.trainer_cls import MyModelTrainerCLS


class wanetTrainerCLS(MyModelTrainerCLS):
    '''

    '''
    def __init__(self, model):
        super(wanetTrainerCLS, self).__init__(model)

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

    def train_one_epoch_wanet(self, train_data, bd_batch_operation, device):

        batch_loss = []
        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):
            x, labels = bd_batch_operation(x,labels)
            batch_loss.append(self.train_one_batch(x, labels, device))
        one_epoch_loss = sum(batch_loss) / len(batch_loss)

        if self.scheduler is not None:
            self.scheduler.step()

        return one_epoch_loss

    def noise_training_in_wanet(self,
                                   train_data,
                                   bd_batch_operation,
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
            one_epoch_loss = self.train_one_epoch_wanet(train_data, bd_batch_operation, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            metrics = self.test(test_data, device)
            metric_info = {
                'epoch': epoch,
                'benign acc': metrics['test_correct'] / metrics['test_total'],
                'benign loss': metrics['test_loss'],
            }
            logging.info(pformat(metric_info))


            adv_metrics = self.test(adv_test_data, device)
            adv_metric_info = {
                'epoch': epoch,
                'ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
                'backdoor loss': adv_metrics['test_loss'],
            }
            logging.info(pformat(adv_metric_info))


            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")

