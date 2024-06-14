'''
Enhancing Fine-Tuning Based Backdoor Defense with Sharpness-Aware Minimization

@InProceedings{Zhu_2023_ICCV,
            author    = {Zhu, Mingli and Wei, Shaokui and Shen, Li and Fan, Yanbo and Wu, Baoyuan},
            title     = {Enhancing Fine-Tuning Based Backdoor Defense with Sharpness-Aware Minimization},
            booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
            month     = {October},
            year      = {2023},
            pages     = {4466-4477}}

basic structure:
1. config args, save_path, fix random seed
2. load the backdoor attack data and backdoor test data
3. load the backdoor model
4. for each round sample a clean batch from given clean subset:
    a. do weight perturb to maximize L constrained by rho
    b. do outer minimization
5. test the result and get ASR, ACC, RC

'''
import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.getcwd())

# TODO:修改yaml文件

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense
from utils.defense_utils.sam import SAM, ProportionScheduler
from utils.defense_utils.sam import smooth_crossentropy

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator
from utils.choose_index import choose_index,choose_by_class
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)): # output: (256,10); target: (256)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk) # 5
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # pred: (256,5)
        pred = pred.t() # (5,256)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # (5,256)

        res = []

        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = torch.flatten(correct[:k]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def given_dataloader_test(
        model,
        test_dataloader,
        criterion,
        non_blocking : bool = False,
        device = "cpu",
        verbose : int = 0
):
    model.to(device, non_blocking=non_blocking)
    model.eval()
    metrics = {
        'test_correct': 0,
        'test_loss_sum_over_batch': 0,
        'test_total': 0,
    }
    criterion = criterion.to(device, non_blocking=non_blocking)

    if verbose == 1:
        batch_predict_list, batch_label_list = [], []

    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_dataloader):
            x = x.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            pred = model(x)
            loss = criterion(pred, target.long())

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            if verbose == 1:
                batch_predict_list.append(predicted.detach().clone().cpu())
                batch_label_list.append(target.detach().clone().cpu())

            metrics['test_correct'] += correct.item()
            metrics['test_loss_sum_over_batch'] += loss.item()
            metrics['test_total'] += target.size(0)

    metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch']/len(test_dataloader)
    metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

    if verbose == 0:
        return metrics, None, None
    elif verbose == 1:
        return metrics, torch.cat(batch_predict_list), torch.cat(batch_label_list)

class dsam(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')
        parser.add_argument('--print_freq', default=1, type=int,help=' print_freq')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/ft-sam/config.yaml", help='the path of yaml')
        parser.add_argument('--bd_yaml_path', type=str, default=None, help='the path of yaml')

        #set the parameter for the dsam defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')

        parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
        parser.add_argument("--adaptive", action='store_false', help="True if you want to use the Adaptive SAM.")
        parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
        parser.add_argument("--rho_max", default=2.0, type=float, help="Rho parameter for SAM.")
        parser.add_argument("--rho_min", default=2.0, type=float, help="Rho parameter for SAM.")
        parser.add_argument("--alpha", default=0.0, type=float, help="Rho parameter for SAM.")
        parser.add_argument("--checkpoint_path", default=None, type=str, help="specify the checkpoint")

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        # save_path = 'record/' + result_file + f'/defense/epochs_{args.epochs}_dsam_{args.ratio}_lr_{args.lr}_rho_{args.rho}/'
        save_path = 'record/' + result_file + f'/defense/ft-sam/'
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))
    
    def set_devices(self):
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

    def eval_step(self, model, clean_test_loader, bd_test_loader, args):
        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            model,
            clean_test_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            model,
            bd_test_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        bd_test_loader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            model,
            bd_test_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
        test_ra = ra_metrics['test_acc']
        bd_test_loader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

        return clean_test_loss_avg_over_batch, \
				bd_test_loss_avg_over_batch, \
				ra_test_loss_avg_over_batch, \
				test_acc, \
				test_asr, \
				test_ra

    def _train_sam(self, args, train_loader, model, optimizer, scheduler,criterion, epoch):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        for idx, (img, target, *flag) in enumerate(train_loader, start=1):
            img = img.to(args.device)
            target = target.to(args.device)
            bsz = target.shape[0]
            def loss_fn(predictions, targets):
                return smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()
            optimizer.set_closure(loss_fn, img, target)
            predictions, loss = optimizer.step()
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == target
                correct = correct.sum()
                scheduler.step()
                optimizer.update_rho_t()

            # update metric

            losses.update(loss.item(), bsz)
            top1.update(correct.detach().cpu().numpy()/bsz, bsz)
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # top1.update(acc1[0].detach().cpu().numpy(), bsz)
            if (idx + 1) % args.print_freq == 0:
                logging.info(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t \
                    loss {losses.val} ({losses.avg}\t \
                    Acc@1 {top1.val} ({top1.avg}')
                sys.stdout.flush()
   
        del loss, img
        torch.cuda.empty_cache()
        return losses.avg, top1.avg, model

    def train_sam(self, model,train_dataloader,
                                   clean_test_dataloader,
                                   bd_test_dataloader,
                                   total_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   amp,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   prefetch,
                                   prefetch_transform_attr_name,
                                   non_blocking,
                                   ):
        
      
        criterion = criterion.to(args.device)

        # Training and Testing
        train_loss_list = []
        train_mix_acc_list = []
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        agg = Metric_Aggregator()


        for epoch in tqdm(range(1, args.epochs+1)):
            train_epoch_loss_avg_over_batch, \
            train_mix_acc, \
            model = self._train_sam(args, train_dataloader, model, optimizer, scheduler,criterion, epoch)

            clean_test_loss_avg_over_batch, \
			bd_test_loss_avg_over_batch, \
			ra_test_loss_avg_over_batch, \
			test_acc, \
			test_asr, \
			test_ra = self.eval_step(
				model,
				clean_test_dataloader,
				bd_test_dataloader,
				args,
			)
            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            
            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)
            agg(
                    {
                        "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                        "train_acc": train_mix_acc,
                        "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                        "bd_test_loss_avg_over_batch" : bd_test_loss_avg_over_batch,
                        "test_acc" : test_acc,
                        "test_asr" : test_asr,
                        "test_ra" : test_ra,
                    }
            )
            agg.to_dataframe().to_csv(f"{args.log}d-sam_df.csv")

        agg.summary().to_csv(f"{args.log}d-sam_df_summary.csv")

        return model


    def mitigation(self):
        args=self.args
        self.set_devices()
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model,self.args.num_classes)
    
       
        if hasattr(args,"checkpoint_path") and args.checkpoint_path != None:
            file_path = 'record/' + args.checkpoint_path 
            checkpoint_path = load_attack_result(file_path + '/defense_result.pt')
            model.load_state_dict(checkpoint_path['model'])
        else:
            model.load_state_dict(self.result['model'])

        if "," in self.args.device:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
        else:
            model.to(self.args.device)
        base_optimizer, scheduler = argparser_opt_scheduler(model, self.args)
   
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.lr, min_lr=0.0,
            max_value=args.rho_max, min_value=args.rho_min)
        optimizer = SAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, sam_alpha=args.alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)
        

        # criterion = nn.CrossEntropyLoss()
        criterion = argparser_criterion(args)

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        # data_all_length = len(clean_dataset)
        # ran_idx = choose_index(self.args, data_all_length) 
        ran_idx = choose_by_class(args,clean_dataset)
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        trainloader = data_loader
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        self.train_sam(
            model,
            trainloader,
            data_clean_loader,
            data_bd_loader,
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='dsam',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )
        
        result = {}
        result['model'] = model

        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    dsam.add_arguments(parser)
    args = parser.parse_args()
    dsam_method = dsam(args)
    result = dsam_method.defense(args.result_file)