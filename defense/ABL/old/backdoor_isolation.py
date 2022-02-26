import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models.selector import *
from utils.util import *
from data_loader import *
# from utils.dataloader_bd import *
from torch.utils.data import DataLoader
from config import get_arguments
from tqdm import tqdm
import pdb
from utils.network import get_network

def compute_loss_value(opt, poisoned_data, model_ascent):
    # Calculate loss value per example
    # Define loss function
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []

    example_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )

    for idx, (img, target, isClean, gt_label) in tqdm(enumerate(example_data_loader, start=0)):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
            # print(loss.item())

        losses_record.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in descending order

    # Show the top 10 loss values
    losses_record_arr = np.array(losses_record)
    print('Top ten loss value:', losses_record_arr[losses_idx[:10]])

    return losses_idx


def isolate_data(opt, poisoned_data, losses_idx):
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0
    ratio = opt.isolation_ratio

    example_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )
    # print('full_poisoned_data_idx:', len(losses_idx))
    perm = losses_idx[0: int(len(losses_idx) * ratio)]

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        img = img.squeeze()
        target = target.squeeze()
        img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = target.cpu().numpy()

        # Filter the examples corresponding to losses_idx
        if idx in perm:
            isolation_examples.append((img, target))
            cnt += 1
        else:
            other_examples.append((img, target))

    # Save data
    if opt.save:
        data_path_isolation = os.path.join(opt.isolate_data_root, "{}_isolation{}_examples.npy".format(opt.model_name,
                                                                                             opt.isolation_ratio * 100))
        data_path_other = os.path.join(opt.isolate_data_root, "{}_other{}_examples.npy".format(opt.model_name,
                                                                                             100 - opt.isolation_ratio * 100))
        # if os.path.exists(data_path_isolation):
        #     raise ValueError('isolation data already exists')
        # else:
        #     # save the isolation examples
        np.save(data_path_isolation, isolation_examples)
        np.save(data_path_other, other_examples)

    print('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    print('Finish collecting {} other examples: '.format(len(other_examples)))


def train_step(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.train()

    for idx, (img, target,isClean,gt_label) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        if opt.gradient_ascent_type == 'LGA':
            output = model_ascent(img)
            loss = criterion(output, target)
            # add Local Gradient Ascent(LGA) loss
            loss_ascent = torch.sign(loss - opt.gamma) * loss

        elif opt.gradient_ascent_type == 'Flooding':
            output = model_ascent(img)
            # output = student(img)
            loss = criterion(output, target)
            # add flooding loss
            loss_ascent = (loss - opt.flooding).abs() + opt.flooding

        else:
            raise NotImplementedError

        prec1 = accuracy(output, target)[0]
        losses.update(loss_ascent.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss_ascent.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'Loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'Prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'Prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.eval()

    for idx, (img, target, isClean, gt_label) in enumerate(test_clean_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1 = accuracy(output, target)[0]
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target,isClean,gt_label) in enumerate(test_bad_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1 = accuracy(output, target)[0]
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

    # save training progress
    if epoch < opt.tuning_epochs + 1:
        log_root = opt.log_root + '/ABL_results_tuning_epochs.csv'
        test_process.append(
            (epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
        df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
                                                 "Test_clean_loss", "Test_bad_loss"))
        df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd


def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    model_ascent, _ = select_model(dataset=opt.dataset,
                           model_name=opt.model_name,
                           pretrained=False,
                           pretrained_models_path=opt.isolation_model_root,
                           n_classes=opt.num_classes)
    # model_ascent = get_network(opt)
    model_ascent.to(opt.device)
    print('finished model init...')

    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    print('----------- Data Initialization --------------')
    if opt.load_fixed_data:
        tf_compose = transforms.Compose([
            transforms.ToTensor()
        ])
        # load the fixed poisoned data, e.g. Dynamic, FC, DFST attacks etc.
        poisoned_data = np.load(opt.poisoned_data_path, allow_pickle=True)
        poisoned_data_loader = DataLoader(dataset=poisoned_data,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            )
    else:
        poisoned_data, poisoned_data_loader = get_backdoor_loader(opt)

    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.tuning_epochs):

        adjust_learning_rate(optimizer, epoch, opt)

        # train every epoch
        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

        train_step(opt, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        # evaluate on testing set
        print('testing the ascended model......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

        if opt.save:
            # remember best precision and save checkpoint
            # is_best = acc_clean[0] > opt.threshold_clean
            # opt.threshold_clean = min(acc_clean[0], opt.threshold_clean)
            #
            # best_clean_acc = acc_clean[0]
            # best_bad_acc = acc_bad[0]
            #
            # save_checkpoint({
            #     'epoch': epoch,
            #     'state_dict': model_ascent.state_dict(),
            #     'clean_acc': best_clean_acc,
            #     'bad_acc': best_bad_acc,
            #     'optimizer': optimizer.state_dict(),
            # }, epoch, is_best, opt.checkpoint_root, opt.model_name)

            # save checkpoint at interval epoch
            if epoch % opt.interval == 0:
                is_best = True
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_ascent.state_dict(),
                    'clean_acc': acc_clean[0],
                    'bad_acc': acc_bad[0],
                    'optimizer': optimizer.state_dict(),
                }, epoch, is_best, opt)

    return poisoned_data, model_ascent


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch < opt.tuning_epochs:
        lr = opt.lr
    else:
        lr = 0.01
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, epoch, is_best, opt):
    if is_best:
        filepath = os.path.join(opt.save_path, opt.model_name + r'-tuning_epochs{}.tar'.format(epoch))
        torch.save(state, filepath)
    print('[info] Finish saving the model')

def main():
    print('----------- Train isolated model -----------')
    opt = get_arguments()#.parse_args()
    poisoned_data, ascent_model = train(opt)

    print('----------- Calculate loss value per example -----------')
    losses_idx = compute_loss_value(opt, poisoned_data, ascent_model)

    print('----------- Collect isolation data -----------')
    isolate_data(opt, poisoned_data, losses_idx)

if (__name__ == '__main__'):
    main()