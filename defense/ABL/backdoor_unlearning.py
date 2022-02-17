import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from models.selector import *
from utils.util import *
from data_loader import *
# from utils.dataloader_bd import *
from config import get_arguments
from utils.network import get_network
import pdb

def train_step_finetuing(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        output = model_ascent(img)

        loss = criterion(output, target)

        prec1 = accuracy(output, target)[0]
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


def train_step_unlearning(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        output = model_ascent(img)

        loss = criterion(output, target)

        prec1 = accuracy(output, target)[0]
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        (-loss).backward()  # Gradient ascent training
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


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

    for idx, (img, target, isClean, gt_label) in enumerate(test_bad_loader, start=1):
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

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target, isClean, gt_label) in enumerate(test_bad_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            gt_label = gt_label.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, gt_label)

        prec1 = accuracy(output, gt_label)[0]
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

    acc_r = [top1.avg, top5.avg, losses.avg]

    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))
    print('[Robust] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_r[0], acc_r[2]))

    # save training progress
    log_root = opt.log_root + '/ABL_unlearning.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
                                             "Test_clean_loss", "Test_bad_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd


def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    model_ascent, checkpoint_epoch = select_model(dataset=opt.dataset,
                                model_name=opt.model_name,
                                pretrained=True,
                                pretrained_models_path=opt.checkpoint_root,
                                n_classes=opt.num_classes)
    # model_ascent = get_network(opt)
    # checkpoint = torch.load(opt.checkpoint_load)
    # model_ascent.load_state_dict(checkpoint['state_dict'])
    model_ascent.to(opt.device)
    print('Finish loading ascent model...')

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
    # data_path_isolation = os.path.join(opt.isolate_data_root, "{}_isolation{}_examples.npy".format(opt.model_name,
    #                                                                                                 opt.isolation_ratio * 100))
    # data_path_other = os.path.join(opt.isolate_data_root, "{}_other{}_examples.npy".format(opt.model_name,
    #                                                                                         100 - opt.isolation_ratio * 100))
    data_path_isolation = opt.data_path_isolation
    data_path_other = opt.data_path_other

    tf_compose_finetuning = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.input_height, opt.input_width)),
        transforms.RandomCrop((opt.input_height, opt.input_width), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])
    # transforms_list = []
    # transforms_list.append(transforms.ToPILImage())
    # transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    # transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
    # # transforms_list.append(transforms.RandomRotation(10))
    # if opt.dataset == "cifar10":
    #     transforms_list.append(transforms.RandomHorizontalFlip())
    # transforms_list.append(transforms.ToTensor())
    # if opt.dataset == "cifar10":
    #     transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
    # elif opt.dataset == "mnist":
    #     transforms_list.append(transforms.Normalize([0.5], [0.5]))
    # elif opt.dataset == 'tiny':
    #     transforms_list.append(transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    # elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
    #     pass
    # else:
    #     raise Exception("Invalid Dataset")
    # tf_compose_finetuning = transforms.Compose(transforms_list)

    tf_compose_unlearning = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.input_height, opt.input_width)),
        transforms.ToTensor()
    ])
    # transforms_list = []
    # transforms_list.append(transforms.ToPILImage())
    # transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    # transforms_list.append(transforms.ToTensor())
    # if opt.dataset == "cifar10":
    #     transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
    # elif opt.dataset == "mnist":
    #     transforms_list.append(transforms.Normalize([0.5], [0.5]))
    # elif opt.dataset == 'tiny':
    #     transforms_list.append(transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    # elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
    #     pass
    # else:
    #     raise Exception("Invalid Dataset")
    # tf_compose_unlearning = transforms.Compose(transforms_list)
    # tf_compose_unlearning = get_transform(opt, train=False)

    isolate_poisoned_data = np.load(data_path_isolation, allow_pickle=True)
    poisoned_data_tf = Dataset_npy(full_dataset=isolate_poisoned_data, transform=tf_compose_unlearning)
    isolate_poisoned_data_loader = DataLoader(dataset=poisoned_data_tf,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      )

    isolate_other_data = np.load(data_path_other, allow_pickle=True)
    isolate_other_data_tf = Dataset_npy(full_dataset=isolate_other_data, transform=tf_compose_finetuning)
    isolate_other_data_loader = DataLoader(dataset=isolate_other_data_tf,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              )

    test_clean_loader, test_bad_loader = get_test_loader(opt)

    if opt.finetuning_ascent_model == True:
        # this is to improve the clean accuracy of isolation model, you can skip this step
        print('----------- Finetuning isolation model --------------')
        for epoch in range(0, opt.finetuning_epochs):
            learning_rate_finetuning(optimizer, epoch, opt)
            train_step_finetuing(opt, isolate_other_data_loader, model_ascent, optimizer, criterion,
                             epoch + 1)
            test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

    print('----------- Model unlearning --------------')
    for epoch in range(0, opt.unlearning_epochs):
        learning_rate_unlearning(optimizer, epoch, opt)

        # train stage
        if epoch == 0:
            # test firstly
            test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch)
        else:
            train_step_unlearning(opt, isolate_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        # evaluate on testing set
        print('testing the ascended model......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

        if opt.save:
            # save checkpoint at interval epoch
            # if epoch + 1 % opt.interval == 0:
            is_best = True
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_ascent.state_dict(),
                'clean_acc': acc_clean[0],
                'bad_acc': acc_bad[0],
                'optimizer': optimizer.state_dict(),
            }, epoch + 1, is_best, opt)


def learning_rate_finetuning(optimizer, epoch, opt):
    if epoch < 40:
        lr = 0.01
    elif epoch < 60:
        lr = 0.001
    else:
        lr = 0.001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def learning_rate_unlearning(optimizer, epoch, opt):
    if epoch < opt.unlearning_epochs:
        lr = 0.0001
    else:
        lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, epoch, is_best, opt):
    if is_best:
        # filepath = os.path.join(opt.save_root, opt.model_name + r'-unlearning_epochs{}.tar'.format(epoch))
        filepath = opt.checkpoint_save
        torch.save(state, filepath)
    print('[info] Finish saving the model')

def main():
    # Prepare arguments
    opt = get_arguments()
    train(opt)

if (__name__ == '__main__'):
    main()