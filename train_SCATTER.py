import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
# from apex import amp
from tensorboardX import SummaryWriter

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model_s as Model
from test_SCATTER import validation

try:
    import apex
    amp = apex.amp
except ImportError:
    apex = None
    amp = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ model configuration """
    # CTCLoss
    converter_ctc = CTCLabelConverter(opt.character)
    # Attention
    converter_atten = AttnLabelConverter(opt.character)
    opt.num_class_ctc = len(converter_ctc.character)
    opt.num_class_atten = len(converter_atten.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class_ctc, opt.num_class_atten, opt.batch_max_length, opt.Transformation,
          opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p_: p_.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    # use fp16 to train
    model = model.to(device)
    if opt.fp16:
        with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
            log.write('==> Enable fp16 training' + '\n')
        print('==> Enable fp16 training')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # data parallel for multi-GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)
    model.train()
    # for i in model.module.Prediction_atten:
    #     i.to(device)
    # for i in model.module.Feat_Extraction.scr:
    #     i.to(device)
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    criterion_ctc = torch.nn.CTCLoss(zero_infinity=True).to(device)
    criterion_atten = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    # loss averager
    loss_avg = Averager()

    """ final options """
    writer = SummaryWriter(f'./saved_models/{opt.exp_name}')
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    # image_tensors, labels = train_dataset.get_batch()
    while True:
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        batch_size = image.size(0)
        text_ctc, length_ctc = converter_ctc.encode(labels, batch_max_length=opt.batch_max_length)
        text_atten, length_atten = converter_atten.encode(labels, batch_max_length=opt.batch_max_length)

        # type tuple; (tensor, list);         text_atten[:, :-1]:align with Attention.forward
        preds_ctc, preds_atten = model(image, text_atten[:, :-1])
        # CTC Loss
        preds_size = torch.IntTensor([preds_ctc.size(1)] * batch_size)
        # _, preds_index = preds_ctc.max(2)
        # preds_str_ctc = converter_ctc.decode(preds_index.data, preds_size.data)
        preds_ctc = preds_ctc.log_softmax(2).permute(1, 0, 2)
        cost_ctc = 0.1 * criterion_ctc(preds_ctc, text_ctc, preds_size, length_ctc)

        # Attention Loss
        # preds_atten = [i[:, :text_atten.shape[1] - 1, :] for i in preds_atten]
        # # select max probabilty (greedy decoding) then decode index to character
        # preds_index_atten = [i.max(2)[1] for i in preds_atten]
        # length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        # preds_str_atten = [converter_atten.decode(i, length_for_pred) for i in preds_index_atten]
        # preds_str_atten2 = preds_str_atten
        # preds_str_atten = []
        # for i in preds_str_atten2:  # prune after "end of sentence" token ([s])
        #     temp = []
        #     for j in i:
        #         j = j[:j.find('[s]')]
        #         temp.append(j)
        #     preds_str_atten.append(temp)
        # preds_str_atten = [j[:j.find('[s]')] for i in preds_str_atten for j in i]
        target = text_atten[:, 1:]  # without [GO] Symbol
        # cost_atten = 1.0 * criterion_atten(preds_atten.view(-1, preds_atten.shape[-1]), target.contiguous().view(-1))
        for index, pred in enumerate(preds_atten):
            if index == 0:
                cost_atten = 1.0 * criterion_atten(pred.view(-1, pred.shape[-1]), target.contiguous().view(-1))
            else:
                cost_atten += 1.0 * criterion_atten(pred.view(-1, pred.shape[-1]), target.contiguous().view(-1))
        # cost_atten = [1.0 * criterion_atten(pred.view(-1, pred.shape[-1]), target.contiguous().view(-1)) for pred in
        #               preds_atten]
        # cost_atten = criterion_atten(preds_atten.view(-1, preds_atten.shape[-1]), target.contiguous().view(-1))
        cost = cost_ctc + cost_atten
        writer.add_scalar('loss', cost.item(), global_step=iteration + 1)

        # cost = cost_ctc
        # cost = cost_atten
        if (iteration + 1) % 100 == 0:
            print('\riter: {:4d}\tloss: {:6.3f}\tavg: {:6.3f}'.format(iteration + 1, cost.item(), loss_avg.val()), end='\n')
        else:
            print('\riter: {:4d}\tloss: {:6.3f}\tavg: {:6.3f}'.format(iteration + 1, cost.item(), loss_avg.val()), end='')
        sys.stdout.flush()
        if cost < 0.001:
            print(f'iter: {iteration + 1}\tloss: {cost}')
            # aaaaaa = 0

        # model.zero_grad()
        optimizer.zero_grad()
        if torch.isnan(cost):
            print(f'iter: {iteration + 1}\tloss: {cost}\t==> Loss is NAN')
            sys.exit()
        elif torch.isinf(cost):
            print(f'iter: {iteration + 1}\tloss: {cost}\t==> Loss is INF')
            sys.exit()
        else:
            if opt.fp16:
                with amp.scale_loss(cost, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)
        writer.add_scalar('loss_avg', loss_avg.val(), global_step=iteration + 1)
        # if loss_avg.val() <= 0.6:
        #     opt.grad_clip = 2
        # if loss_avg.val() <= 0.3:
        #     opt.grad_clip = 1

        # validation part
        if iteration == 0 or (iteration + 1) % opt.valInterval == 0:  # To see training progress, we also conduct validation when 'iteration == 0'
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion_atten, valid_loader, converter_atten, opt)
                model.train()
                writer.add_scalar('accuracy', current_accuracy, global_step=iteration + 1)

                # training loss and validation loss
                loss_log = f'[{iteration + 1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration + 1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()

        # if (iteration + 1) % opt.valInterval == 0:
        #     print(f'iter: {iteration + 1}\tloss: {cost}')
        iteration += 1


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--fp16', action='store_true', help='whether to train with fp16')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz',
                        # default='*0123456789abcdefghijklmnopqrstuvwxyz+-/.,?<>;\':"\[\]\\{}|=_()&%$#@!~',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # SCATTER
    parser.add_argument('--LSTM_Layer', type=int, default=2, help='the layers of the LSTM')
    parser.add_argument('--Selective_Layer', type=int, default=1, help='the layers of the Selective decoder')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
