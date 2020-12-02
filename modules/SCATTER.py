from modules.sequence_modeling import BidirectionalLSTM
from modules.ResNet29 import get_model

import torch
import torch.nn as nn
import torchvision


class SCATTER(nn.Module):
    def __init__(self, opt, input_channel=3, lstm_layer=2, selective_layer=1):
        super().__init__()
        # resnet34 = torchvision.models.resnet34(pretrained=True)
        # conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=(3, 3))
        # self.resnet34_new = nn.Sequential(
        #     conv1,
        #     resnet34.bn1,
        #     resnet34.relu,
        #     resnet34.layer1,
        #     resnet34.layer2,
        #     resnet34.layer3,
        #     resnet34.layer4,
        #     # nn.ReLU(False),
        # )
        self.resnet29 = get_model()

        self.text_attention = Text_Attention_Module()

        # for CTC loss
        self.avg = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Linear(512, opt.num_class_ctc)

        self.scr = nn.ModuleList([Selective_Contextual_Refinement(opt=opt, lstm_layer=lstm_layer)
                                  for i in range(selective_layer)])
        # self.scr = nn.Sequential()
        # for i in range(selective_layer):
        #     self.scr.add_module(str(i), Selective_Contextual_Refinement(lstm_layer=lstm_layer))
        # self.scr = Selective_Contextual_Refinement(opt=opt, lstm_layer=lstm_layer)

    def forward(self, x):  # bs = 1
        # feat_raw = self.resnet34_new(x)  # torch.Size([1, 512, 4, 13])
        feat_raw = self.resnet29(x)  # torch.Size([1, 512, 1, 26])
        attention_feat = self.text_attention(feat_raw)  # torch.Size([1, 512, 1, 26])
        # attention_feat = feat_raw  # torch.Size([1, 512, 1, 26])
        # attention_feat = feat_raw  # torch.Size([1, 512, 4, 13])

        # Convert to Feature Sequence
        # bs, c, h, w  ->  bs, (c*h), w
        # bs, c, h, w = attention_feat.shape
        # convert_feat = attention_feat.view(bs, c*h, w)
        # convert_feat = convert_feat.permute(0, 2, 1)  # torch.Size([1, 13, 2048])

        # return to CTCLoss
        # bs, c, h, w  ->  bs, w, c, h
        convert_feat = attention_feat.permute(0, 3, 1, 2)  # torch.Size([1, 26, 512, 1])
        convert_feat = self.avg(convert_feat).squeeze(3)  # bs, w, c | torch.Size([1, 26, 512])
        if self.training:
            ctc_feat = self.fc(convert_feat.contiguous())  # torch.Size([1, 26, num_class])
        # ctc_feat = None

        # return to Attention Loss
        # feat_h, feat_attention = self.scr(convert_feat, convert_feat)
        feat_attention = []  # selective_layer * torch.Size([1, 26, 1024])
        for index, s in enumerate(self.scr):
            if index == 0:
                # return torch.Size([1, 26, 512]) torch.Size([1, 26, 1024])
                feat_h, feat_d = s(convert_feat, convert_feat)
            else:
                feat_h, feat_d = s(convert_feat, feat_h)
            feat_attention.append(feat_d)
        
        if self.training:
            # return ctc_feat, feat_h
            return ctc_feat, feat_attention
        else:
            # only need last decoder , if inference
            return feat_attention
            # return out_final


# Text Attention Module
class Text_Attention_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_attention = nn.Sequential(
            # nn.Conv1d(512, 1, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.Conv2d(512, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Sigmoid(),
            # nn.ReLU(True),
        )
        
    def forward(self, x):
        attention_mask = self.text_attention(x)
        attention_feat = x * attention_mask
#         bs, c, h, w = attention_feat.shape
#         convert_feat = attention_feat.view(bs, c*h, w)
        return attention_feat


class Selective_Contextual_Refinement(nn.Module):
    def __init__(self, opt, lstm_layer=2):
        super().__init__()
        self.block = nn.Sequential(
            BidirectionalLSTM(512, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, 512),
        )
        # self.block = nn.Sequential()
        # for i in range(lstm_layer):
        #     if i == 0:
        #         self.block.add_module(str(i), BidirectionalLSTM(512, opt.hidden_size, opt.hidden_size))
        #     elif i == lstm_layer - 1:
        #         self.block.add_module(str(i), BidirectionalLSTM(opt.hidden_size, opt.hidden_size, 512))
        #     else:
        #         self.block.add_module(str(i), BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))

        # LSTM_layer = []
        # for i in range(lstm_layer):
        #     if i == 0:
        #         LSTM_layer.append(BidirectionalLSTM(2048, opt.hidden_size, opt.hidden_size))
        #     elif i == lstm_layer - 1:
        #         LSTM_layer.append(BidirectionalLSTM(opt.hidden_size, opt.hidden_size, 2048))
        #     else:
        #         LSTM_layer.append(BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        # self.block = nn.ModuleList(LSTM_layer)

        self.sd = Selective_Decoder()
        
    def forward(self, feat_v, feat_h):  # [1, 26, 512]), [1, 26, 512]
        # for index, b in enumerate(self.block):
        #     # b.rnn.flatten_parameters()
        #     feat_h = b(feat_h)
        feat_h = self.block(feat_h)  # [1, 26, 512]
        feat_d = torch.cat([feat_v, feat_h], dim=2)
        # feat_d = self.sd(feat_d)
        return feat_h, feat_d  # [1, 26, 512], [1, 26, 512*2]


class Selective_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)
        # self.fc = nn.Linear(1024, 1)

    def forward(self, x):  # [1, 26, 1024]
        map_a = self.fc(x)  # [1, 26, 1024]
        feat_a = x.mul(map_a)  # [1, 26, 1024]
        return feat_a


if __name__ == '__main__':
    from train_SCATTER import get_parser
    parser = get_parser()
    opt = parser.parse_args()

    scatter = SCATTER(opt=opt, input_channel=3, lstm_layer=2, selective_layer=1)
    scatter.train()
    
    img = torch.randn(1, 3, 32, 100)
    out_mymodel = scatter(img)
    
    print(out_mymodel[0].shape)  # [1, 26, num_class]
    print(type(out_mymodel[1]))  # <class 'list'>
    print(len(out_mymodel[1]))  # 1
    print(out_mymodel[1][0].shape)  # [1, 26, 1024]
