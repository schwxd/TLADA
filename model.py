import torch
from torch import nn

from functions import ReverseLayerF

class UDAModel(nn.Module):
    def __init__(self, n_flattens, n_hiddens, n_class, bn=False):
        super(UDAModel, self).__init__()
        # feature之后的维度
        self.n_flattens = n_flattens
        # 全连接层的hidden size
        self.n_hiddens = n_hiddens
        # 分类器输出的类别个数
        self.n_class = n_class

        self.feature = nn.Sequential()
        self.class_classifier = nn.Sequential()
        self.domain_classifier = nn.Sequential()
        self.domain_critic = nn.Sequential()

        # features 使用1-D CNN提取特征
        self.feature.add_module('f_conv1', nn.Conv1d(1, 8, kernel_size=32, stride=2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop1', nn.Dropout(0.2))

        self.feature.add_module('f_conv2', nn.Conv1d(8, 16, kernel_size=16, stride=2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop2', nn.Dropout(0.2))

        self.feature.add_module('f_conv3', nn.Conv1d(16, 32, kernel_size=8, stride=2))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_pool3', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop3', nn.Dropout(0.2))

        self.feature.add_module('f_conv4', nn.Conv1d(32, 32, kernel_size=3, stride=2))
        self.feature.add_module('f_relu4', nn.ReLU(True))
        self.feature.add_module('f_pool4', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop4', nn.Dropout(0.2)) 

        # class_classifier 使用全连接层进行分类
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.n_flattens, self.n_hiddens))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(0.2))
        self.class_classifier.add_module('c_fc2', nn.Linear(self.n_hiddens, self.n_hiddens))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_drop2', nn.Dropout(0.2))
        self.class_classifier.add_module('c_fc3', nn.Linear(self.n_hiddens, self.n_class))

        # domain_classifier 使用全连接层进行领域分类
        # 注意区分最后输出是1还是2，loss对应选择BCELoss或CE
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.n_flattens, self.n_hiddens))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_drop1', nn.Dropout(0.2))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.n_hiddens, self.n_hiddens))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_drop2', nn.Dropout(0.2))
        self.domain_classifier.add_module('d_fc3', nn.Linear(self.n_hiddens, 2))

        # 用于w距离
        self.domain_critic = nn.Sequential()
        self.domain_critic.add_module('dc_fc1', nn.Linear(self.n_flattens, self.n_hiddens))
        self.domain_critic.add_module('dc_relu1', nn.ReLU(True))
        self.domain_critic.add_module('dc_drop1', nn.Dropout(0.2))
        self.domain_critic.add_module('dc_fc2', nn.Linear(self.n_hiddens, self.n_hiddens))
        self.domain_critic.add_module('dc_relu2', nn.ReLU(True))
        self.domain_critic.add_module('dc_drop2', nn.Dropout(0.2))
        self.domain_critic.add_module('dc_fc3', nn.Linear(self.n_hiddens, 1))

    def extract_feature(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x

    def class_classify(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.class_classifier(x)
        return x

    def dann(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(feature.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, feature

    def deepcoral(self, source, target):
        loss = 0
        source = self.feature(source)
        source = source.view(source.size(0), -1)

        if self.training == True:
            target = self.feature(target)
            target = target.view(target.size(0), -1)
            loss += CORAL(source, target)
        source = self.class_classifier(source)

        return source, loss

    def ddc(self, source, target):
        loss = 0
        source = self.feature(source)
        source = source.view(source.size(0), -1)

        if self.training == True:
            target = self.feature(target)
            target = target.view(target.size(0), -1)
            loss = mmd.mmd_linear(source, target)

        source = self.class_classifier(source)
        return source, loss

    def forward(self, x):
        print('forward called')
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        output = self.class_classifier(x)
        return output

    def output_num(self):
        return self.__in_features

