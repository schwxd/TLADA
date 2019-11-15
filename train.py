import os
import math
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import UDAModel
from Coral import CORAL
import mmd
import loss
from functions import gradient_penalty, set_requires_grad
from triplet_loss import triplet_loss
from vis import draw_tsne, draw_confusion_matrix


TEST_INTERVAL = 10
VIS_INTERNAL = 100
title = 'the title'

def set_log_config(res_dir):
    logging.basicConfig(
            filename='{}/app.log'.format(res_dir),
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
    )

def train(config):
    model = UDAModel(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    if torch.cuda.is_available():
        model = model.cuda()

    if config['models'] == 'sourceonly':
        train_sourceonly(model, config)
    elif config['models'] == 'deepcoral':
        train_deepcoral(model, config)
    elif config['models'] == 'ddc':
        train_ddc(model, config)
    elif config['models'] in ['JAN', 'JAN_Linear', 'DAN', 'DAN_Linear']:
        train_dan_jan(model, config)
    elif config['models'] == 'dann':
        train_dann(model, config)
    elif config['models'] in ['dann_triplet', 'dann_triplet_src', 'dann_triplet_tgt']:
        train_dann_triplet(model, config)
    elif config['models'] == 'wasserstein':
        train_wt(model, config)
    else:
        pass


def test(model, data_loader, epoch):
    model.eval()

    loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for (features, labels) in data_loader:
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()

            preds = model.class_classify(features)
            loss += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            correct += pred_cls.eq(labels.data).cpu().sum().item()
            total += features.size(0)

    loss /= len(data_loader)
    accuracy = correct/total

    print("Epoch {}, {}/{}, Loss {}, Accuracy {:2%}".format(epoch, correct, total, loss, accuracy))
    logging.debug("Epoch {}, {}/{}, Loss {}, Accuracy {:2%}".format(epoch, correct, total, loss, accuracy))


def train_wt(model, config):

    triplet_type = config['triplet_type']
    gamma = config['w_gamma'] 
    weight_wd = config['w_weight']
    weight_triplet = config['t_weight']
    t_margin = config['t_margin']
    t_confidence = config['t_confidence']
    k_critic = 3
    k_clf = 1
    TRIPLET_START_INDEX = 90 

    if triplet_type == 'none':
        TEST_INTERVAL = 10
        res_dir = os.path.join(config['res_dir'], 'bs{}-lr{}-w{}-gamma{}'.format(config['batch_size'], config['lr'], weight_wd, gamma))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        modelpath = os.path.join(res_dir, "model.pth")
        EPOCH_START = 1

    else:
        TEST_INTERVAL = 1 
        w_dir = os.path.join(config['res_dir'], 'bs{}-lr{}-w{}-gamma{}'.format(config['batch_size'], config['lr'], weight_wd, gamma))
        if not os.path.exists(w_dir):
            os.makedirs(w_dir)
        res_dir = os.path.join(w_dir, '{}_t_weight{}_margin{}_confidence{}'.format(triplet_type, weight_triplet, t_margin, t_confidence))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        modelpath = os.path.join(w_dir, "model.pth")

        if os.path.exists(modelpath):
            model.load_state_dict(torch.load(modelpath))
            print('load model from {}'.format(modelpath))
            EPOCH_START = TRIPLET_START_INDEX
        else:
            EPOCH_START = 1

    set_log_config(res_dir)
    print('start epoch {}'.format(EPOCH_START))
    print('model path {}'.format(modelpath))
    print('triplet type {}'.format(triplet_type))
    print(config)

    logging.debug('train_wt')
    logging.debug(model.feature)
    logging.debug(model.class_classifier)
    logging.debug(model.domain_critic)
    logging.debug(config)

    criterion = torch.nn.CrossEntropyLoss()
    softmax_layer = nn.Softmax(dim=1)

    critic_opt = torch.optim.Adam(model.domain_critic.parameters(), lr=config['lr'])
    classifier_opt = torch.optim.Adam(model.class_classifier.parameters(), lr=config['lr'])
    #feature_opt = torch.optim.Adam(model.feature.parameters(), lr=config['lr']/10)
    feature_opt = torch.optim.Adam(model.feature.parameters(), lr=config['lr'])


    def train(model, config, epoch):
        model.feature.train()
        model.class_classifier.train()
        model.domain_critic.train()

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for step in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if step % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            # 1. train critic
            set_requires_grad(model.feature, requires_grad=False)
            set_requires_grad(model.class_classifier, requires_grad=False)
            set_requires_grad(model.domain_critic, requires_grad=True)
            with torch.no_grad():
                h_s = model.feature(data_source)
                h_s = h_s.view(h_s.size(0), -1)
                h_t = model.feature(data_target)
                h_t = h_t.view(h_t.size(0), -1)

            for j in range(k_critic):
                gp = gradient_penalty(model.domain_critic, h_s, h_t)
                critic_s = model.domain_critic(h_s)
                critic_t = model.domain_critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()
                critic_cost = -wasserstein_distance + gamma*gp

                critic_opt.zero_grad()
                critic_cost.backward()
                critic_opt.step()

                if step == 10 and j == 0:
                    print('EPOCH {}, DISCRIMINATOR: wd {}, gp {}, loss {}'.format(epoch, wasserstein_distance.item(), (gamma*gp).item(), critic_cost.item()))
                    logging.debug('EPOCH {}, DISCRIMINATOR: wd {}, gp {}, loss {}'.format(epoch, wasserstein_distance.item(), (gamma*gp).item(), critic_cost.item()))

            # 2. train feature and class_classifier
            set_requires_grad(model.feature, requires_grad=True)
            set_requires_grad(model.class_classifier, requires_grad=True)
            set_requires_grad(model.domain_critic, requires_grad=False)
            for _ in range(k_clf):
                h_s = model.feature(data_source)
                h_s = h_s.view(h_s.size(0), -1)
                h_t = model.feature(data_target)
                h_t = h_t.view(h_t.size(0), -1)

                source_preds = model.class_classifier(h_s)
                clf_loss = criterion(source_preds, label_source)
                wasserstein_distance = model.domain_critic(h_s).mean() - model.domain_critic(h_t).mean()

                if triplet_type != 'none' and epoch >= TRIPLET_START_INDEX:
                    target_preds = model.class_classifier(h_t)
                    target_labels = target_preds.data.max(1)[1]
                    target_logits = softmax_layer(target_preds)
                    if triplet_type == 'all':
                        triplet_index = np.where(target_logits.data.max(1)[0].cpu().numpy() > t_margin)[0]
                        images = torch.cat((h_s, h_t[triplet_index]), 0)
                        labels = torch.cat((label_source, target_labels[triplet_index]), 0)
                    elif triplet_type == 'src':
                        images = h_s
                        labels = label_source
                    elif triplet_type == 'tgt':
                        triplet_index = np.where(target_logits.data.max(1)[0].cpu().numpy() > t_confidence)[0]
                        images = h_t[triplet_index]
                        labels = target_labels[triplet_index]
                    elif triplet_type == 'sep':
                        triplet_index = np.where(target_logits.data.max(1)[0].cpu().numpy() > t_confidence)[0]
                        images = h_t[triplet_index]
                        labels = target_labels[triplet_index]
                        t_loss_sep, _ = triplet_loss(model.extract_feature, {"X": images, "y": labels}, t_confidence)
                        images = h_s
                        labels = label_source

                    t_loss, _ = triplet_loss(model.extract_feature, {"X": images, "y": labels}, t_margin)
                    loss = clf_loss + \
                        weight_wd * wasserstein_distance + \
                        weight_triplet * t_loss
                    if triplet_type == 'sep':
                        loss += t_loss_sep
                    feature_opt.zero_grad()
                    classifier_opt.zero_grad()
                    loss.backward()
                    feature_opt.step()
                    classifier_opt.step()

                    if step == 10:
                        print('EPOCH {}, CLASSIFIER: clf_loss {}, wd {}, t_loss {}, total loss {}'.format(
                            epoch, clf_loss.item(),
                            weight_wd * wasserstein_distance.item(),
                            weight_triplet * t_loss.item(),
                            loss.item()))
                        logging.debug('EPOCH {}, CLASSIFIER: clf_loss {}, wd {}, t_loss {}, total loss {}'.format(
                            epoch, clf_loss.item(),
                            weight_wd * wasserstein_distance.item(),
                            weight_triplet * t_loss.item(),
                            loss.item()))

                else:
                    loss = clf_loss + weight_wd * wasserstein_distance
                    feature_opt.zero_grad()
                    classifier_opt.zero_grad()
                    loss.backward()
                    feature_opt.step()
                    classifier_opt.step()

                    if step == 10:
                        print('EPOCH {}, CLASSIFIER: clf_loss {}, wd {},  loss {}'.format(
                            epoch, clf_loss.item(),
                            weight_wd * wasserstein_distance.item(),
                            loss.item()))
                        logging.debug('EPOCH {}, CLASSIFIER: clf_loss {}, wd {},  loss {}'.format(
                            epoch, clf_loss.item(),
                            weight_wd * wasserstein_distance.item(),
                            loss.item()))

    # pretrain(model, config, pretrain_epochs=20)
    for epoch in range(EPOCH_START, config['n_epochs'] + 1):
        train(model, config, epoch)
        if epoch % TEST_INTERVAL == 0:
            print('test on source_test_loader')
            test(model, config['source_test_loader'], epoch)
            # print('test on target_train_loader')
            # test(model, config['target_train_loader'], epoch)
            print('test on target_test_loader')
            test(model, config['target_test_loader'], epoch)
        if epoch % VIS_INTERNAL == 0:
            if triplet_type == 'none':
                title = '(a) WDGRL'
            else:
                title = '(b) TLADA'
            draw_confusion_matrix(model, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)
    if triplet_type == 'none':
        torch.save(model.state_dict(), modelpath)



def train_dan_jan(model, config):

    criterion = nn.CrossEntropyLoss()
    transfer_criterion = loss.loss_dict[config['models']]

    ## add additional network for some methods
    if config['models'] == "JAN" or config['models'] == "JAN_Linear":
        softmax_layer = nn.Softmax(dim=1).cuda()

    l2_decay = 5e-4
    momentum = 0.9

    res_dir = os.path.join(config['res_dir'], 'lr{}-mmdgamma{}'.format(config['lr'], config['mmd_gamma']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)

    logging.debug('train_dan_jan {}'.format(config['models']))
    logging.debug(model.feature)
    logging.debug(model.class_classifier)
    logging.debug(config)

    def train(model, config, epoch):
        model.class_classifier.train()
        model.feature.train()

        #LEARNING_RATE = config['lr'] / math.pow((1 + 10 * (epoch - 1) / config['n_epochs']), 0.75)
        LEARNING_RATE = config['lr']
        #print('epoch {}, learning rate{: .4f}'.format(epoch, LEARNING_RATE) )
        optimizer = torch.optim.SGD([
            {'params': model.feature.parameters()},
            {'params': model.class_classifier.parameters(), 'lr': LEARNING_RATE},
            ], lr= LEARNING_RATE / 1, momentum=momentum, weight_decay=l2_decay)

        # optimizer = optim.Adam(model.parameters(), lr=lr)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            optimizer.zero_grad()

            feature_source = model.feature(data_source)
            feature_source = feature_source.view(feature_source.size(0), -1)
            preds_source = model.class_classifier(feature_source)
            classifier_loss = criterion(preds_source, label_source)

            feature_target = model.feature(data_target)
            feature_target = feature_target.view(feature_target.size(0), -1)
            preds_target = model.class_classifier(feature_target)


            if config['models'] == "DAN" or config['models'] == "DAN_Linear":
                transfer_loss = transfer_criterion(feature_source, feature_target)
            elif config['models'] == "JAN" or config['models'] == "JAN_Linear":
                softmax_source = softmax_layer(preds_source)
                softmax_target = softmax_layer(preds_target)
                transfer_loss = transfer_criterion([feature_source, softmax_source], [feature_target, softmax_target])

            total_loss = config['mmd_gamma'] * transfer_loss + classifier_loss
            if i % 50 == 0:
                print('transfer_loss: {}, classifier_loss: {}, total_loss: {}'.format(transfer_loss.item(), classifier_loss.item(), total_loss.item()))

            total_loss.backward()
            optimizer.step()

    for epoch in range(1, config['n_epochs'] + 1):
        train(model, config, epoch)
        if epoch % TEST_INTERVAL == 0:
            print('test on source_test_loader')
            test(model, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(model, config['target_test_loader'], epoch)
        if epoch % VIS_INTERNAL == 0:
            draw_confusion_matrix(model, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)



def train_ddc(model, config):

    res_dir = os.path.join(config['res_dir'], 'lr{}-mmdgamma{}'.format(config['lr'], config['mmd_gamma']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)

    criterion = torch.nn.CrossEntropyLoss()
    l2_decay = 5e-4
    momentum = 0.9

    logging.debug('train_ddc')
    logging.debug(model.feature)
    logging.debug(model.class_classifier)
    logging.debug(config)

    optimizer = optim.Adam(
        list(model.feature.parameters()) + list(model.class_classifier.parameters()),
        lr = config['lr'])

    def train(model, config, epoch):
        model.class_classifier.train()
        model.feature.train()

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            optimizer.zero_grad()

            preds = model.class_classify(data_source)
            loss_cls = criterion(preds, label_source)
            
            source = model.feature(data_source)
            source = source.view(source.size(0), -1)
            target = model.feature(data_target)
            target = target.view(target.size(0), -1)
            loss_mmd = mmd.mmd_linear(source, target)

            loss = loss_cls + config['mmd_gamma'] * loss_mmd
            if i % 50 == 0:
                print('loss_cls {}, loss_mmd {}, gamma {}, total loss {}'.format(loss_cls.item(), loss_mmd.item(), config['mmd_gamma'], loss.item()))
            loss.backward()
            optimizer.step()

    for epoch in range(1, config['n_epochs'] + 1):
        train(model, config, epoch)
        if epoch % TEST_INTERVAL == 0:
            print('test on source_test_loader')
            test(model, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(model, config['target_test_loader'], epoch)
        if epoch % VIS_INTERNAL == 0:
            draw_confusion_matrix(model, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)


def train_deepcoral(model, config):

    res_dir = os.path.join(config['res_dir'], 'lr{}-mmdgamma{}'.format(config['lr'], config['mmd_gamma']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    criterion = torch.nn.CrossEntropyLoss()
    momentum = 0.9
    l2_decay = 5e-4

    set_log_config(res_dir)

    logging.debug('train_deepcoral')
    logging.debug(model.feature)
    logging.debug(model.class_classifier)
    logging.debug(config)

    optimizer = optim.Adam(
        list(model.feature.parameters()) + list(model.class_classifier.parameters()),
        lr = config['lr'])

    def train(model, config, epoch):
        model.class_classifier.train()
        model.feature.train()

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            optimizer.zero_grad()

            preds = model.class_classify(data_source)
            loss_cls = criterion(preds, label_source)
            
            source = model.feature(data_source)
            source = source.view(source.size(0), -1)
            target = model.feature(data_target)
            target = target.view(target.size(0), -1)
            loss_coral = CORAL(source, target)

            loss = loss_cls + config['mmd_gamma'] * loss_coral
            if i % 50 == 0:
                print('loss_cls {}, loss_coral {}, gamma {}, total loss {}'.format(loss_cls.item(), loss_coral.item(), config['mmd_gamma'], loss.item()))
            loss.backward()
            optimizer.step()

    for epoch in range(1, config['n_epochs'] + 1):
        train(model, config, epoch)
        if epoch % TEST_INTERVAL == 0:
            print('test on source_test_loader')
            test(model, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(model, config['target_test_loader'], epoch)
        if epoch % VIS_INTERNAL == 0:
            draw_confusion_matrix(model, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)



def train_sourceonly(model, config):

    res_dir = os.path.join(config['res_dir'], 'lr{}'.format(config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)
    logging.debug('train_sourceonly')
    logging.debug(config)
    logging.debug(model.feature)
    logging.debug(model.class_classifier)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model.feature.parameters()) + list(model.class_classifier.parameters()),
        lr = config['lr'])

    def train(model, config, epoch):
        model.class_classifier.train()
        model.feature.train()

        for step, (features, labels) in enumerate(config['source_train_loader']):
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()

            optimizer.zero_grad()
            preds = model.class_classify(features)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

    for epoch in range(1, config['n_epochs'] + 1):
        train(model, config, epoch)
        if epoch % TEST_INTERVAL == 0:
            print('test on source_test_loader')
            test(model, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(model, config['target_test_loader'], epoch)
        if epoch % VIS_INTERNAL == 0:
            draw_confusion_matrix(model, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            draw_tsne(model, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)


