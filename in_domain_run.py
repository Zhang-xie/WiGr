from torch.utils.data import TensorDataset, DataLoader
import torch
import argparse
from collections import defaultdict
import models
import DataSet
import pytorch_lightning as pl
from pathlib import Path
import torch.utils.data as tdata
import matplotlib as plt

import numpy as np
from evaluation import confmat

from parameter_config import config


def run(args,setting):
    # torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    source_data = DataSet.create(name=args.dataset, root=args.root,roomid=args.train_roomid,
                                 userid=args.train_userid,
                                 location=args.train_location,orientation=args.train_orientation,
                                 receiverid=args.train_receiverid,sampleid=args.train_sampleid,
                                 data_shape=args.data_shape,chunk_size=args.chunk_size,num_shot=args.num_shot,
                                 batch_size=args.batch_size,mode=args.mode)

    target_data = DataSet.create(name=args.dataset, root=args.root,roomid=args.test_roomid,
                                 userid=args.test_userid,
                                 location=args.test_location,orientation=args.test_orientation,
                                 receiverid=args.test_receiverid,sampleid=args.test_sampleid,
                                 data_shape=args.data_shape,chunk_size=args.chunk_size,num_shot=args.num_shot,
                                 batch_size=args.batch_size,mode=args.mode)


    length=len(source_data)
    train_size =int(0.8*length)
    validate_size = length - train_size
    # print(length,train_size,validate_size)
    #first param is data set to be saperated, the second is list stating how many sets we want it to be.
    train_set,validate_set=torch.utils.data.random_split(source_data,[train_size,validate_size])
    # print(train_set,validate_set)

    tr_loader = DataLoader(dataset=train_set,collate_fn=lambda x:x)
    te_loader = DataLoader(dataset=validate_set, collate_fn=lambda x:x)

    data_model_match = True  # Whether the data format matches the model
    if args.model_name == 'PrototypicalResNet':
        if args.data_shape != '1D':
            data_model_match = False
        else:
            model = models.create(name=args.model_name,
                                  layers=args.layers,
                                  strides=args.strides,
                                  inchannel=args.ResNet_inchannel,
                                  groups=args.groups,
                                  align=args.align,
                                  metric_method=args.metric_method,
                                  k_shot=args.num_shot,
                                  num_class_linear_flag=args.num_class_linear_flag,
                                  combine=args.combine)
    elif args.model_name == 'PrototypicalCnnLstmNet':
        if args.data_shape == 'split':
            model = models.create(name=args.model_name,
                                  in_channel_cnn=args.in_channel_cnn,
                                  out_feature_dim_cnn=args.out_feature_dim_cnn,
                                  out_feature_dim_lstm=args.out_feature_dim_lstm,
                                  num_lstm_layer=args.num_lstm_layer,
                                  metric_method=args.metric_method,
                                  k_shot=args.num_shot,
                                  num_class_linear_flag=args.num_class_linear_flag,
                                  combine=args.combine)
        else:
            data_model_match = False
    elif args.model_name == 'PrototypicalMobileNet':
        if args.data_shape != '2D':
            data_model_match = False
        else:
            model = models.create(name=args.model_name,
                                  width_mult=args.width_mult,
                                  inchannel=args.MobileNet_inchannel,
                                  align=args.out_feature_dim_lstm,
                                  metric_method=args.metric_method,
                                  k_shot=args.num_shot,
                                  num_class_linear_flag=args.num_class_linear_flag,
                                  combine=args.combine)
    else:
        model = models.create(name=args.model_name,
                              in_channel_cnn=args.in_channel_cnn,
                              out_feature_dim_cnn=args.out_feature_dim_cnn,
                              out_feature_dim_lstm=args.out_feature_dim_lstm,
                              num_lstm_layer=args.num_lstm_layer,
                              metric_method=args.metric_method,
                              k_shot=args.num_shot,
                              num_class_linear_flag=args.num_class_linear_flag,
                              combine=args.combine)

    if data_model_match:
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint( monitor='GesVa_loss', save_last =False, save_top_k =0)
        trainer = pl.Trainer(callbacks=[checkpoint_callback,],log_every_n_steps=1,max_epochs=args.max_epochs,gpus=1)   # precision = 16
        trainer.fit(model,tr_loader,te_loader)

        # print([x.shape for x in model.comfmat_metric_all])
        cm_tensor_type = torch.stack(model.comfmat_metric_all)
        cm_numpy_type = cm_tensor_type.numpy()
        # cm = np.array(model.comfmat_metric_all)
        # print(type(model.comfmat_metric_all))
        # print(type(model.comfmat_metric_all[0]))
        from pathlib import Path
        np.save(Path(trainer.logger.log_dir)/'comfumat_metirc_all',cm_numpy_type)        
        # labels_name = np.array(['user1', 'user2', 'user3', 'user4', 'user5','ss'])
        # confmat.plot_confusion_matrix(cm,labels_name,title="test")

        fo = open(Path(trainer.logger.log_dir)/"config_parameter.txt", "w")
        fo.write(setting)
        fo.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WiFi CSI gesture sensing cross_domain experiment')

    parser.add_argument('--train_roomid', default=None,
                        help="the chosen rooms that the data selected from. Widar:[1,2,3]; Aril:useless; CSI_301: [0,1]")
    parser.add_argument('--train_userid', default=None,
                        help="the chosen users that the data selected from. Widar:1-17; Aril: useless; CSI_301: 0-4")
    parser.add_argument('--train_location', default=None,
                        help="the chosen locations that the data selected from. Widar:1-5; Aril: 1-16; CSI_301: 0-1/2")
    parser.add_argument('--train_orientation',  default=None,
                        help="the chosen orientations that the data selected from. Widar:1-5; Aril: useless; CSI_301: useless")
    parser.add_argument('--train_receiverid', default=None,
                        help="the chosen receivers that the data selected from. Widar:1-6; Aril: useless; CSI_301: useless")
    parser.add_argument('--train_sampleid', default=None,
                        help="the chosen samples that the data selected from. Widar:1-5; Aril: useless; CSI_301: useless")

    parser.add_argument('--test_roomid',  default=None,
                        help="the chosen rooms that the data selected from. Widar:[1,2,3]; Aril:useless; CSI_301: [0,1]")
    parser.add_argument('--test_userid',  default=None,
                        help="the chosen users that the data selected from. Widar:1-17; Aril: useless; CSI_301: 0-4")
    parser.add_argument('--test_location', default=None,
                        help="the chosen locations that the data selected from. Widar:1-5; Aril: 1-16; CSI_301: 0-1/2")
    parser.add_argument('--test_orientation',  default=None,
                        help="the chosen orientations that the data selected from. Widar:1-5; Aril: useless; CSI_301: useless")
    parser.add_argument('--test_receiverid',  default=None,
                        help="the chosen receivers that the data selected from. Widar:1-6; Aril: useless; CSI_301: useless")
    parser.add_argument('--test_sampleid', default=None,
                        help="the chosen samples that the data selected from. Widar:1-5; Aril: useless; CSI_301: useless")

    parser.add_argument('--root', default="F:/Widar3.0ReleaseData/np_f_denoise",
                        help="the storage path of datasets: Widar:F:/Widar3.0ReleaseData/np_f_denoise; ARIL,  the path is : ../Datasets")
    parser.add_argument('--dataset', default=None, help="the dataset name: aril,csi_301,widar")
    parser.add_argument('--data_shape', default=False, help="the data shape: 1D, 2D, split")
    parser.add_argument('--chunk_size', default=None,
                        help="if we using LSTM model, we can cut the time series into pices, the chunck size is the length of each pice.")
    parser.add_argument('--num_shot', default=None, help="the number of samples in support set of each class.")
    parser.add_argument('--batch_size', default=None, help="how much samples each class in one batch")
    parser.add_argument('--mode', default='amplitude', help="phase,amplitude,None; useless in aril.")
    parser.add_argument('--align', default=False,
                        help="the series has fixed length or not")

    parser.add_argument('--model_name', default=None,
                        help="name of model:PrototypicalResNet,PrototypicalCnnLstmNet")

    parser.add_argument('--layers', default=None,
                        help="ResNet layer set: eg, [1,1,1,1]")
    parser.add_argument('--strides', default=None,
                        help="the strides of convolution: eg, [1,1,2,2]")
    parser.add_argument('--ResNet_inchannel', default=None,
                        help="input channel of ResNet1D")
    parser.add_argument('--groups', default=None,
                        help="convolutional groups")

    parser.add_argument('--in_channel_cnn', default=None,
                        help="the input channel of cnn encoder")
    parser.add_argument('--out_feature_dim_cnn', default=None,
                        help="the dimensions of cnn")
    parser.add_argument('--out_feature_dim_lstm', default=None,
                        help="the output dimension of lstm")
    parser.add_argument('--num_lstm_layer', default=None,
                        help="the number of lstm layers in LSTM")

    parser.add_argument('--width_mult', default=None,
                        help="the rate of width expansion in MobileNet2D")
    parser.add_argument('--MobileNet_inchannel', default=None,
                        help="the input channel of MobileNet2D")

    parser.add_argument('--metric_method', default=None,
                        help="metric_method")
    parser.add_argument('--num_class_linear_flag', default=None,
                        help="the number of categories and the flag of using linear or not")
    parser.add_argument('--combine', default=False,
                        help="combine linear method with metric or not")

    parser.add_argument('--max_epochs', default=None, help="the max epoches")

    args = parser.parse_args()

    for i,setting in enumerate(config):
        source_data_config = setting['source_data_config']
        target_data_config = setting['target_data_config']
        data_sample_config = setting['data_sample_config']
        encoder_config = setting['encoder_config']
        PrototypicalResNet_config = setting['PrototypicalResNet_config']
        PrototypicalCnnLstmNet_config = setting['PrototypicalCnnLstmNet_config']
        PrototypicalMobileNet_config = setting['PrototypicalMobileNet_config']
        metric_config = setting['metric_config']
        max_epochs = setting['max_epochs']

        args.train_roomid = source_data_config['roomid']
        args.train_userid = source_data_config['userid']
        args.train_location = source_data_config['location']
        args.train_orientation = source_data_config['orientation']
        args.train_receiverid = source_data_config['receiverid']
        args.train_sampleid = source_data_config['sampleid']

        args.test_roomid = target_data_config['roomid']
        args.test_userid = target_data_config['userid']
        args.test_location = target_data_config['location']
        args.test_orientation = target_data_config['orientation']
        args.test_receiverid = target_data_config['receiverid']
        args.test_sampleid = target_data_config['sampleid']

        args.root = Path(data_sample_config['root'])
        args.dataset = data_sample_config['dataset']
        args.data_shape = data_sample_config['data_shape']
        args.chunk_size = data_sample_config['chunk_size']
        args.num_shot = data_sample_config['num_shot']
        args.batch_size = data_sample_config['batch_size']
        args.mode = data_sample_config['mode']
        args.align = data_sample_config['align']

        args.model_name = encoder_config['model_name']

        args.layers = PrototypicalResNet_config['layers']
        args.strides = PrototypicalResNet_config['strides']
        args.ResNet_inchannel = PrototypicalResNet_config['inchannel']
        args.groups = PrototypicalResNet_config['groups']

        args.in_channel_cnn = PrototypicalCnnLstmNet_config['in_channel_cnn']
        args.out_feature_dim_cnn = PrototypicalCnnLstmNet_config['out_feature_dim_cnn']
        args.out_feature_dim_lstm = PrototypicalCnnLstmNet_config['out_feature_dim_lstm']
        args.num_lstm_layer = PrototypicalCnnLstmNet_config['num_lstm_layer']

        args.width_mult = PrototypicalMobileNet_config['width_mult']
        args.MobileNet_inchannel = PrototypicalMobileNet_config['inchannel']

        args.metric_method = metric_config['metric_method']
        args.num_class_linear_flag = metric_config['num_class_linear_flag']
        args.combine = metric_config['combine']

        args.max_epochs = max_epochs
        ex_repeat  =  setting['ex_repeat']

        print("Experiment Setting Index:{}".format(i))
        print("Experiment Setting Config:{}".format(setting))

        for j in range(ex_repeat):
            run(args,setting)
        # print(args)

