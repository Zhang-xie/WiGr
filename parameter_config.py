config = [
            {
                 'source_data_config':{'roomid': None, 'userid':None, 'location': None, 'orientation': None,'receiverid': None,'sampleid':None,},
                 'target_data_config':{'roomid': None,  'userid':None, 'location':[2], 'orientation': None,'receiverid': None,'sampleid':None,},
                 'data_sample_config':{'root':"/media/yk/Samsung_T5/",'dataset':"aril",'data_shape':'1D','chunk_size': 50, 'num_shot': 1, 'batch_size': 5, 'mode': 'amplitude','align':True},
                 'encoder_config':{'model_name':'PrototypicalResNet'},  # PrototypicalCnnLstmNet,PrototypicalMobileNet
                 'PrototypicalResNet_config':{'layers':[1,1,1],'strides':[1,2,2],'inchannel':52,'groups':1,},
                 'PrototypicalCnnLstmNet_config':{'in_channel_cnn':3,'out_feature_dim_cnn':128,'out_feature_dim_lstm':256,'num_lstm_layer':2},
                 'PrototypicalMobileNet_config':{'width_mult':0.5,'inchannel':3,},
                 'metric_config':{'metric_method':'Euclidean','num_class_linear_flag':None,'combine':False,},
                 'max_epochs':200,'ex_repeat':10,
             },

{
                 'source_data_config':{'roomid': [0], 'userid':[0], 'location': [0], 'orientation': None,'receiverid': None,'sampleid':None,},
                 'target_data_config':{'roomid': [1],  'userid':[0], 'location':[0], 'orientation': None,'receiverid': None,'sampleid':None,},
                 'data_sample_config':{'root':"/media/yk/Samsung_T5",'dataset':"csi_301",'data_shape':'1D','chunk_size': 50, 'num_shot': 5, 'batch_size': 5, 'mode': 'amplitude','align':True},
                 'encoder_config':{'model_name':'PrototypicalResNet'},  # PrototypicalCnnLstmNet,PrototypicalMobileNet
                 'PrototypicalResNet_config':{'layers':[1,1,1],'strides':[1,2,2],'inchannel':342,'groups':3,},
                 'PrototypicalCnnLstmNet_config':{'in_channel_cnn':3,'out_feature_dim_cnn':256,'out_feature_dim_lstm':512,'num_lstm_layer':3},
                 'PrototypicalMobileNet_config':{'width_mult':1.0,'inchannel':3,},
                 'metric_config':{'metric_method':'Euclidean','num_class_linear_flag':None,'combine':False,},
                 'max_epochs':300,'ex_repeat':10,
             },

{
                 'source_data_config':{'roomid': [0], 'userid':[0], 'location': [0], 'orientation': None,'receiverid': None,'sampleid':None,},
                 'target_data_config':{'roomid': [1],  'userid':[0], 'location':[0], 'orientation': None,'receiverid': None,'sampleid':None,},
                 'data_sample_config':{'root':"/media/yk/Samsung_T5",'dataset':"csi_301",'data_shape':'1D','chunk_size': 50, 'num_shot': 5, 'batch_size': 5, 'mode': 'amplitude','align':True},
                 'encoder_config':{'model_name':'PrototypicalResNet'},  # PrototypicalCnnLstmNet,PrototypicalMobileNet
                 'PrototypicalResNet_config':{'layers':[1,1,1],'strides':[1,2,2],'inchannel':342,'groups':3,},
                 'PrototypicalCnnLstmNet_config':{'in_channel_cnn':3,'out_feature_dim_cnn':256,'out_feature_dim_lstm':512,'num_lstm_layer':3},
                 'PrototypicalMobileNet_config':{'width_mult':1.0,'inchannel':3,},
                 'metric_config':{'metric_method':'relation_1D','num_class_linear_flag':None,'combine':False,},
                 'max_epochs':300,'ex_repeat':10,
             },

]