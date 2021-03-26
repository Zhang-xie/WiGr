config = [
# aril in server: /media/yk/Samsung_T5
# widar in server: /media/yk/Samsung_T5/Widar3.0ReleaseData + /np_f2  or  /np_f_denoise  or /np_f_denoise_2
# csi_301 in server : /media/yk/Samsung_T5

# ResNet Widar amplitude :
            # {
            #      'source_data_config':{'roomid': [1], 'userid':None, 'location': [1], 'orientation': [1],'receiverid': [1],'sampleid':None,},
            #      'target_data_config':{'roomid': [2],  'userid':None, 'location':[1], 'orientation': [1],'receiverid': [1],'sampleid':None,},
            #      'data_sample_config':{'root':"/media/yk/Samsung_T5/Widar3.0ReleaseData/np_f2/",'dataset':"widar",'data_shape':'1D','chunk_size': 50, 'num_shot': 1, 'batch_size': 5, 'mode': 'amplitude','align':False},
            #      'encoder_config':{'model_name':'PrototypicalResNet'}, 
            #      'PrototypicalResNet_config':{'layers':[1,1,1],'strides':[1,2,2],'inchannel':90,'groups':3,},
            #      'metric_config':{'metric_method':'Euclidean','num_class_linear_flag':None,'combine':False,},
            #      'max_epochs':2,'ex_repeat':2,
            #  },

            {
                 'source_data_config':{'roomid': None, 'userid':None, 'location':None, 'orientation': None,'receiverid': None,'sampleid':None,},
                 'target_data_config':{'roomid': None,  'userid':None, 'location':None, 'orientation': None,'receiverid': None,'sampleid':None,},
                 'data_sample_config':{'root':"/media/yk/Samsung_T5/",'dataset':"aril",'data_shape':'1D','chunk_size': 50, 'num_shot': 1, 'batch_size': 3, 'mode': 'amplitude','align':True},
                 'encoder_config':{'model_name':'PrototypicalResNet'},  
                 'PrototypicalResNet_config':{'layers':[1,1,1],'strides':[1,2,2],'inchannel':52,'groups':1,},
                 'metric_config':{'metric_method':'cosine','num_class_linear_flag':None,'combine':False,},
                 'max_epochs':2,'ex_repeat':2,
            },


            {
                 'source_data_config':{'roomid': [0], 'userid':None, 'location': [0], 'orientation': None,'receiverid': None,'sampleid':None,},
                 'target_data_config':{'roomid': [0],  'userid':None, 'location':[0], 'orientation': None,'receiverid': None,'sampleid':None,},
                 'data_sample_config':{'root':"/media/yk/Samsung_T5",'dataset':"csi_301",'data_shape':'1D','chunk_size': 50, 'num_shot': 1, 'batch_size': 5, 'mode': 'amplitude','align':True},
                 'encoder_config':{'model_name':'PrototypicalResNet'},  
                 'PrototypicalResNet_config':{'layers':[1,1,1],'strides':[1,2,2],'inchannel':342,'groups':3,},
                 'metric_config':{'metric_method':'Euclidean','num_class_linear_flag':6,'combine':False,},
                 'max_epochs':2,'ex_repeat':2,
             },



]