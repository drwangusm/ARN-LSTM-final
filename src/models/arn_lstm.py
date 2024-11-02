from keras.layers import Dense, Input, Dropout
from keras.layers import LSTM, TimeDistributed, Lambda, Concatenate, Average
from keras.models import Model
from keras import backend as K
from models.attention import ARNAttention

from . import rn
import tensorflow as tf

# +--------------------+
# |    Input Layer     |
# | (seq_len, num_objs,|
# |  object_shape)     |
# +---------+----------+
#           |
#           v
# +--------------------+
# |  TimeDistributed   |
# |    (f_phi model)   |
# +---------+----------+
#           |
#           v
# +--------------------+
# |       LSTM 1       |
# |  (256 units,       |
# |  dropout=0.1)      |
# +---------+----------+
#           |
#           v
# +--------------------+
# |       LSTM 2       |
# |  (256 units,       |
# |  dropout=0.1)      |
# |     [Optional]     |
# +---------+----------+
#           |
#           v
# +--------------------+
# |      Dense Layer   |
# |     (softmax)      |
# +--------------------+
#           |
#           v
# +--------------------+
# |      Output        |
# +--------------------+


def get_irn(num_objs, object_shape, prune_at_layer=None, 
        **irn_kwargs):
    
    irn = rn.f_phi(num_objs, object_shape, **irn_kwargs)
    
    if prune_at_layer is not None:
        for layer in irn.layers[::-1]: # reverse looking for desired f_phi_fc* layer
            if layer.name.startswith(prune_at_layer) or layer.name.endswith(prune_at_layer):
                top_layer_out = layer.output
                break
        irn = Model(inputs=irn.input, outputs=top_layer_out)
    
    return irn

def g_theta_lstm(seq_len, object_shape, kernel_init, drop_rate, 
        prune_at_layer=None, **g_theta_kwargs):
    g_theta_model = rn.g_theta(object_shape, kernel_init, **g_theta_kwargs)
    
    if prune_at_layer is not None:
        # Reverse looking for desired g_theta_fc* layer
        for layer in g_theta_model.layers[::-1]: 
            if layer.name.endswith(prune_at_layer):
                top_layer_out = layer.output
                break
        g_theta_model = Model(inputs=g_theta_model.input, outputs=top_layer_out)
    
    # Wrapping with timedist
    input_g_theta = Input(shape=((2,)+object_shape))
    slice = Lambda(lambda x: [x[:,0], x[:,1]] )(input_g_theta)
    g_theta_model_out = g_theta_model(slice)
    merged_g_theta_model = Model(inputs=input_g_theta, outputs=g_theta_model_out)
    
    temporal_input = Input(shape=((seq_len, 2,) + object_shape))
    x = TimeDistributed(merged_g_theta_model)(temporal_input)
    
    x = LSTM(500, dropout=drop_rate, return_sequences=True)(x)
    g_theta_lstm_model = Model(inputs=temporal_input, outputs=x, name='g_theta_lstm')
    
    return g_theta_lstm_model

def average_per_sequence(tensors_list):
    expanded_dims = [ K.expand_dims(t, axis=1) for t in tensors_list ]
    single_tensor = K.concatenate(expanded_dims, axis=1)
    
    averages_per_seq = K.mean(single_tensor, axis=1)
    return averages_per_seq

def create_relationships(rel_type, g_theta_model, temp_input, p1_joints, p2_joints, use_attention=False, use_relations=True, attention_proj_size=None, return_attention=False):
    num_objs = int(temp_input.shape[2])//2
    g_theta_outs = []

    if not use_relations:
        for object_i in p1_joints:
            g_theta_outs.append(g_theta_model([object_i]))

        if use_attention:
            # Output may be tuple if return_attention is true, second element is attention vector
            return ARNAttention(projection_size=attention_proj_size, return_attention=return_attention)(g_theta_outs)
        else:
            rel_out = Average()(g_theta_outs)

        return rel_out
    
    if rel_type == 'inter':
        # All joints from person1 connected to all joints of person2, and back
        g_theta_outs = []
        
        for idx_i in range(num_objs): # Indexes Person 1
            for idx_j in range(num_objs, num_objs*2): # Indexes Person 2
                pair_name = 'pair_p0-j{}_p1-j{}'.format(idx_i, idx_j-num_objs)
                slice = Lambda(lambda x: K.concatenate([x[:,:,idx_i:idx_i+1], 
                    x[:,:,idx_j:idx_j+1]], axis=2), name=pair_name)(temp_input)
                g_theta_outs.append(g_theta_model(slice))
        for idx_j in range(num_objs, num_objs*2):
            for idx_i in range(num_objs):
                pair_name = 'pair_p1-j{}_p0-j{}'.format(idx_j-num_objs, idx_i)
                slice = Lambda(lambda x: K.concatenate([x[:,:,idx_j:idx_j+1], 
                    x[:,:,idx_i:idx_i+1]], axis=2), name=pair_name )(temp_input)
                g_theta_outs.append(g_theta_model(slice))
        
        g_theta_merged_out = Lambda(average_per_sequence, name='avg_seqs')(g_theta_outs)
    elif rel_type == 'indivs' or rel_type == 'intra':
        # All joints from person connected to all other joints of itself
        g_theta_indiv1_outs = []
        for idx_i in range(num_objs): # Indexes Person 1
            for idx_j in range(idx_i+1, num_objs):
                pair_name = 'pair_p0-j{}_p0-j{}'.format(idx_i, idx_j)
                slice = Lambda(lambda x: K.concatenate([x[:,:,idx_i:idx_i+1], 
                    x[:,:,idx_j:idx_j+1]], axis=2), 
                    name=pair_name )(temp_input)
                g_theta_indiv1_outs.append(g_theta_model(slice))
                
        g_theta_indiv2_outs = []
        for idx_i in range(num_objs, num_objs*2): # Indexes Person 2
            for idx_j in range(idx_i+1, num_objs*2):
                pair_name = 'pair_p1-j{}_p1-j{}'.format(idx_i-num_objs, idx_j-num_objs)
                slice = Lambda(lambda x: K.concatenate([x[:,:,idx_i:idx_i+1], 
                    x[:,:,idx_j:idx_j+1]], axis=2), 
                    name=pair_name )(temp_input)
                g_theta_indiv2_outs.append(g_theta_model(slice))
        
        indiv1_avg = Lambda(average_per_sequence, name='avg_seqs_p0')(g_theta_indiv1_outs)
        indiv2_avg = Lambda(average_per_sequence, name='avg_seqs_p1')(g_theta_indiv2_outs)
        g_theta_merged_out = Concatenate()([indiv1_avg, indiv2_avg])
    else:
        raise ValueError("Invalid rel_type:"+rel_type)
        
    return g_theta_merged_out

def create_timedist_top(input_top, kernel_init, drop_rate=0, fc_units=[500,100,100],
        fc_drop=False):
    x = TimeDistributed(
        Dropout(drop_rate), name='timedist_dropout')(input_top)
    x = TimeDistributed(
        Dense(fc_units[0], activation='relu', kernel_initializer=kernel_init), 
        name='timedist_fc1')(x)
    if fc_drop:
        x = TimeDistributed(Dropout(drop_rate), name='timedist_dropout_1')(x)
    x = TimeDistributed(
        Dense(fc_units[1], activation='relu', kernel_initializer=kernel_init), 
        name='timedist_fc2')(x)
    if fc_drop:
        x = TimeDistributed(Dropout(drop_rate), name='timedist_dropout_2')(x)
    x = TimeDistributed(
        Dense(fc_units[2], activation='relu', kernel_initializer=kernel_init), 
        name='timedist_fc3')(x)
    
    return x

def get_model(num_objs, object_shape, output_size, seq_len=4, 
        num_lstms=1, prune_at_layer=None, lstm_location='top',
        kernel_init_type='TruncatedNormal', kernel_init_param=0.045, kernel_init_seed=None,
        **irn_kwargs):
    
    drop_rate = irn_kwargs.get('drop_rate', 0)
    
    kernel_init = rn.get_kernel_init(kernel_init_type, param=kernel_init_param, 
        seed=kernel_init_seed)

    if irn_kwargs['rel_type'] == 'joint_stream' or irn_kwargs['rel_type'] == 'temp_stream':

        temp_input = Input(shape=((seq_len, num_objs,) + object_shape))
        if lstm_location == 'top': # After f_phi
            irn_model = get_irn(num_objs, object_shape, prune_at_layer=prune_at_layer,
                                kernel_init=kernel_init, **irn_kwargs)

        if 'return_attention' in irn_kwargs and irn_kwargs['return_attention']:
            print('implement later')
        else:
            input_irn = Input(shape=((num_objs,)+object_shape))
            slice = Lambda(lambda x: [ x[:,i] for i in range(num_objs)])(input_irn)
            irn_model_out = irn_model(slice)
            merged_irn_model = Model(inputs=input_irn, outputs=irn_model_out)
            x = TimeDistributed(merged_irn_model)(temp_input)
            if num_lstms == 2:
                x = LSTM(256, dropout=drop_rate, return_sequences=True)(x)
            x = LSTM(256, dropout=drop_rate)(x)

            out_softmax = Dense(output_size, activation='softmax',
                                kernel_initializer=kernel_init, name='model')(x)

            model = Model(inputs=temp_input, outputs=out_softmax, name="temp_rel_net")

            return model
    else:
        temp_input = Input(shape=((seq_len, num_objs*2,) + object_shape))
        if lstm_location == 'top': # After f_phi
            irn_model = get_irn(num_objs, object_shape, prune_at_layer=prune_at_layer,
                    kernel_init=kernel_init, **irn_kwargs)
        # Creating model with merged input then slice, to apply TimeDistributed
        input_irn = Input(shape=((num_objs*2,)+object_shape))
        slice = Lambda(lambda x: [ x[:,i] for i in range(num_objs*2) ])(input_irn)
        irn_model_out = irn_model(slice)
        merged_irn_model = Model(inputs=input_irn, outputs=irn_model_out)
        # Wrapping merged model with TimeDistributed
        x = TimeDistributed(merged_irn_model)(temp_input)
   
        if num_lstms == 2:
            x = LSTM(256, dropout=drop_rate, return_sequences=True)(x)
        x = LSTM(256, dropout=drop_rate)(x)
        out_softmax = Dense(output_size, activation='softmax',
            kernel_initializer=kernel_init, name='softmax')(x)
        model = Model(inputs=temp_input, outputs=out_softmax, name="temp_rel_net")
        return model

def get_fusion_model(new_arch, output_size, seq_len, train_kwargs,
        models_kwargs, weights_filepaths, freeze_g_theta=False, fuse_at_fc1=False, avg_at_end=False):
    print("===Into get_fusion_model function===")
    print("new_arch:",new_arch)
    print("output_size:",output_size)
    print("seq_len:",seq_len)
    print("train_kwargs:",train_kwargs)
    print("models_kwargs:",models_kwargs)
    print("weights_filepaths:",weights_filepaths)
    print("freeze_g_theta:",freeze_g_theta)
    print("fuse_at_fc1:",fuse_at_fc1)
    print("avg_at_end:",avg_at_end)
    prunned_models = []
    for model_kwargs, weights_filepath in zip(models_kwargs, weights_filepaths):
        temp_model = get_model(output_size=output_size, seq_len=seq_len, **model_kwargs)
        
        if weights_filepath != []:
            temp_model.load_weights(weights_filepath)
        for layer in temp_model.layers: # Looking for time_distributed layer
            if layer.name.startswith('time_distributed'):
                time_distributed_layer = layer
                break
        
        model = time_distributed_layer.layer.get_layer('f_phi')
        
        model_inputs = []
        for layer in model.layers:
            if layer.name.startswith('person') or layer.name.startswith('joint'):
                model_inputs.append(layer.input)
        
        if not fuse_at_fc1 and not avg_at_end:
            for layer in model.layers[::-1]: # reverse looking for last pool layer
                if layer.name.startswith(('average','concatenate','irn_attention')):
                    out_pool = layer.output
                    break
            prunned_model = Model(inputs=model_inputs, outputs=out_pool)
        elif fuse_at_fc1: # Prune keeping dropout + f_phi_fc1
            for layer in model.layers[::-1]: # reverse looking for last f_phi_fc1 layer
                if layer.name.startswith(('f_phi_fc1')):
                    out_f_phi_fc1 = layer.output
                    break
            prunned_model = Model(inputs=model_inputs, outputs=out_f_phi_fc1)

        elif avg_at_end:
            prunned_model = Model(inputs=model_inputs, outputs=model.outputs)

        if freeze_g_theta:
            for layer in prunned_model.layers: # Freezing model
                layer.trainable = False
        prunned_models.append(prunned_model)
    
    # Train params
    drop_rate = train_kwargs.get('drop_rate', 0.1)
    kernel_init_type = train_kwargs.get('kernel_init_type', 'TruncatedNormal')
    kernel_init_param = train_kwargs.get('kernel_init_param', 0.045)
    kernel_init_seed = train_kwargs.get('kernel_init_seed')
    
    kernel_init = rn.get_kernel_init(kernel_init_type, param=kernel_init_param, 
        seed=kernel_init_seed)

    if new_arch:

        joint_stream_objects = []
        temp_stream_objects = []

        for i in range(models_kwargs[0]['num_objs']):
            obj_joint = Input(shape=models_kwargs[0]['object_shape'], name="joint_stream_object"+str(i))
            joint_stream_objects.append(obj_joint)

        for i in range(models_kwargs[1]['num_objs']):
            obj_temp = Input(shape=models_kwargs[1]['object_shape'], name="temp_stream_object"+str(i))
            temp_stream_objects.append(obj_temp)

        inputs = joint_stream_objects + temp_stream_objects
        inputs_list = [joint_stream_objects, temp_stream_objects]
        models_outs = [ prunned_models[0](joint_stream_objects), prunned_models[1](temp_stream_objects) ]

    else:
        person1_joints = []
        person2_joints = []
        num_objs = model_kwargs['num_objs']
        object_shape = model_kwargs['object_shape']
        for i in range(num_objs):
            object_i = Input(shape=object_shape, name="person1_object"+str(i))
            object_j = Input(shape=object_shape, name="person2_object"+str(i))
            person1_joints.append(object_i)
            person2_joints.append(object_j)

        inputs = person1_joints + person2_joints
        models_outs = [ m(inputs) for m in prunned_models ]

    num_objs = model_kwargs['num_objs']
    object_shape = model_kwargs['object_shape']
    print("model_kwargs['num_objs']==",num_objs)
    print("model_kwargs['object_shape']==",object_shape)

    if not avg_at_end:
        x = Concatenate()(models_outs)
        # Building top and Model
        top_kwargs = rn.get_relevant_kwargs(model_kwargs, rn.create_top)
        out_rn = rn.create_top(x, kernel_init, **top_kwargs)
        irn_model = Model(inputs=inputs, outputs=out_rn)

        slices = []
        inputs_irn =[]
        temp_inputs = []
        xs = []
        if new_arch:
            for stream_kwarg, stream_input, stream_output, stream_model  in zip(models_kwargs, inputs_list, models_outs, prunned_models):

                input_irn = Input(shape=((stream_kwarg['num_objs'],) + stream_kwarg['object_shape']))
                slice = Lambda(lambda x: [ x[:,i] for i in range(stream_kwarg['num_objs'])])(input_irn)
                irn_model_out = stream_model(slice)
                temp_input = Input(shape=((seq_len, stream_kwarg['num_objs'],) + stream_kwarg['object_shape']))
                merged_irn_model = Model(inputs=input_irn, outputs=irn_model_out)
                x = TimeDistributed(merged_irn_model)(temp_input)
                xs.append(x)
                temp_inputs.append(temp_input)
                slices.append(slice)

            x = Concatenate()(xs)
            top_kwargs = rn.get_relevant_kwargs(model_kwargs, rn.create_top)
            out_rn = rn.create_top(x, kernel_init, **top_kwargs)
            lstm = LSTM(256, dropout=drop_rate)(out_rn)
            out_softmax = Dense(output_size, activation='softmax',
                                kernel_initializer=kernel_init, name='softmax')(lstm)
            model = Model(inputs=temp_inputs, outputs=out_softmax, name="fused_temp_rel_net")

        else:
            x = Concatenate()(models_outs)
            # Building top and Model
            top_kwargs = rn.get_relevant_kwargs(model_kwargs, rn.create_top)
            out_rn = rn.create_top(x, kernel_init, **top_kwargs)
            irn_model = Model(inputs=inputs, outputs=out_rn)
            input_irn = Input(shape=((num_objs*2,)+object_shape))
            slice = Lambda(lambda x: [ x[:,i] for i in range(num_objs*2)])(input_irn)
            irn_model_out = irn_model(slice)
            merged_irn_model = Model(inputs=input_irn, outputs=irn_model_out)
            temp_input = Input(shape=((seq_len, num_objs*2,) + object_shape))

            x = TimeDistributed(merged_irn_model)(temp_input)
            lstm = LSTM(256, dropout=drop_rate)(x)
            out_softmax = Dense(output_size, activation='softmax',
                kernel_initializer=kernel_init, name='softmax')(lstm)
            model = Model(inputs=temp_input, outputs=out_softmax, name="fused_temp_rel_net")

    else:
        complete_models = []
        inputs = []
        for stream_kwarg, stream_input, stream_output, stream_model in zip(models_kwargs, inputs_list, models_outs, prunned_models):
            print("stream_kwarg['num_objs']:",stream_kwarg['num_objs'])
            input_irn = Input(shape=((stream_kwarg['num_objs'],) + stream_kwarg['object_shape']))
            slice = Lambda(lambda x: [ x[:,i] for i in range(stream_kwarg['num_objs'])])(input_irn)
            irn_model_out = stream_model(slice)
            merged_irn_model = Model(inputs=input_irn, outputs=irn_model_out)
            temp_input = Input(shape=((seq_len, stream_kwarg['num_objs'],) + stream_kwarg['object_shape']))
            x = TimeDistributed(merged_irn_model)(temp_input)
            lstm = LSTM(256, dropout=drop_rate)(x)
            complete_models.append(lstm)
            inputs.append(temp_input)

        out = Average()(complete_models)
        out_softmax = Dense(output_size, activation='softmax', kernel_initializer=kernel_init)(out)

        model = Model(inputs=inputs, outputs=out_softmax, name="fused_temp_rel_net")

    print("===Finished get_fusion_model, model:",model)
    return model


