# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""
from tensorflow.python.keras.layers import Dense, Flatten # Dense：全连接层，Flatten：扁平化层
from tensorflow.python.keras.models import Model # 将输入和输出定义为Keras模型

from ...feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features # SparseFeat：稀疏特征，VarLenSparseFeat：变长稀疏特征，DenseFeat：密集特征
from ...inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list
from ...layers.core import DNN, PredictionLayer # DNN：深度神经网络，PredictionLayer：预测层
from ...layers.sequence import AttentionSequencePoolingLayer # AttentionSequencePoolingLayer：自定义的注意力池化层，用于处理序列数据
from ...layers.utils import concat_func, combined_dnn_input # concat_func, combined_dnn_input：拼接不同特征或者层的函数


def DIN(dnn_feature_columns, history_feature_list, dnn_use_bn=False,
        dnn_hidden_units=(256, 128, 64), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
        att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
        task='binary'):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model. 一个可迭代对象，模型的输入特征列表
    :param history_feature_list: list,to indicate  sequence sparse field 一个列表，历史行为数据的特征列表（用于计算用户兴趣历史）
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net 一个布尔值，是否在DNN中使用Batch Normalization
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net一个列表， 其中元素为正整数或者为空列表，用于指定DNN网络中各隐藏层的神经元数量
    :param dnn_activation: Activation function to use in deep net 字符串类型，DNN网络的激活函数（比如relu）
    ###### 与注意力机制相关的配置  #########
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net 一个列表，其中元素为正整数，用于定义注意力网络中各层的神经元数量
    :param att_activation: Activation function to use in attention net 字符串类型，指定注意力网络中使用的激活函数，比如dice等
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit. 布尔值，决定是否对局部激活单元的注意力得分进行归一化操作
    #####################################
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN 浮点数，L2正则化项的系数（设置应用于DNN的L2正则化强度（
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector 浮点数，L2正则化项的系数，用于嵌入层
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate. 范围在[0,1)的浮点数，DNN中的dropout比率，代表DNN中神经元随机失活的概率
    :param seed: integer ,to use as random seed. 整数，随机种子，用于保证在模型训练等涉及随机操作时结果的可复现性
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss 字符串，任务类型（binary：二分类任务，regression：回归任务）
    :return: A Keras model instance. Keras模型实例，该模型基于给定的输入特征、配置参数等构建完成，可以用于后续的训练、预测等操作，其输入对应构建的各种输入特征，输出为符合 指定任务类型的预测结果

    """
    ############## 一、输入特征的构建与筛选 ################
    # 构建输入特征（稀疏特征、密集特征、变长稀疏特征）
    features = build_input_features(dnn_feature_columns)

    # 使用filter操作从dnn_feature_columns中筛选出类型为SparseFeat的特征列
    # 1. isinstance(x, SparseFeat)：判断x是否为SparseFeat类型，如果是则返回true
    # 2. filter：过滤后的迭代器，该迭代器包含所有满足Lambda条件的元素
    # 从dnn_feature_columns中筛选出类型为SparseFeat的特征列
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    # 从dnn_feature_columns中筛选出类型为DenseFeat的特征列
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    # 从dnn_feature_columns中筛选出类型为VarLenSparseFeat的特征列
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    # 从varlen_sparse_feature_columns中筛选出哪些特征是历史特征
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    ########### 二、输入数据的准备与嵌入矩阵的构建 ############
    # 字典，包含了所有输入特征的定义
    # features: {'user_id': <Keras Input Tensor>, 'age': <Keras Input Tensor>}
    inputs_list = list(features.values())

    # 嵌入矩阵，通常用于处理离散类别，通过嵌入将它们转化为稠密的向量表示
    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="")

    ########## 三、查找特征的嵌入向量 ####################
    # 查找嵌入向量
    # 1. query
    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
                                      history_feature_list, to_list=True)
    # 2. keys
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)
    # DNN部分的特征的嵌入向量
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    # 获取密集特征输入
    dense_value_list = get_dense_input(features, dense_feature_columns)

    ######## 四、处理变长稀疏特征 ##############
    # 处理变长稀疏特征
    # 1. 根据sparse_varlen_feature_columns查找变长稀疏特征的嵌入向量
    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    # 2. 对变长特征进行池化操作，通常用于序列数据
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)

    # 合并嵌入向量
    dnn_input_emb_list += sequence_embed_list

    ########## 五、合并特征和计算注意力 ###############
    # 合并不同类型的特征（concat_func将多个特征拼接成一个大向量）
    keys_emb = concat_func(keys_emb_list, mask=True) # 历史特征的拼接结果
    deep_input_emb = concat_func(dnn_input_emb_list) # 所有输入特征的拼接结果
    query_emb = concat_func(query_emb_list, mask=True) # 当前特征的拼接结果

    # 自定义的注意力层，用于处理序列数据（基于当前特征query_emb和历史特征keys_emb，通过注意力机制来计算加权的历史信息）
    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                         weight_normalization=att_weight_normalization, supports_masking=True)([
        query_emb, keys_emb])

    ######### 六、组合深度特征并通过DNN进行处理 ############
    # 将deep_input_emb和hist进行拼接
    deep_input_emb = concat_func([deep_input_emb, hist])
    # 将拼接后的特征展平
    deep_input_emb = Flatten()(deep_input_emb)
    # 将展平后的特征和密集特征dense_value_list组合，作为DNN的输入
    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    # DNN层（全连接神经网络层，将组合后的特征输入通过多个全连接层进行处理）
    # 1. dnn_hidden_units：隐藏层的单元数
    # 2. dnn_activation：激活函数
    # 3. l2_reg_dnn：DNN部分的L2正则化
    # 4. dnn_dropout：Dropout比例
    # 5. dnn_use_bn：是否使用BatchNormalization
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    ############ 七、输出层与最终预测 ################
    # 输出层是一个单神经元的全连接层，输入一个标量
    final_logit = Dense(1, use_bias=False)(output)

    # 预测层
    output = PredictionLayer(task)(final_logit)

    ############ 八、构建模型 ###################
    # 构建模型
    model = Model(inputs=inputs_list, outputs=output)
    return model
