#coding:utf8
import tensorflow.compat.v1 as tf #使用1.0版本的方法
tf.disable_v2_behavior() #禁用2.0版本的方法
from base.herbRecommender import herbRecommender
from scipy.sparse import coo_matrix,hstack
#import tensorflow as tf
import numpy as np
from math import sqrt
import pandas as pds
import scipy.sparse as sp
# from spektral.layers import GraphAttention
from time import strftime,localtime,time
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dropout
import os

import networkx as nx
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# tf.compat.v1.set_random_seed(4321)
from util.io import FileIO

class HDCTI(herbRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(HDCTI, self).__init__(conf,trainingSet,testSet,fold)


    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        i=0
        for pair in self.data.trainingData:
            # symmetric matrix，对称矩阵
            if int(pair[2])!=0:
                row += [self.data.compound[pair[0]]]
                col += [self.data.protein[pair[1]]]
                entries += [1]
                i+=1
        print('i======i',i)
        # u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_diseases),dtype=np.float32)
        u_i_adj = coo_matrix((entries,(row,col)), shape=(self.num_compounds, self.num_proteins),dtype=np.float32)
        return u_i_adj

    def buildhcAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.hcassociation:
            # symmetric matrix
            #x=random.randint(0,1008)
            #y=random.randint(0,1192)
            #row += [x]
            #col += [y]
            row += [self.data.herb[pair[0]]]
            col += [self.data.compound[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_compounds),dtype=np.float32)
        return u_i_adj

    def buildcpAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.cpassociation:
            # symmetric matrix
            row += [self.data.compound[pair[0]]]
            col += [self.data.protein[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_compounds,self.num_proteins),dtype=np.float32)
        return u_i_adj

    def buildpdAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.pdassociation:
            # symmetric matrix
            #x=random.randint(0,7257)
            #y=random.randint(0,11070)
            #row += [x]
            #col += [y]
            row += [self.data.protein[pair[0]]]
            col += [self.data.disease[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_proteins,self.num_diseases),dtype=np.float32)
        return u_i_adj
    def buildJointAdjacency(self):
        indices = [[self.data.herb[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_herbs, self.num_diseases])
        return norm_adj

    def buildGraphAndPageRank(self, adjacency_matrix):
        G = nx.Graph()
        coo = adjacency_matrix.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            G.add_edge(i, j, weight=v)
        pr = nx.pagerank(G, alpha=0.85)
        return pr
    # def buildJointAdjacency(self):    #建立联合邻接矩阵
    #     indices = [[self.data.compound[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
    #     values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
    #               for item in self.data.trainingData]
    #     norm_adj = tf.SparseTensor(indices=indices, values=values,
    #                                dense_shape=[self.num_compounds, self.num_proteins])
    #     return norm_adj

    def initModel(self):
        super(HDCTI, self).initModel()
        #Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        #A=A.dot(A.transpose().dot(A))
        cp=self.buildcpAdjacencyMatrix()
        pd=self.buildpdAdjacencyMatrix()
        hc=self.buildhcAdjacencyMatrix()



        
        # 计算 hc 的转置
        H_c = hc.transpose()
        # 计算 H_c 中每一行的和，并将其形状调整为 (1, -1)
        D_hc_v = H_c.sum(axis=1).reshape(1, -1)
        # 计算 H_c 中每一列的和，并将其形状调整为 (1, -1)
        D_hc_e = H_c.sum(axis=0).reshape(1, -1)
        # 计算边的归一化矩阵
        temp1 = (H_c.multiply(1.0 / D_hc_e)).transpose()
        # 计算节点的归一化矩阵
        temp2 = (H_c.transpose().multiply(1.0 / D_hc_v)).transpose()
        # 将边的矩阵转换为 COO 格式
        edge = temp1.tocoo()
        # 将节点的矩阵转换为 COO 格式
        node = temp2.tocoo()
        # 获取边和节点的索引
        edge_indices = np.mat([edge.row, edge.col]).transpose()
        node_indices = np.mat([node.row, node.col]).transpose()
        # 构建稀疏张量表示边和节点
        H_e = tf.SparseTensor(edge_indices, edge.data.astype(np.float32), edge.shape)
        H_n = tf.SparseTensor(node_indices, node.data.astype(np.float32), node.shape)
        
        # 获取 pd 的矩阵
        P_d = pd
        # 计算 P_d 中每一行的和，并将其形状调整为 (1, -1)
        D_P_v = P_d.sum(axis=1).reshape(1, -1)
        # 计算 P_d 中每一列的和，并将其形状调整为 (1, -1)
        D_P_e = P_d.sum(axis=0).reshape(1, -1)
        # 计算边的矩阵
        temp1 = (P_d.multiply(1.0 / D_P_e)).transpose()
        # 计算节点的矩阵
        temp2 = (P_d.transpose().multiply(1.0 / D_P_v)).transpose()
        # 将边的矩阵转换为 COO 格式
        pd_edge = temp1.tocoo()
        # 将节点的矩阵转换为 COO 格式
        pd_node = temp2.tocoo()
        # 获取边和节点的索引
        A_pde = np.mat([pd_edge.row, pd_edge.col]).transpose()
        A_pdn = np.mat([pd_node.row, pd_node.col]).transpose()
        # 构建稀疏张量表示边和节点
        P_e = tf.SparseTensor(A_pde, pd_edge.data.astype(np.float32), pd_edge.shape)
        P_n = tf.SparseTensor(A_pdn, pd_node.data.astype(np.float32), pd_node.shape)




        #Build network
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer =tf.keras.initializers.glorot_normal()
        self.n_layer = 2
        self.weights={}
        self.attention_weights = {}
        attention_size = 64
        self.compound_attention_weights = []
        self.protein_attention_weights = []

        num_heads = 2  # Number of attention heads
        head_dim = self.emb_size // num_heads  # Dimension of each attention head



        pr_compound = self.buildGraphAndPageRank(cp)
        pr_protein = self.buildGraphAndPageRank(pd)
        pr_compound_embeddings = np.array([pr_compound.get(i, 0) for i in range(self.num_compounds)])
        pr_protein_embeddings = np.array([pr_protein.get(i, 0) for i in range(self.num_proteins)])
        pr_compound_embeddings = np.reshape(pr_compound_embeddings, (self.num_compounds, 1))
        pr_protein_embeddings = np.reshape(pr_protein_embeddings, (self.num_proteins, 1))


        initializer = tf.variance_scaling_initializer(scale=2.0)

        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_%d' % (i + 1))
            self.weights['layer_1_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_1_%d' % (i + 1))
            self.weights['layer_2_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_2_%d' % (i + 1))
            self.weights['layer_att_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_%d' %(i+1))
            self.attention_weights['compound' ] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='compound')
            self.attention_weights['protein'] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='protein')
            for h in range(num_heads):
                self.attention_weights['compound_q_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='compound_q_%d_%d' % (i + 1, h))
                self.attention_weights['compound_k_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='compound_k_%d_%d' % (i + 1, h))
                self.attention_weights['compound_v_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='compound_v_%d_%d' % (i + 1, h))

                self.attention_weights['protein_q_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='protein_q_%d_%d' % (i + 1, h))
                self.attention_weights['protein_k_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='protein_k_%d_%d' % (i + 1, h))
                self.attention_weights['protein_v_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='protein_v_%d_%d' % (i + 1, h))

        for i in range(2):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='g_W_%d_1' % (i + 1))
            # self.weights['gating_bias%d' % (i + 1)] = tf.Variable(tf.zeros([1, self.emb_size]),
            #                                                       name='g_W_b_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                       name='g_W_b_%d_1' % (i + 1))

        def multi_head_attention_compound(embeddings, attention_weights, num_heads, head_dim):
            attention_heads = []

            for h in range(num_heads):
                q = tf.matmul(embeddings, attention_weights['compound_q_%d_%d' % (i + 1, h)])
                k = tf.matmul(embeddings, attention_weights['compound_k_%d_%d' % (i + 1, h)])
                v = tf.matmul(embeddings, attention_weights['compound_v_%d_%d' % (i + 1, h)])

                attn_logits = tf.matmul(q, k, transpose_b=True)
                attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(float(head_dim)), axis=-1)
                attn_output = tf.matmul(attn_weights, v)

                attention_heads.append(attn_output)

            # Concatenate all attention heads
            concat_attention = tf.concat(attention_heads, axis=-1)
            return concat_attention

        def multi_head_attention_protein(embeddings, attention_weights, num_heads, head_dim):
            attention_heads = []

            for h in range(num_heads):
                q = tf.matmul(embeddings, attention_weights['protein_q_%d_%d' % (i + 1, h)])
                k = tf.matmul(embeddings, attention_weights['protein_k_%d_%d' % (i + 1, h)])
                v = tf.matmul(embeddings, attention_weights['protein_v_%d_%d' % (i + 1, h)])

                attn_logits = tf.matmul(q, k, transpose_b=True)
                attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(float(head_dim)), axis=-1)
                attn_output = tf.matmul(attn_weights, v)

                attention_heads.append(attn_output)

            # Concatenate all attention heads
            concat_attention = tf.concat(attention_heads, axis=-1)
            return concat_attention


        def self_gating(em,channel):

            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel]))

        compound_embeddings = self_gating(self.compound_embeddings, 1)
        protein_embeddings = self_gating(self.protein_embeddings, 2)

        all_compound_embeddings = [compound_embeddings]

        protein_embeddings = protein_embeddings
        all_protein_embeddings = [protein_embeddings]
        all_hc_embeddings = []
        all_pd_embeddings = []

        dense_H_e = tf.sparse_tensor_to_dense(H_e, validate_indices=False)
        dense_H_n = tf.sparse_tensor_to_dense(H_n, validate_indices=False)
        dense_P_e = tf.sparse_tensor_to_dense(P_e, validate_indices=False)
        dense_P_n = tf.sparse_tensor_to_dense(P_n, validate_indices=False)

        for i in range(self.n_layer):
            new_hc_edge=tf.matmul(dense_H_e,compound_embeddings,a_is_sparse = True)
            new_compound_embeddings = tf.matmul(dense_H_n,new_hc_edge,a_is_sparse = True)
            new_pd_edge=tf.matmul(dense_P_e,protein_embeddings,a_is_sparse = True)
            new_protein_embeddings = tf.matmul(dense_P_n,new_pd_edge,a_is_sparse = True)
            new_compound_embeddings = new_compound_embeddings * pr_compound_embeddings
            new_protein_embeddings = new_protein_embeddings * pr_protein_embeddings
        
            new_compound_embeddings = multi_head_attention_compound(new_compound_embeddings, self.attention_weights, num_heads,
                                                            head_dim)
            new_compound_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_compound_embeddings, self.weights['layer_%d' % (i + 1)]) + compound_embeddings)
        
            # 计算蛋白质节点的新嵌入
            new_protein_embeddings = multi_head_attention_protein(new_protein_embeddings, self.attention_weights, num_heads,
                                                          head_dim)
            new_protein_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_protein_embeddings, self.weights['layer_1_%d' % (i + 1)]) + protein_embeddings)
        
            # 添加节点的注意力机制
        
            attn_weights_compound = tf.nn.softmax(
                tf.matmul(new_compound_embeddings, self.attention_weights['compound']))
            attn_weights_protein = tf.nn.softmax(tf.matmul(new_protein_embeddings, self.attention_weights['protein']))
        
            # 使用注意力权重对邻居节点进行加权求和
            new_compound_embeddings = tf.nn.leaky_relu(tf.matmul(attn_weights_compound * new_compound_embeddings,
                                                self.weights['layer_%d' % (i + 1)]) + compound_embeddings)
            new_protein_embeddings = tf.nn.leaky_relu(tf.matmul(attn_weights_protein * new_protein_embeddings,
                                               self.weights['layer_1_%d' % (i + 1)]) + protein_embeddings)
        
            compound_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_compound_embeddings, self.weights['layer_%d' % (i + 1)]) + compound_embeddings)
            
            protein_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_protein_embeddings, self.weights['layer_1_%d' % (i + 1)]) + protein_embeddings)
        
            compound_embeddings = tf.nn.leaky_relu(new_compound_embeddings)
            protein_embeddings = tf.nn.leaky_relu(new_protein_embeddings)
            # compound_embeddings = tf.nn.leaky_relu(compound_embeddings)
            # protein_embeddings = tf.nn.leaky_relu(protein_embeddings)
        
            compound_embeddings = tf.math.l2_normalize(compound_embeddings,axis=1)
            protein_embeddings = tf.math.l2_normalize(protein_embeddings,axis=1)
            new_hc_edge=tf.math.l2_normalize(new_hc_edge,axis=1)
            new_pd_edge=tf.math.l2_normalize(new_pd_edge,axis=1)
        
        
        
            all_compound_embeddings+=[compound_embeddings]
            all_protein_embeddings+=[protein_embeddings]
            all_hc_embeddings+=[new_hc_edge]
            all_pd_embeddings+=[new_pd_edge]


        compound_embeddings = tf.reduce_sum(all_compound_embeddings,axis=0)
        protein_embeddings = tf.reduce_sum(all_protein_embeddings, axis=0)
        # compound_embeddings = tf.math.l2_normalize(compound_embeddings)
        # protein_embeddings = tf.math.l2_normalize(protein_embeddings)
        compound_embeddings = tf.nn.leaky_relu(compound_embeddings, alpha=0.2)
        protein_embeddings = tf.nn.leaky_relu(protein_embeddings,alpha=0.2)
        # a = tf.reduce_sum(all_compound_embeddings, axis=0)
        # b = tf.reduce_sum(all_protein_embeddings, axis=0)
        hc_edge=tf.reduce_sum(all_hc_embeddings,axis=0)
        pd_edge=tf.reduce_sum(all_pd_embeddings,axis=0)

        # new_hc_dege = tf.nn.leaky_relu(hc_edge, alpha=0.2)
        # new_pd_edge = tf.nn.leaky_relu(pd_edge, alpha=0.2)



        self.neg_idx = tf.placeholder(tf.float32, name="neg_holder")

        self.neg_disease_embedding = tf.convert_to_tensor(self.neg_idx,dtype=tf.float32)
        #self.neg_disease_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(A.toarray(),dtype=tf.float32), self.u_idx)

        self.final_iembedding = protein_embeddings
        self.final_uembedding = compound_embeddings

        self.final_hcedge=hc_edge
        self.final_pdedge=pd_edge

        # self.u_embedding = tf.nn.embedding_lookup(self.final_hcedge, self.u_idx)
        # self.v_embedding = tf.nn.embedding_lookup(self.final_pdedge, self.v_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.final_uembedding, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_iembedding, self.v_idx)

        #self.v_embedding = self.final_pdedge


    def trainModel(self):
        
        sigmoid_output = tf.sigmoid(tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1))
      


        y = tf.reduce_sum(tf.multiply(self.neg_disease_embedding, tf.log(sigmoid_output)) + tf.multiply(
            (1 - self.neg_disease_embedding), tf.log(1 - sigmoid_output)), 0)

        reg_loss = 0
        for key in self.weights:
          
            reg_loss += 0.01 * tf.nn.l2_loss(self.weights[key])
        # reg_loss += self.regU * (tf.nn.l2_loss(self.final_hcedge) + tf.nn.l2_loss(self.final_pdedge))
            reg_loss += self.regU * (tf.nn.l2_loss(self.final_iembedding) + tf.nn.l2_loss(self.final_uembedding))
        loss = -tf.reduce_sum(y) + reg_loss

        optimizer = tf.train.AdamOptimizer(self.lRate)
        train = optimizer.minimize(loss)


        init = tf.global_variables_initializer()
        self.sess.run(init)


        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                herb_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                    feed_dict={self.u_idx: herb_idx, self.neg_idx: j_idx, self.v_idx: i_idx,
                                                self.isTraining: 1})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)
        self.u, self.i, self.weight = self.sess.run([self.final_uembedding, self.final_iembedding, self.weights],
                                                        feed_dict={self.isTraining: 0})
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        np.savetxt('./results/herbedgeDHCN_herb_embedding' + currentTime + '.txt', self.u)
        np.savetxt('./results/diseaseedgeDHCN_disease_embedding' + currentTime + '.txt', self.i)
        saver = tf.train.Saver()
        model_dir = os.path.join("./saved_model", currentTime)
        os.makedirs(model_dir, exist_ok=True)  # 创建目录（如果不存在）

        # 构造完整保存路径：./saved_model/{currentTime}/hdcti_model.ckpt
        model_path = os.path.join(model_dir, "hdcti_model.ckpt")

        # 保存模型
        save_path = saver.save(self.sess, model_path)
        print("模型权重保存成功: %s" % save_path)
    def predictForRanking(self):
        print('hdctipredict----------------------------------------------------------------------------')
        return self.u.dot(self.i.transpose())