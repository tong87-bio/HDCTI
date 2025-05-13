from .diseaseRecommender import diseaseRecommender
from random import shuffle,randint,choice
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.set_random_seed(4321)
class herbRecommender(diseaseRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(herbRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(herbRecommender, self).readConfiguration()
        self.batch_size = int(self.config['batch_size'])

    def printAlgorConfig(self):
        super(herbRecommender, self).printAlgorConfig()

    def initModel(self):
        super(herbRecommender, self).initModel()
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")

        self.r = tf.placeholder(tf.float32, name="rating")
        self.emb_size = int(self.config['num.factors'])
        self.protein_embeddings = tf.Variable(
            tf.truncated_normal(shape=[self.num_proteins, self.emb_size], stddev=0.05),
            name='U'
        )
        self.compound_embeddings = tf.Variable(
            tf.truncated_normal(shape=[self.num_compounds, self.emb_size], stddev=0.05),
            name='V'
        )

        self.batch_compound_emb = tf.nn.embedding_lookup(self.compound_embeddings, self.u_idx)
        self.batcsh_pos_protein_emb = tf.nn.embedding_lookup(self.protein_embeddings, self.v_idx)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                herbs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                diseases = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                # compounds = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                # proteins = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                rating = [self.data.trainingData[idx][2] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                herbs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                diseases = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                # compounds = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                # proteins = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                rating = [self.data.trainingData[idx][2] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            # compound_list = list(self.data.compound.keys())
            # disease_list = list(self.data.protein.keys())
            disease_list = list(self.data.disease.keys())
            #  i_idx.append(self.data.protein[diseases[i]])
            #                 u_idx.append(self.data.compound[herb])
            for i, herb in enumerate(herbs):
                i_idx.append(self.data.protein[diseases[i]])
                # u_idx.append(self.data.compound[herb])
                u_idx.append(self.data.compound[herb])
                # i_idx.append(self.data.disease[diseases[i]])
                # u_idx.append(self.data.herb[herb])
                j_idx.append(rating[i])
            # for i,compound in enumerate(compounds):
            #     i_idx.append(self.data.protein[proteins[i]])
            #     u_idx.append(self.data.compound[compound])
            #     j_idx.append(rating[i])

            yield u_idx, i_idx, j_idx

    def predictForRanking(self,u):
        pass


