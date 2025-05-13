# coding:utf8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
from util.config import ModelConf
from HDCTI import HDCTI
import numpy as np
from util.io import FileIO
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    conf = ModelConf('./HDCTI.conf')
    if not conf.contains('datapath'):
        raise ValueError("配置文件中缺少 datapath")

    print("📥 加载数据中...")
    data = FileIO.loadDataSet(conf, conf['datapath'])
    print("✅ 数据加载完毕")

    auc_list = []

    for fold in range(5):
        print(f"\n🔁 处理第 {fold} 折...")
        test_path = f'./saved_model/tcmsuite/test_fold_{fold}.txt'
        model_ckpt_path = f'./saved_model/tcmsuite/fold{fold}/hdcti_model.ckpt'

        if not os.path.exists(test_path):
            print(f"❌ 找不到测试文件: {test_path}")
            continue
        if not os.path.exists(model_ckpt_path + ".index"):
            print(f"❌ 找不到模型权重文件: {model_ckpt_path}")
            continue

        # 读取测试集
        test_set = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                herb_name, protein_name, label = parts
                test_set.append((herb_name, protein_name, int(float(label))))
        print(f"✅ 成功加载测试集，共 {len(test_set)} 条")

        # ⚠️ 每一折新建图和会话上下文（不改 HDCTI 源码）
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()
            with sess.as_default():
                model = HDCTI(conf, data, data)  # 不传 session
                model.initModel()

                saver = tf.train.Saver()
                saver.restore(sess, model_ckpt_path)
                print("✅ 成功加载模型参数")

                model.u, model.i = sess.run([model.final_uembedding, model.final_iembedding],
                                            feed_dict={model.isTraining: 0})
                scores = model.predictForRanking()

                y_true = []
                y_score = []
                for herb_name, protein_name, label in test_set:
                    if herb_name not in model.data.compound or protein_name not in model.data.protein:
                        continue
                    herb_idx = model.data.compound[herb_name]
                    protein_idx = model.data.protein[protein_name]
                    y_true.append(int(label))
                    y_score.append(scores[herb_idx, protein_idx])

                if len(set(y_true)) < 2:
                    print("⚠️ 当前折测试集中只有一种标签，无法计算 AUC")
                    auc = 0.0
                else:
                    auc = roc_auc_score(y_true, y_score)

                auc_list.append(auc)
                print(f"🎯 第 {fold} 折 AUC: {auc:.4f}")

            sess.close()  # 手动关闭 session

    # 平均 AUC 输出
    if auc_list:
        mean_auc = sum(auc_list) / len(auc_list)
        print("\n📊 所有折 AUC:")
        for i, auc in enumerate(auc_list):
            print(f"Fold {i}: {auc:.4f}")
        print(f"\n⭐ 平均 AUC: {mean_auc:.4f}")
    else:
        print("❌ 所有折都未成功计算 AUC")
