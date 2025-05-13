from HDR import HDR
from util.config import ModelConf
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# tf.config.set_visible_devices([], 'GPU')


if __name__ == '__main__':


    import time
    s = time.time()
    #Register your model here and add the conf file yuinto the config directory

    try:
        conf = ModelConf('HDCTI.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    recSys = HDR(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
