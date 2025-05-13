# HDCTI

Identifying novel therapeutic targets of natural compounds in traditional Chinese medicine herbs with hypergraph representation learning
## 🚀 Installation

Installation of the project runtime environment，First you can create a virtual environment for the project:
```bash
$ python -m venv [your env name]
```
Activate the virtual environment:
```bash
$ source [your env name]/bin/activate
```
Install project dependencies:
```bash
$ pip install -r requirements.txt
```

## &#x1F3C3; Running
Go to the project directory:
```bash
$ cd Project_Path/
```
Run the main.py file：
```bash
$ python main.py
```
##  🛠️ Configuration
The model can be configured via HGHDA.conf in the src folder
 - datapath: Set the file path of the dataset
 - ratings.setup: Defaults to -columns 0 1 2 (herb,disease,rating)
 - evaluation.setup: Folds for cross validation
 - num.factors: The number of latent factors
 - num.max.epoch: The maximum number of epoch for algorithms.
 - output.setup: The directory path of output results
```
```
## 📈 Inference & Evaluation
To evaluate the model performance using 5-fold cross-validation, you can run the provided inference script: predict.py. The model will automatically:

 - Load each fold’s test set from ./saved_model/{dataset}/test_fold_{i}.txt

 - Load the corresponding trained model weights from ./saved_model/{dataset}/fold{i}/hdcti_model.ckpt

 - Predict compound–target interaction scores

Run the predict.py file：
```bash
$ python predict.py
```
