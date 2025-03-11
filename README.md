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
Run the main.py file in the src folder：
```bash
$ python main.py
```
##  🛠️ Configuration
The model can be configured via HGHDA.conf in the src folder
 - datapath:Set the file path of the dataset
 - ratings.setup:Defaults to -columns 0 1 2 (herb,disease,rating)
 - evaluation.setup:Folds for cross validation
 - num.factors:the number of latent factors
 - num.max.epoch:the maximum number of epoch for algorithms.
 - output.setup:the directory path of output results
