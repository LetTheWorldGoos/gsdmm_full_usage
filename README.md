## Gsdmm for quick use
A trainable gsdmm unsupervised model with python3. Able to train clustering of topics. Chinese words fitted especially.
Here's the brief description of files:
```
-- gsdmm.py                   # main model, including the inner process and the prediction function of GSDMM.
|
-- predict.py                 # example of prediction(clustering of chinese words) with gsdmm model.
|
-- train.py                   # training sample on server. Need to prepare for the data before training.
|
-- config.ini                 # database & file path config.
|
-- db_utils.py                # tool to connect database, optional.
|
-- utils -- pre_process.py    # prepare for the data before training. Chinese words specified.
|
-- word -- addword.txt        # special words being noticed by the tokenizer. Chinese words specified.
       |-- stopwords.txt      # special words being omitted by the tokenizer. Chinese words specified.
```
If you have any questions during the training process, feel free to contact me with <u>cilia1912pro55@gmail.com<u>.
