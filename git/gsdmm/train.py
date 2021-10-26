'''
training on internal server.
should include the expire-delete rule;
*should combined with internal connection DB.
*if file input needed, plz change the format on your own or connect me.
'''
from gsdmm import GSDMM_model
import os
import logging
from configparser import ConfigParser
from argparse import ArgumentParser
import gc
from Conn import conn
from utils.pre_process import *
import time


# configs:
#       addword / stopword path
#       data_table / time range(days)
#       model save / load path
#       backup path / expire time range

# args:
#       factors: K=K, n_iters=n_iters, alpha=0.1, beta=0.1
#       config path

def get_config(path):
    c = ConfigParser()
    c.read(path, encoding="utf-8")
    return c


def read_words(path):
    with open(path, 'r+', encoding='utf-8') as f:
        words = f.readlines()
        f.close()
    return [x.replace('\n', '').replace('\t', '').strip() for x in words]


def read_data(conn, table, date_range):
    date = time.localtime(time.time() - 86400 * date_range)
    date = [x.__str__() for x in date]
    date_s = '-'.join(date[:3]) + ' ' + ':'.join(date[3:6])
    sel_sql = 'select wtms from {table} where fxsj >= {date}'.format(table=table, date=date_s)
    data = pd.read_sql(sel_sql, conn)
    return data


def preprocessing(raw_corpus, model_path, addword, stopwords):
    dataset = split_words_chinese(dataset=raw_corpus, addwords=addword, cut_type='HMM')
    logging.info('words split for whole corpus.')
    dataset = delete_stopwords(dataset=dataset, stopwords=stopwords)
    logging.info('stopwords deleted.')
    dataset = chinese_only(dataset)
    logging.info('only chinese words left.')
    dataset = del_low_frequence(dataset=dataset, min_freq=2)
    logging.info('only high freq words left.')
    dataset = del_one_char(dataset)
    logging.info('one char words are deleted.')
    df = pd.DataFrame(columns=['corpus', 'dataset'])
    df['corpus'] = raw_corpus
    df['dataset'] = dataset
    logging.info('length of data before filtering:\t{}'.format(len(df)))
    df['dataset'] = df['dataset'].apply(lambda x: 'Drop' if isinstance(x, list) and len(x) == 0 else x)
    df = df[df['dataset'] != 'Drop']
    logging.info('length of data after filtering:\t{}'.format(len(df)))
    raw_corpus = df['corpus'].values
    dataset = df['dataset'].values
    np.save(model_path + '/dataset_raw.npy', raw_corpus)
    np.save(model_path + '/dataset.npy', dataset)
    logging.info('corpus saved'.format(len(df)))
    return raw_corpus, dataset


if __name__ == '__main__':
    start = time.time()
    # set log
    log_path = os.path.dirname(os.path.abspath(__file__)) + '/log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + '/gsdmm.log'
    logging.basicConfig(filename=log_name, filemode='a', level=logging.INFO)
    logging.info('GSDMM training task start at: %s' % (time.ctime()))

    # set argparser and config
    config_path = os.path.dirname(os.path.abspath(__file__)) + '/config.ini'
    parser = ArgumentParser(description='Train the GSDMM model for caller-info.')
    # parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--alpha', type=int, default=0.1)
    parser.add_argument('--beta', type=int, default=0.1)
    args = parser.parse_args()

    config = get_config(config_path)
    addword_p = config.get('word', 'addword_path')
    stopword_p = config.get('word', 'stopword_path')

    addword = read_words(addword_p)
    stopwords = read_words(stopword_p)
    k, n_iters, alpha, beta = args.k, args.iters, args.alpha, args.beta
    model_path = config.get('model', 'save_path')

    backup_path = config.get('model', 'backup_path')
    backup_expire_range = config.getint('model', 'backup_expire_range')
    conn = conn()
    datatable = config.get('data', 'data_table')
    time_len = config.getint('data', 'time_len')

    logging.info('factor initialized.')

    # load data from db
    dataraw = read_data(conn=conn, table=datatable, date_range=time_len)
    raw_corpus = dataraw['wtms']
    logging.info('data loaded.')
    del dataraw
    gc.collect()

    # model preprocessing
    raw_corpus, dataset = preprocessing(raw_corpus, model_path, addword, stopwords)

    # model training
    model = GSDMM_model(K=k, n_iters=n_iters, alpha=alpha, beta=beta)
    model.train(documents=dataset)

    logging.info('GSDMM training task is over.\nTime cost {}:\tfor {} in data, {} in cluster, and {} in iteration'.format(time.time() - start,
                                                                                     len(dataset),
                                                                                     k, n_iters))
    # model save
    model.save(documents=raw_corpus, path=model_path)
    logging.info('model saved.')

    # save backup and clean expired model files
    datenum = ''.join([str(x) if len(str(x))>1 else '0' + str(x) for x in time.localtime()[:3]])
    model.save(documents=raw_corpus, path=backup_path, name=datenum)
    logging.info('backup model saved.')
    earliest_timestamp = time.time() - 86400 * backup_expire_range
    del_models = []
    for info in os.listdir(backup_path):
        file_at_time = os.path.getmtime(backup_path + info)
        if file_at_time < earliest_timestamp:
            del_models.append(info)
    if del_models.__len__():
        for m in del_models:
            os.remove(backup_path + m)
        logging.info('expired model deleted.')

    logging.info('GSDMM training task end at: %s' % (time.ctime()))