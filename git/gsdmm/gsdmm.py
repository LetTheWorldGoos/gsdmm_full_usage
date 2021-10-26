# *--utf8--*
"""
main model, including the inner process and the prediction function of GSDMM.
"""
import numpy as np
import math
from tqdm import tqdm
import csv
import gc
import logging

# path = os.path.dirname(os.path.realpath('__file__'))


class GSDMM_model:
    def __init__(self, K=25, n_iters=100, alpha=0.1, beta=0.05):
        self.K = K  # num of cluster, also the maximum num of cluster
        self.k_iter = K  # for count
        self.V = 0  # num of vocabulary in total
        self.D = 0  # num of documents in corpus
        self.n_iters = n_iters  # num of iteration
        self.alpha = alpha
        self.beta = beta

        # slots for computed variables
        # _ means variable that is not saved in memory
        self.z_d = []
        self.m_z = np.zeros(shape=(K,), dtype='int')  # number of documents in each cluster
        self.n_z = np.zeros(shape=(K,), dtype='int')  # number of words in each cluster
        self.n_z_w = [{} for _ in range(K)]  # number of word separately in each cluster
        self.n_transfer = 0

    @staticmethod
    def multi_possibility(p_list):
        """
        :param p_list: distributions like [0.2,0.4,0.1,0.3] for each cluster; total p = 1
        :return: 1 random answer, the id of the cluster
        """
        return [id for id, value in enumerate(np.random.multinomial(1, p_list)) if value][0]

    @staticmethod
    def scaling(value_list):
        """
        linear min-max scaling method(selective, used in exponential scoring part).
        :param value_list: list of values before the possibility is counted
        :return: scaled value list
        """
        return [(x - min(value_list)) / (max(value_list) - min(value_list)) for x in value_list]

    @staticmethod
    def scaling_v2(value_list):
        """
        gauss scaling method(selective, used in exponential scoring part).
        :param value_list: list of values before the possibility is counted
        :return: scaled value list
        """
        return [(x - np.mean(value_list) / math.sqrt(np.var(value_list))) for x in value_list]

    # algorithm realization part

    # initializing
    def init_model(self, documents):
        vocabs = []
        for i, doc in enumerate(documents):
            vocabs += doc

        self.V = len(np.unique(vocabs))
        del vocabs
        gc.collect()
        self.D = len(documents)
        logging.info('vocab num:\t{}, dobument num:\t{}'.format(self.V, self.D))
        self.z_d = np.zeros(shape=(self.D,), dtype='int')

        for i, doc in enumerate(documents):
            zd = self.multi_possibility([1.0 / self.K for _ in range(self.K)])
            self.z_d[i] = zd
            self.m_z[zd] += 1
            self.n_z[zd] += len(doc)
            for word in doc:
                if word not in self.n_z_w[zd].keys():
                    self.n_z_w[zd][word] = 0
                self.n_z_w[zd][word] += 1

        return self

    # normal iteration after init, could be used to update training
    def iter_model(self, documents, debug=False):
        k_count = []
        for n_iter in tqdm(range(self.n_iters)):
            n_transfer = 0
            for i, doc in enumerate(documents):
                doc_len = len(doc)
                z_old = self.z_d[i]
                self.m_z[z_old] -= 1
                self.n_z[z_old] -= doc_len
                for word in doc:
                    self.n_z_w[z_old][word] -= 1
                    if self.n_z_w[z_old][word] == 0:
                        del self.n_z_w[z_old][word]
                try:
                    new_p_list = self.score(doc)
                    z_new = self.multi_possibility(new_p_list)
                except Exception as e:
                    logging.warning('wrong occurred while scoring for doc:\t{}'.format(doc))
                    logging.error(e)
                    z_new = z_old

                self.z_d[i] = z_new
                self.m_z[z_new] += 1
                self.n_z[z_new] += doc_len
                for word in doc:
                    if word not in self.n_z_w[z_new].keys():
                        self.n_z_w[z_new][word] = 0
                    self.n_z_w[z_new][word] += 1
                if z_new != z_old:
                    n_transfer += 1
            # count effective cluster (clusters with docs inside)
            k_new = [x for x in self.m_z if x].__len__()
            k_count.append(k_new)
            if n_transfer == 0 and k_new == self.k_iter and n_iter > self.K / 2:
                # iteration is set for too much.
                logging.warning(
                    'for No.{iter} iteration:\tgot {n_trans} in transfer, and {clus} cluster in final.'.format(
                        iter=n_iter,
                        n_trans=n_transfer,
                        clus=k_new))
                break
            # ATTENTION: k_iter is different from K
            self.k_iter = k_new
        if debug:
            logging.info('K in every iter:\t', k_count)
        return self

    # score a doc to judge its belonging cluster
    def score(self, doc):
        LD1 = math.log((self.D - 1 + self.K * self.alpha))
        p = np.zeros(shape=(self.K,), dtype='float64')
        for cluster in range(self.K):
            LN1 = math.log(self.m_z[cluster] + self.alpha)
            LN2, LD2 = 0, 0
            j = 0
            for word in doc:
                try:
                    LN2 += math.log(self.n_z_w[cluster].get(word, 0) + self.beta + j)
                    j += 1
                except Exception as e:
                    logging.error(e)
            for i in range(len(doc)):
                LD2 += math.log(self.n_z[cluster] + self.V * self.beta + i)

            rsl = LN1 + LN2 - LD1 - LD2
            p[cluster] = rsl
        # selective: scaling

        # p = self.scaling(p)
        # p = self.scaling_v2(p)
        p = [math.exp(x) for x in p]
        return [x / sum(p) for x in p]

    # train as simple
    def train(self, documents):
        self.init_model(documents)
        logging.info('The initialization part of the model is over.')
        self.iter_model(documents)
        logging.info('The iteration part of the model is over.')
        return self

    # model usage part

    # save model to np
    def save(self, path, documents=None, name=''):
        # docs with clusters
        if documents is not None:
            with open(path + name + 'docs_cluster_rsl.csv', 'w+', encoding='utf-8', newline='') as f1:
                # csv写入对象
                csv_writer = csv.writer(f1)
                # 表头
                csv_writer.writerow(['text', 'cluster_result'])
                for doc_id, value in enumerate(self.z_d):
                    a = ''.join(documents[doc_id])
                    b = value.__str__()
                    csv_writer.writerow([a, b])
                f1.close()
        # z_d vector
        np.save(path + name + 'z_d_vector.npy', self.z_d)
        # n_z vector
        np.save(path + name + 'n_z_vector.npy', self.n_z)
        # m_z vector
        np.save(path + name + 'm_z_vector.npy', self.m_z)
        # n_z_w vector
        np.save(path + name + 'n_z_w_vector.npy', self.n_z_w)

        # variables
        factors = {'alpha': self.alpha, 'beta': self.beta, 'K': self.K, 'V': self.V, 'D': self.D}
        np.save(path + name + 'dmm_factors.npy', factors)

    # load the model from np
    def load(self, path, name=''):
        # variables
        factors = np.load(path + name + 'dmm_factors.npy', allow_pickle=True)
        factors = factors.item()
        self.alpha, self.beta, self.K, self.V, self.D = \
            factors['alpha'], factors['beta'], factors['K'], factors['V'], factors['D']
        # z_d vector
        self.z_d = np.load(path + name + 'z_d_vector.npy', allow_pickle=True)
        # n_z vector
        self.n_z = np.load(path + name + 'n_z_vector.npy', allow_pickle=True)
        # m_z vector
        self.m_z = np.load(path + name + 'm_z_vector.npy', allow_pickle=True)
        # n_z_w vector
        self.n_z_w = np.load(path + name + 'n_z_w_vector.npy', allow_pickle=True)

    # get top words of a certain cluster/ each clusters
    def get_top_words(self, cluster, n=20):
        """
        获取每个cluster的self.n_z_w 然后range一下取max top
        :param cluster: int
        :param n: int, max num of words
        :return: words list with frequency
        """
        wordlist = self.n_z_w[cluster]  # big dict
        opt = []
        try:
            if isinstance(wordlist, dict):
                if wordlist.keys().__len__():
                    wordlist = sorted(wordlist.items(), key=lambda x: x[1], reverse=True)  # 按value倒序排序
                    opt = [{x[0]: x[1]} for x in wordlist[:n]]
        except Exception as e:
            logging.error(e)
        finally:
            return opt

    # get number of docs in such cluster
    def get_num_docs(self, cluster):
        return self.m_z[cluster]

    # get cluster of a doc
    # score of input -> m cluster with highest possibility
    def predict(self, doc, m=3):
        # a = time.time()

        p_doc = self.score(doc)
        p_doc = [(clus_id, value) for clus_id, value in enumerate(p_doc)]
        # print('predict time cost:\t{}'.format(time.time() - a))
        return sorted(p_doc, key=lambda x: x[1], reverse=True)[:m]

    # backup use frequency only for the prediction
    def predict_freq(self, doc, m=3):
        freqs = []
        for cluster in range(self.K):
            freq = 0
            for word in doc:
                freq += self.n_z_w[cluster].get(word, 0) / self.n_z[cluster]
            freqs.append(freq)

        f_doc = [(clus_id, value) for clus_id, value in enumerate(freqs)]
        return sorted(f_doc, key=lambda x: x[1], reverse=True)[:m]
