"""
example of prediction(clustering of chinese words) with gsdmm model.
"""
from gsdmm import GSDMM_model
from utils.pre_process import split_words_chinese

def model_init():
    model = GSDMM_model()
    model.load(path='./model/')
    return model


def predict(sentence,model,addword=None,top_n_cluster=3,top_n_word=5):
    # sentence is in str
    sentence = [sentence]
    if addword:
        test_sentence = split_words_chinese(dataset=sentence, addwords=addword)
    else:
        test_sentence = split_words_chinese(dataset=sentence)
    p_sentence = test_sentence[0]
    ans_clusters = model.predict(doc=p_sentence, m=top_n_cluster)
    rsl = []
    for c, possi in ans_clusters:
        ans_split = model.get_top_words(cluster=c, n=top_n_word)
        ans = '，'.join([list(x.keys())[0] for x in ans_split])
        rsl.append(ans)
    return rsl



