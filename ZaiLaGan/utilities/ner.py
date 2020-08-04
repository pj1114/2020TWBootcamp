import os
import numpy as np
import jellyfish as jf
import tensorflow as tf
import re
import argparse, time, random
from .NER.model import BiLSTM_CRF
from .NER.utils import str2bool, get_entity
from .NER.data import read_corpus, read_dictionary, tag2label, random_embedding
from .NER.ssc import *

class NER():
    def __init__(self, ner_path, pinyin, stroke, place, person, ssc_dir):
        tf.compat.v1.reset_default_graph()
                
        ## Session configuration
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

        parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
        parser.add_argument('--train_data', type=str, default=ner_path, help='train data source')
        parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
        parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
        parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
        parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
        parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
        args = parser.parse_args(args=[])

        train_path = os.path.join('.', args.train_data, 'train_data')
        train_data = read_corpus(train_path)
        paths = {}
        timestamp = '1594657031'
        output_path = os.path.join('.', args.train_data+"_save", timestamp)
        model_path = os.path.join(output_path, "checkpoints/")
        ckpt_prefix = os.path.join(model_path, "model")
        paths['model_path'] = ckpt_prefix
        word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
        embeddings = random_embedding(word2id, args.embedding_dim)

        #USE MODEL
        self.ckpt_file = tf.train.latest_checkpoint(model_path)
        paths['model_path'] = self.ckpt_file
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths)#, config=self.config)
        model.build_graph()
        self.recognizer = model
        self.sess = tf.compat.v1.Session(config=self.config)
        self.saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.sess, self.ckpt_file)
        self.person_dict = person
        self.place_dict = place
        self.same_pinyin = pinyin
        self.same_stroke = stroke
        self.ssc = ssc(ssc_dir)
    
    def spliteKeyWord(self, str):
        regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
        matches = re.findall(regex, str, re.UNICODE)
        return matches

    def is_good_sentence(self, sentence):
        tmp = self.spliteKeyWord(sentence)
        if len(tmp)==0:
            return False
        cnt=0
        for i in tmp:
            if all(['\u4e00' <= x <= '\u9fff' for x in i]):
              cnt+=1
        return cnt/len(tmp)>=0.7

    def ner_process_document(self, article):
        #函數用來呼叫NER，讓NER能被多次使用
        data = list()
        for sentence in article: 
            if self.is_good_sentence(sentence):
                demo_sent = sentence.strip()
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = self.recognizer.demo_one(self.sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                per = [(i, 'NR', j.start(), j.end()) for i in set(PER)  for j in re.finditer(i, sentence)]
                loc = [(i, 'NT', j.start(), j.end()) for i in set(LOC) for j in re.finditer(i, sentence)]
                org = [(i, 'NS', j.start(), j.end()) for i in set(ORG) for j in re.finditer(i, sentence)]
                per+=loc
                per+=org
                per = sorted(per, key=lambda x:(x[2],x[2]-x[3]))

                tmp_keep = []
                if len(per)!=0:
                    left = per[0][2]
                    right = per[0][3]

                for idx in per:
                    if left==idx[2] and right==idx[3]:
                        tmp_keep.append(idx)
                    elif left<=idx[2] and right>=idx[3]:
                        continue
                    else:
                        tmp_keep.append(idx)
                        left = idx[2]
                        right = idx[3]
                data.append(tmp_keep)
            else:
                data.append([])
        return data

    def throw_NER(self):
        self.sess.close()
        print("NER model closed")

    def get_NER(self, sentence):
        return self.ner_process_document(sentence)
        
    def harmonic_mean(self, a, b):
        return 2*a*b/(a+b) if a+b!=0 else 0

    def get_closest_match(self, x, list_strings, k = 5):
        best_match = None
        highest_jw = 0
        tmp = []
        if x not in list_strings:
            for current_string in list_strings:
                jwscore = jf.jaro_winkler(x, current_string)
                if jwscore < 0.4:
                    continue
                else:
                    len_score = 1 if len(current_string)==len(x) else 0.8
                    current_score = self.harmonic_mean(jwscore, self.ssc.compute_similarity(x, current_string)) * len_score
                if(current_score > highest_jw):
                    highest_jw = current_score
                    best_match = current_string
                    tmp.append((best_match, highest_jw))
        else:
            tmp.append((x,1))
        tmp = sorted(tmp, key=lambda tup: tup[1], reverse=True)
        return tmp[:k]

    def find_similar(self, name, name_dict):
        best_score_name = self.get_closest_match(name, name_dict, 10)
        flag = 0
        for best, score in best_score_name:
            flag=0
            if best:
                position = []
                for idx in range(max(len(best), len(name))):
                    if len(best) > len(name):
                        if idx+1>len(name):
                            position.append(idx)
                        elif name[idx] != best[idx]:
                            position.append(idx)
                    elif len(name) > len(best):
                        if idx+1>len(best):
                            position.append(idx)
                        elif name[idx] != best[idx]:
                            position.append(idx)
                    else:
                        if name[idx] != best[idx]:
                            position.append(idx)
                cnt=0
                for j, idx in enumerate(position):
                    if j==0 and idx+1 == max(len(best), len(name)) and len(best) != len(name) and min(len(best),len(name))>=2:
                        flag=1
                        return best, flag, position
                    if idx+1>len(best) or idx+1>len(name):
                        break
                    try:
                        if best[idx] in self.same_pinyin[name[idx]] or best[idx] in self.same_stroke[name[idx]]:
                            cnt+=1
                        elif idx+1<len(name):
                            if best[idx]==name[idx+1] or best[idx] in self.same_pinyin[name[idx+1]] or best[idx] in self.same_stroke[name[idx+1]]:
                                cnt+=1
                        elif idx-1>=0:
                            if best[idx]==name[idx-1] or best[idx] in self.same_pinyin[name[idx-1]] or best[idx] in self.same_stroke[name[idx-1]]:
                                cnt+=1
                    except:
                        continue

                if cnt==len(position):
                    flag=1
                    return best, flag, position
                    
        return name, flag, position
      
    def check_name(self, sentence):
        # start = time.time()
        answer = self.get_NER(sentence)
        all_truth = []
        for ans in answer:
            truth = []
            for i in ans:
                if i[1] == 'NR':   ## Person
                    if all(['\u4e00' <= j <= '\u9fff' for j in i[0]]):
                        person, change, position = self.find_similar(i[0], self.person_dict)
                        truth.append((person,i[1],i[2],i[3], change, position))
                    else:
                        truth.append((i[0],i[1],i[2],i[3], 0, []))
                elif i[1] == 'NT' or i[1] == 'NS':  ## place
                    if all(['\u4e00' <= j <= '\u9fff' for j in i[0]]):
                        place, change, position = self.find_similar(i[0], self.place_dict)
                        truth.append((place,i[1],i[2],i[3], change, position))
                    else:
                        truth.append((i[0],i[1],i[2],i[3], 0, []))
            all_truth.append(truth)
            del truth
        # end = time.time()
        #print("NER time: %.2f" % (end-start))
        return all_truth
    
    def check_ner(self, sentence, task_name='correction'):
        sentence = [re.sub(r'[\x00-\x20\x7E-\xFF\u3000\xa0\t]', '',i) for i in sentence]
        all_data = []
        all_truth = self.check_name(sentence)
        if task_name=='correction':
            for idx, i in enumerate(all_truth):
                tmp = []
                new_sentence = sentence[idx]
                cumlen = 0
                for j in i:
                    if j[2]+1==j[3]:
                        continue
                    else:
                        indicies = (j[2]+cumlen,j[3]+cumlen)
                        new_sentence = j[0].join([new_sentence[:indicies[0]], new_sentence[indicies[1]:]])
                        if len(j[0])>j[3]-j[2]:
                            new_len = len(j[0])-(j[3]-j[2])
                            tmp.extend(list(range(j[2]+cumlen, j[3]+cumlen+new_len)))
                            cumlen+=new_len
                        else:
                            tmp.extend(list(range(j[2]+cumlen, j[3]+cumlen)))
                all_data.append((new_sentence, tmp))
        elif task_name=='detection':
            for idx, i in enumerate(all_truth):
                tmp_pos=[]
                new_sentence = sentence[idx]
                tmp = []
                for j in i:
                    if j[4]==1:
                        tmp_pos.extend([j[2]+i for i in j[5] if j[2]+i < j[3] ])
                    tmp.extend(list(range(j[2], j[3])))
                all_data.append((new_sentence, tmp, tmp_pos))
        return all_data


