# _*_ coding:utf-8 _*_
from tqdm import tqdm
from utils import *
import json
import os
import string
import pickle
import lap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from zhon.hanzi import punctuation

# Set hyperparameters and random seed
seed = 12345
np.random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load entity name mappings (Chinese to English)
ent_names = json.load(open("translated_ent_name/dbp_zh_en.json", "r"))
file_path = "KGs/dbp_zh_en/"
d = {}
count = 0
no = 0
depth = 2
entity_num = 15000

# Load pre-trained GloVe word vectors
word_vecs = {}
with open("./glove.6B.300d.txt", encoding='UTF-8') as f:
    for line in tqdm(f.readlines()):
        line = line.split()
        word_vecs[line[0]] = np.array([float(x) for x in line[1:]])

pickle.dump(word_vecs, open("word_vecs.pkl", "wb"))
word_vecs = pickle.load(open("word_vecs.pkl", "rb"))

# Remove punctuation from a given string
def remove_punc(str, punc=None):
    if punc is None:
        punc = PUNC
    if punc == '':
        return str
    return ''.join(['' if i in punc else i for i in str])

# Collect both English and Chinese punctuations
def get_punctuations():
    en = string.punctuation
    zh = punctuation
    puncs = set()
    for i in (zh + en):
        puncs.add(i)
    return puncs

# Load knowledge graph triples and entity alignment pairs
all_triples, node_size, rel_size = load_triples(file_path, True)
train_pair, test_pair = load_aligned_pair(file_path, ratio=0)
PUNC = get_punctuations()

# Compute TF-IKGF vector representations for each entity
corpus = [None for _ in range(len(ent_names))]
for i in ent_names:
    corpus[i[0]] = ' '.join(i[1])
    corpus[i[0]] = remove_punc(corpus[i[0]].lower())

# Build bigram vocabulary and check for missing word vectors
for name in corpus:
    for word in name.split():
        if word not in word_vecs:
            no += 1
        for idx in range(len(word) - 1):
            if word[idx:idx + 2] not in d:
                d[word[idx:idx + 2]] = count
                count += 1

tokenizer = lambda x: x.split()
vectorizer = CountVectorizer(tokenizer=tokenizer)
X = vectorizer.fit_transform(corpus)
word = vectorizer.get_feature_names_out()
transform = TfidfTransformer()
Y = transform.fit_transform(X)

# Extract TF-IKGF scores for a given document index
def get_tfikgf(doc):
    tfikgf = {}
    feature_index = Y[doc, :].nonzero()[1]
    feature_names = vectorizer.get_feature_names_out()
    tfikgf_scores = zip(feature_index, [Y[doc, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfikgf_scores]:
        tfikgf[w] = s
    return tfikgf

# Generate TF-IKGF score list for all entities
tfikgf_list = []
for i, _ in tqdm(ent_names):
    tfikgf = get_tfikgf(i)
    tfikgf_list.append(tfikgf)
pickle.dump(tfikgf_list, open("zhen_nopunc_tfikgf_list.pkl", "wb"))
tfikgf_list = pickle.load(open("zhen_nopunc_tfikgf_list.pkl", "rb"))

# Initialize feature matrices
ent_vec = np.zeros((node_size, 300))
char_vec = np.zeros((node_size, len(d)))

# Construct entity-level embeddings
for i, name in tqdm(enumerate(corpus)):
    k = 0
    tfikgf = tfikgf_list[i]
    ent_vec_list = []
    ent_tfikgf_list = []
    for word in name.split():
        if word in word_vecs:
            ent_vec_list.append(word_vecs[word])
            ent_tfikgf_list.append(tfikgf[word])
            k += 1
        for idx in range(len(word) - 1):
            char_vec[i, d[word[idx:idx + 2]]] += 1
    if k:
        for j in range(len(ent_vec_list)):
            ent_vec[i] += ent_vec_list[j] * ent_tfikgf_list[j]
    else:
        ent_vec[i] = np.random.random(300) - 0.5

    if np.sum(char_vec[i]) == 0:
        char_vec[i] = np.random.random(len(d)) - 0.5
    ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
    char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

# Build sparse adjacency matrix for the knowledge graph
dh = {}
dr = {}
dt = {}
for x, r, y in all_triples:
    dh[x] = dh.get(x, 0) + 1
    dt[y] = dt.get(y, 0) + 1
    dr[r] = dr.get(r, 0) + 1

sparse_rel_matrix = []
for i in range(node_size):
    sparse_rel_matrix.append([i, i, np.log(len(all_triples) / node_size)])
for h, r, t in all_triples:
    weight = np.log(len(all_triples) ** 3 / (dr[r] * dh[h] * dt[t]))
    sparse_rel_matrix.append([h, t, weight])

sparse_rel_matrix = np.array(sorted(sparse_rel_matrix, key=lambda x: x[0]))
sparse_rel_matrix = tf.SparseTensor(
    indices=sparse_rel_matrix[:, :2],
    values=sparse_rel_matrix[:, 2],
    dense_shape=(node_size, node_size)
)

# Concatenate semantic and character-level features
feature = np.concatenate([ent_vec, char_vec], axis=-1)
feature = tf.nn.l2_normalize(feature, axis=-1)

# Define GCN-based similarity computation
def cal_sims(test_pair, feature):
    feature_a = tf.gather(indices=test_pair[:, 0], params=feature)
    feature_b = tf.gather(indices=test_pair[:, 1], params=feature)
    return tf.matmul(feature_a, tf.transpose(feature_b, [1, 0]))

# Compute initial similarity matrix
sims = cal_sims(test_pair, feature)

# Apply multi-hop message passing using sparse matrix
for i in tqdm(range(depth)):
    feature = tf.sparse.sparse_dense_matmul(sparse_rel_matrix, feature)
    feature = tf.nn.l2_normalize(feature, axis=-1)
    sims += cal_sims(test_pair, feature)

sims /= (depth + 1)
sims = sims.numpy()

# Apply LAPJV assignment algorithm
cost, x, y = lap.lapjv(-sims)
print("lapjv:", np.sum(x == y) / entity_num)

# Apply Sinkhorn normalization
sims = tf.exp(sims * 50)
for k in tqdm(range(10)):
    sims = sims / tf.reduce_sum(sims, axis=1, keepdims=True)
    sims = sims / tf.reduce_sum(sims, axis=0, keepdims=True)
sims = np.array(sims)
test(sims, "sinkhorn")
