import numpy as np
import os
import tensorflow as tf


def load_triples(file_path, reverse=True):
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i, 0] = triples[i, 2]
            reversed_triples[i, 2] = triples[i, 0]
            if reverse:
                reversed_triples[i, 1] = triples[i, 1] + rel_size
            else:
                reversed_triples[i, 1] = triples[i, 1]
        return reversed_triples

    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()

    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()

    triples = np.array([line.replace("\n", "").split("\t") for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:, 0]), np.max(triples[:, 2])]) + 1
    rel_size = np.max(triples[:, 1]) + 1
    # aaa = reverse_triples(triples)
    all_triples = np.concatenate([triples, reverse_triples(triples)], axis=0)  # 如果翻转头尾实体，则关系接着之前的最大的关系编号排序，因为头尾实体换了，关系都是新的了
    # all_triples = triples
    all_triples = np.unique(all_triples, axis=0)  # 去重

    return all_triples, node_size, rel_size * 2 if reverse else rel_size  # 如果翻转头尾实体，则关系数量翻倍


def load_aligned_pair(file_path, ratio=0.3):
    if "sup_ent_ids" not in os.listdir(file_path):
        with open(file_path + "ref_ent_ids") as f:
            aligned = f.readlines()
    else:
        with open(file_path + "ref_ent_ids") as f:
            ref = f.readlines()
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
        aligned = ref + sup

    aligned = np.array([line.replace("\n", "").split("\t") for line in aligned]).astype(np.int64)
    np.random.shuffle(aligned)
    return aligned[:int(len(aligned) * ratio)], aligned[int(len(aligned) * ratio):]


def test(sims, mode="sinkhorn", batch_size=1024):
    if mode == "sinkhorn":
        result = []
        for epoch in range(len(sims) // batch_size + 1):
            sim = sims[epoch * batch_size:(epoch + 1) * batch_size]
            rank = tf.argsort(-sim, axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch + 1) * batch_size, len(sims)))])
            cast = tf.cast(rank, ans_rank.dtype)
            expand = np.expand_dims(ans_rank, axis=1)
            tile = tf.tile(expand, [1, len(sims)])
            equal = tf.equal(cast, tile)  #
            result.append(tf.where(equal).numpy())
        results = np.concatenate(result, axis=0)

        def cal(results):
            hits1, hits10, mrr = 0, 0, 0
            for x in results[:, 1]:
                if x < 1:
                    hits1 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1 / (x + 1)
            return hits1, hits10, mrr

        hits1, hits10, mrr = cal(results)
        print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1 / len(sims) * 100, hits10 / len(sims) * 100, mrr / len(sims) * 100))
    else:
        c = 0
        for i, j in enumerate(sims[1]):
            if i == j:
                c += 1
        print("hits@1 : %.2f%%" % (100 * c / len(sims[0])))
