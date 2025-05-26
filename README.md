## About:
Code for Neurocomputing 2023 paper: 

Using combinatorial optimization to solve entity alignment: An efficient unsupervised model

## Abstract:
Entity alignment (EA) aims to discover unique equivalent entity pairs with the same meaning across different knowledge graphs (KG), which is a crucial step for expanding the scale of KGs. Existing EA methods commonly leverage graph neural networks (GNNs) to align entities. However, these methods inherit the complex structure of GNNs, which results in lower efficiency. Meanwhile, most EA methods are either limited in their performance due to insufficient utilization of available information or require extensive manual preprocessing to obtain additional information. Furthermore, seed alignment acquisition is challenging for most EA methods that rely on supervised learning. To address these challenges, this paper proposes a simple and effective unsupervised EA model named COEA. COEA leverages the entity name information to obtain reliable supplementary information for EA and enhances performance by combining text features captured by entity names with structural features of the KG. Importantly, COEA inherits the advantages of GNN while reducing redundancy. It only uses the way of aggregating neighbor features in graph convolutional network (GCN), and transforms the EA problems into combination optimization problems. Sufficient experimental of COEA on five datasets have validated the exceptional performance and generalization capabilities of the framework. COEA achieved the best performance in all performance indicators. Notably, the framework enables the rapid implementation of entity alignment with minimal computational delays.

## Bilibili
HIT-MCAD: https://space.bilibili.com/435747358

## Organization
```
./COEA
├── README.md                           # Doc of COEA
├── KGs                                 # Dataset (Taking dbp_zh_en dataset as an example)
├── translated_ent_name                 # Entity names translated by Google translator
├── COEA.py                             # Main codes
├── glove.6B.300d.txt                   # glove word vectors
└── utils.py                            # Sampling methods, ...
```

## To reproduce the results
1) pip install -r requirements.txt
2) python COEA.py

PS: Please download the zip file from http://nlp.stanford.edu/data/glove.6B.zip and choose "glove.6B.300d.txt" as the word vectors. The first time loading the golve word vectors will take some time, when running the program for the second time, you can comment out the code for loading Glove and directly read the word_vecs.pkl.

## Citation:

@article{LIN2023126802,
  title = {Using combinatorial optimization to solve entity alignment: An efficient unsupervised model},
  journal = {Neurocomputing},
  volume = {558},
  pages = {126802},
  year = {2023},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2023.126802},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231223009256},
}

## Acknowledgement
This repo benefits from BootEA, EASY, LightEA, SEU, DATTI, OpenEA and EAKit. Thanks for their wonderful works.

BootEA: https://github.com/nju-websoft/BootEA

EASY: https://github.com/ZJU-DAILY/EASY

LightEA: https://github.com/MaoXinn/LightEA

SEU: https://github.com/MaoXinn/SEU

DATTI: https://github.com/MaoXinn/DATTI

OpenEA: https://github.com/nju-websoft/OpenEA

EAKit: https://github.com/THU-KEG/EAKit

If you want to learn more EA algorithms, please see [Entity_Alignment_Papers](https://github.com/THU-KEG/Entity_Alignment_Papers).

