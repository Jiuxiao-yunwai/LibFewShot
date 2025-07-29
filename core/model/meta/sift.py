# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from PIL import Image
from ..metric.metric_model import MetricModel
from ...utils import accuracy
from ...data.dataset import GeneralDataset


# #############################################################################
# #                           辅助模块 (Helper Modules)                         #
# #############################################################################

class Encoder(nn.Module):
    """
    编码器: 将高维的视觉特征映射到低维的语义空间。
    """
    def __init__(self, z_dim, z_sem, bias=True):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(z_dim, z_sem, bias=bias)

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    """
    解码器: 将低维的语义特征映射回高维的视觉特征空间。
    """
    def __init__(self, z_dim, z_sem, bias=True):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_sem, z_dim, bias=bias)

    def forward(self, x):
        return self.fc(x)


class Transformation(nn.Module):
    """
    转换网络: 在语义空间中，学习一个从基类语义到新类语义的线性变换。
    """
    def __init__(self, z_sem, bias=True):
        super(Transformation, self).__init__()
        self.trans = nn.Linear(z_sem, z_sem, bias=bias)

    def forward(self, x):
        return self.trans(x)


# #############################################################################
# #                           SIFT 核心逻辑层 (SIFTLayer)                      #
# #############################################################################

class SIFTLayer(nn.Module):
    """
    封装了 SIFT 模型针对单个少样本任务（episode）的全部核心计算逻辑。
    包括基类选择、生成器在线训练、特征生成、原型修正和最终分类。
    """
    def __init__(self, feat_dim, way_num, shot_num,
                 emb_base, emb_novel, feat_base, label_base,
                 setting, n_generated, st_iter, lr_st, dc_st,
                 compactness_w, final_classifier, classifier_max_iter, device):
        super(SIFTLayer, self).__init__()

        # 基础参数 (由框架传入)
        self.feat_dim = feat_dim
        self.way_num = way_num
        self.shot_num = shot_num
        self.device = device

        # 预加载的数据 (由 SIFT 主类传入)
        self.emb_base = emb_base           # (num_base_classes, z_sem)
        self.emb_novel = emb_novel         # (num_novel_classes, z_sem)
        self.z_sem = emb_base.shape[1]
        self.feat_base = feat_base         # (num_base_samples, feat_dim)
        self.label_base = label_base       # (num_base_samples,)
        
        self.setting = setting

        # SIFT 专属超参数
        self.n_generated = n_generated
        self.st_iter = st_iter
        self.lr_st = lr_st
        self.dc_st = dc_st
        self.compactness_w = compactness_w
        self.final_classifier = final_classifier
        self.classifier_max_iter = classifier_max_iter

        # 初始化模型组件 (编码器、解码器、转换网络)
        self.encoder = Encoder(self.feat_dim, self.z_sem).to(device)
        self.decoder = Decoder(self.feat_dim, self.z_sem).to(device)
        self.trans = Transformation(self.z_sem).to(device)

        # 损失函数
        self.loss_func = nn.CrossEntropyLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, support_feat, query_feat, support_target):
        """
        处理单个少样本任务的前向传播。
        """
        # --- 步骤 1: 语义基类选择 ---
        # 使用匈牙利算法（通过线性规划求解）为每个新类匹配一个语义最相关的基类
        unique_novel_targets_indices = torch.unique(support_target).sort()[0]
        emb_novel_task = self.emb_novel[unique_novel_targets_indices]
        cost_matrix = torch.cdist(emb_novel_task, self.emb_base).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # row_ind: novel_idx, col_ind: base_idx
        
        # --- 步骤 2: 在线训练生成器 ---
        # 针对当前任务动态训练 Encoder-Transform-Decoder 流水线
        self._train_st_and_generator(support_feat, support_target, query_feat, col_ind, row_ind)
        
        # --- 步骤 3: 生成特征 ---
        # 使用训练好的生成器为新类生成新的、有意义的视觉特征
        feat_generated, gen_targets = self._generate_features(col_ind, row_ind)
        
        # --- 步骤 4: 增强支持集 ---
        # 将生成的特征与原始支持集特征合并，形成一个扩充后的支持集
        if feat_generated is not None:
            s_feat_aug = torch.cat([support_feat, feat_generated], dim=0)
            s_target_aug = torch.cat([support_target, gen_targets], dim=0)
        else:
            s_feat_aug = support_feat
            s_target_aug = support_target

        # --- 步骤 5: 最终分类 ---
        # 使用增强后的支持集训练最终分类器，并对查询集进行预测
        output = self._final_classification(s_feat_aug, s_target_aug, query_feat)
        return output

    def _train_st_and_generator(self, support_feat, support_target, query_feat, base_indices, novel_indices):
        """
        在线训练 Encoder-Transform-Decoder 生成器。
        """
        with torch.set_grad_enabled(True):
            # 关键修复：克隆输入张量并手动开启梯度，为内部训练建立一个局部的计算图
            support_feat_clone = support_feat.clone().detach().requires_grad_(True)

            params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.trans.parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr_st, weight_decay=self.dc_st)

            for _ in range(self.st_iter):
                optimizer.zero_grad()

                # 从所有基类特征中，为每个匹配的基类随机采样 'shot_num' 个样本
                feat_b_list = [self.feat_base[self.label_base == idx][torch.randperm(torch.sum(self.label_base == idx))[:self.shot_num]] for idx in base_indices]
                feat_b = torch.cat(feat_b_list, dim=0)

                # 获取匹配的基类和新类的语义嵌入
                emb_b = self.emb_base[base_indices]
                emb_n = self.emb_novel[novel_indices]
                
                # 编码器损失: 包含映射损失(视觉->语义)和重构损失(语义->视觉)
                feat_b_sem = self.encoder(feat_b)
                loss_en_map = self.l2_loss(feat_b_sem, emb_b.repeat_interleave(self.shot_num, dim=0))
                loss_en_rec = self.l2_loss(self.decoder(feat_b_sem), feat_b)
                
                # 转换损失: 学习从基类语义到新类语义的转换
                feat_n_sem_hallucinated = self.trans(feat_b_sem - emb_b.repeat_interleave(self.shot_num, dim=0)) + emb_n.repeat_interleave(self.shot_num, dim=0)
                # loss_trans = self.l2_loss(feat_n_sem_hallucinated, self.encoder(support_feat))
                loss_trans = self.l2_loss(feat_n_sem_hallucinated, self.encoder(support_feat_clone))
                
                # 解码器损失: 同样包含映射损失和重构损失
                feat_n_hallucinated = self.decoder(feat_n_sem_hallucinated)
                # loss_de_map = self.l2_loss(self.encoder(feat_n_hallucinated), self.encoder(support_feat))
                # loss_de_rec = self.l2_loss(feat_n_hallucinated, support_feat)
                loss_de_map = self.l2_loss(self.encoder(feat_n_hallucinated), self.encoder(support_feat_clone))
                loss_de_rec = self.l2_loss(feat_n_hallucinated, support_feat_clone)

                # 原型修正: 使用最短路径算法修正原型
                proto_s = self._get_prototype(support_feat, support_target)
                # proto_rectified = self._prototype_rectification(query_feat, proto_s)
                if self.setting == 'tr':  # Transductive setting: 使用查询集信息修正原型
                    proto_rectified = self._prototype_rectification(query_feat, proto_s)
                else:  # Inductive setting: 不使用查询集，直接使用支持集原型
                    proto_rectified = proto_s
                
                # 紧凑性损失: 促使生成的特征紧凑地分布在修正后的原型周围
                loss_compact = self._compactness_loss(feat_n_hallucinated, proto_rectified, support_target)

                # 汇总总损失并反向传播
                total_loss = loss_en_map + loss_en_rec + loss_trans + loss_de_map + loss_de_rec + self.compactness_w * loss_compact
                total_loss.backward(retain_graph=True)  # retain_graph=True 以便在循环中继续计算梯度
                optimizer.step()

    def _generate_features(self, base_indices, novel_indices):
        """
        使用训练好的生成器为新类生成特征。
        """
        n_samples_per_base = self.n_generated // len(base_indices)
        if n_samples_per_base == 0:
            return None, None

        feat_generated_list, target_generated_list = [], []
        emb_b, emb_n = self.emb_base[base_indices], self.emb_novel[novel_indices]

        for i, base_cls_idx in enumerate(base_indices):
            # 从基类中随机采样特征作为生成的“源”
            base_cls_feats = self.feat_base[self.label_base == base_cls_idx]
            feat_b_sample = base_cls_feats[torch.randperm(base_cls_feats.size(0))[:n_samples_per_base]]

            with torch.no_grad():  # 生成过程不计算梯度
                feat_b_sem = self.encoder(feat_b_sample)
                feat_n_sem = self.trans(feat_b_sem - emb_b[i]) + emb_n[i]
                feat_n = self.decoder(feat_n_sem)
            
            feat_generated_list.append(feat_n)
            target_generated_list.append(torch.full((n_samples_per_base,), novel_indices[i], dtype=torch.long, device=self.device))
        
        return torch.cat(feat_generated_list, dim=0), torch.cat(target_generated_list, dim=0)

    def _final_classification(self, support_feat, support_target, query_feat):
        """
        根据配置文件选择并训练最终的分类器。
        """
        if self.final_classifier == 'LR':
            classifier = LogisticRegression(max_iter=self.classifier_max_iter, random_state=0, solver='lbfgs')
            classifier.fit(support_feat.detach().cpu().numpy(), support_target.cpu().numpy())
            # return torch.from_numpy(classifier.predict_proba(query_feat.detach().cpu().numpy())).float().to(self.device)
            probs = classifier.predict_proba(query_feat.detach().cpu().numpy())
            return torch.from_numpy(probs).float().to(self.device).requires_grad_()

        elif self.final_classifier == 'SVM':
            classifier = SVC(probability=True, gamma='auto', random_state=0, max_iter=self.classifier_max_iter)
            classifier.fit(support_feat.detach().cpu().numpy(), support_target.cpu().numpy())
            # return torch.from_numpy(classifier.predict_proba(query_feat.detach().cpu().numpy())).float().to(self.device)
            probs = classifier.predict_proba(query_feat.detach().cpu().numpy())
            return torch.from_numpy(probs).float().to(self.device).requires_grad_()
            
        else:  # 'GD' - 使用梯度下降优化的线性分类器
            classifier = nn.Linear(self.feat_dim, self.way_num).to(self.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=5e-4)
            for _ in range(self.classifier_max_iter):
                out_cls = classifier(support_feat)
                loss_cls = self.loss_func(out_cls, support_target)
                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()
            return classifier(query_feat)

    def _get_prototype(self, feat, target):
        """计算每个类别的原型（均值特征）。"""
        return torch.stack([feat[target == i].mean(0) for i in torch.unique(target).sort()[0]])

    def _prototype_rectification(self, query_feat, support_proto):
        """
        使用最短路径算法进行原型修正，完整复现论文逻辑。
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.way_num, random_state=0, n_init='auto').fit(query_feat.detach().cpu().numpy())
        query_proto = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        
        num_proto = support_proto.shape[0]
        cost_matrix = torch.cdist(support_proto, query_proto).cpu().numpy()

        # 构建支持集原型和查询集原型之间的二分图
        graph = np.full((num_proto * 2, num_proto * 2), np.inf)
        graph[:num_proto, num_proto:] = cost_matrix
        graph[num_proto:, :num_proto] = cost_matrix

        # 计算所有节点对之间的最短路径
        dist_matrix, predecessors = shortest_path(csgraph=csr_matrix(graph), directed=False, return_predecessors=True)
        
        rectified_proto_list = []
        for i in range(num_proto):
            min_dist, min_path_nodes = np.inf, []
            
            # 找到连接支持原型i和所有查询原型的最短路径
            for j in range(num_proto):
                if dist_matrix[i, num_proto + j] < min_dist:
                    min_dist = dist_matrix[i, num_proto + j]
                    
                    # 回溯前驱节点，构建完整路径
                    path, curr = [num_proto + j], num_proto + j
                    while predecessors[i, curr] != -9999:
                        path.append(predecessors[i, curr])
                        curr = predecessors[i, curr]
                    path.reverse()
                    min_path_nodes = path
            
            # 收集路径上所有节点对应的特征向量进行融合
            fusion_list = [support_proto[node] if node < num_proto else query_proto[node - num_proto] for node in min_path_nodes]
            rectified_proto_list.append(torch.stack(fusion_list).mean(dim=0))
            
        return torch.stack(rectified_proto_list)
        
    def _compactness_loss(self, features, prototypes, targets):
        """计算紧凑性损失。"""
        loss = 0.0
        unique_targets = torch.unique(targets)
        for i in unique_targets:
            class_features = features[targets == i]
            if class_features.shape[0] > 0:
                loss += ((class_features - prototypes[i])**2).mean()
        # 对所有类别的损失求平均
        return loss / len(unique_targets) if len(unique_targets) > 0 else torch.tensor(0.0, device=features.device)


# #############################################################################
# #                           SIFT 主模型 (框架接口)                             #
# #############################################################################

class SIFT(MetricModel):
    """
    SIFT 模型主类。
    负责与 LibFewShot 框架交互，处理数据加载、模型初始化和任务分发。
    """
    def __init__(self, **kwargs):
        super(SIFT, self).__init__(**kwargs)
        
        # 从 kwargs 中获取所有需要的配置参数
        self.data_root = kwargs.get('data_root')

        # # 新增：定义所有缓存文件的路径
        # self.pre_extract_dir = os.path.dirname(kwargs.get('pre_extract_path', './results/pre_extract/'))
        # self.wordnet_map_path = os.path.join(self.pre_extract_dir, "miniImageNet_word_map.json")
        # self.wordnet_words_path = os.path.join(self.pre_extract_dir, "words.txt")

        # # 新增：自动进行 WordNet ID 到单词的映射
        # self.word_map = self._create_word_mapping()

        self.pre_extract_path = kwargs.get('pre_extract_path')
        
        # 自动进行基类特征提取 (如果需要)
        self.feat_base, self.label_base = self._pre_extract_base_features()
        # 使用您提供的 JSON 文件加载语义嵌入
        self.emb_base, self.emb_novel, _ = self._load_glove_embeddings(
            kwargs.get('glove_path'), 
        )
        
        # 加载语义嵌入 (GloVe)
        # self.emb_base, self.emb_novel, _ = self._load_glove_embeddings(kwargs.get('glove_path'))

        # 将 SIFT 专属的超参数和 device 传入 SIFTLayer
        # kwargs['device'] = self.device
        # self.sift_layer = SIFTLayer(
        #     feat_dim=self.feat_dim, 
        #     way_num=self.way_num, 
        #     shot_num=self.shot_num,
        #     emb_base=self.emb_base,
        #     emb_novel=self.emb_novel,
        #     feat_base=self.feat_base,
        #     label_base=self.label_base,
        #     **kwargs  # 将所有从配置文件传入的 kwargs 传递给 SIFTLayer
        # )
        # 显式地将所有需要的参数传递给 SIFTLayer，避免重复
        self.sift_layer = SIFTLayer(
            feat_dim=self.feat_dim,
            way_num=self.way_num,
            shot_num=self.shot_num,
            emb_base=self.emb_base,
            emb_novel=self.emb_novel,
            feat_base=self.feat_base,
            label_base=self.label_base,
            device=self.device,
            # 从 kwargs 中安全地获取 SIFT 专属的超参数
            n_generated=kwargs.get('n_generated'),
            st_iter=kwargs.get('st_iter'),
            lr_st=kwargs.get('lr_st'),
            dc_st=kwargs.get('dc_st'),
            compactness_w=kwargs.get('compactness_w'),
            setting=kwargs.get('setting', 'tr'),
            final_classifier=kwargs.get('final_classifier'),
            classifier_max_iter=kwargs.get('classifier_max_iter')
        )

        self.loss_func = nn.CrossEntropyLoss()

    def _pre_extract_base_features(self):
        """
        如果缓存文件存在，则加载；否则，自动提取、保存并加载所有基类（训练集）的特征。
        这个函数只会在第一次运行时执行耗时操作。
        """
        if self.pre_extract_path is None:
            raise ValueError("`pre_extract_path` 必须在配置文件中提供。")

        if os.path.exists(self.pre_extract_path):
            print(f"检测到已提取的特征文件，正在从 {self.pre_extract_path} 加载...")
            data = np.load(self.pre_extract_path, allow_pickle=True).item()
            features = torch.from_numpy(data['features']).float().to(self.device)
            labels = torch.from_numpy(data['labels']).long().to(self.device)
            return features, labels
        
        # --- 如果缓存文件不存在，则执行一次性提取 ---
        print("未找到基类特征文件，现在开始一次性提取...")
        if self.data_root is None:
            raise ValueError("`data_root` 未在配置文件中找到，无法提取特征。")
        
        # 定义一个简单的 transform 用于特征提取
        transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 定义一个简单的 collate_fn 来应用 transform
        # GeneralDataset 返回的是图片路径，我们需要在这里加载并转换
        def collate_fn(batch):
            images, labels = zip(*batch)
            images = [transform(img) for img in images]
            return torch.stack(images), torch.tensor(labels)

        train_dataset = GeneralDataset(data_root=self.data_root, mode='train', use_memory=False)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate_fn)
        
        all_features, all_labels = [], []
        self.emb_func.to(self.device)
        self.emb_func.eval()  # 确保骨干网络处于评估模式

        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                feats = self.emb_func(images)
                all_features.append(feats.cpu())
                all_labels.append(labels)
                if (i + 1) % 20 == 0:
                    print(f"正在处理批次 {i + 1}/{len(train_loader)}...")
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 确保目录存在并保存文件
        os.makedirs(os.path.dirname(self.pre_extract_path), exist_ok=True)
        np.save(self.pre_extract_path, {'features': all_features.numpy(), 'labels': all_labels.numpy()})
        print(f"\n基类特征已提取并保存到: {self.pre_extract_path}")

        return all_features.to(self.device), all_labels.to(self.device)

    def _load_glove_embeddings(self, glove_path):
        """加载 GloVe 词向量。"""
        # print(f"正在从 {glove_path} 加载 GloVe 词向量...")
        # w2v = {}
        # with open(glove_path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         parts = line.split()
        #         if len(parts) > 2:
        #             w2v[parts[0]] = np.array([float(val) for val in parts[1:]])
        
        # # 移除可能为空的条目
        # w2v = {k: v for k, v in w2v.items() if v is not None}
        # z_sem = next(iter(w2v.values())).shape[0]
        
        # # 为 miniImageNet 的类别划分（64 base, 16 val, 20 test）准备足够的词向量
        # num_base, num_novel = 64, 36
        # np.random.seed(0)  # 固定随机种子以保证实验可复现
        # selected = np.random.choice(list(w2v.keys()), num_base + num_novel, replace=False)
        
        # emb_base = torch.from_numpy(np.stack([w2v[w] for w in selected[:num_base]])).float()
        # emb_novel = torch.from_numpy(np.stack([w2v[w] for w in selected[num_base:]])).float()
        
        # return emb_base.to(self.device), emb_novel.to(self.device), z_sem
        wordnet_map_path = "/root/autodl-tmp/dataset01/imagenet_class_index.json"
        if not os.path.exists(wordnet_map_path):
            raise FileNotFoundError(f"映射文件未找到: {wordnet_map_path}")

        print(f"正在从 {wordnet_map_path} 加载单词映射...")
        with open(wordnet_map_path, 'r') as f:
            # 解析 JSON，创建一个 {wordnet_id: word} 的字典
            json_data = json.load(f)
            self.word_map = {v[0]: v[1] for k, v in json_data.items()}

        print(f"正在从 {glove_path} 加载 GloVe 词向量...")
        w2v = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if len(parts) > 2:
                    w2v[parts[0]] = np.array([float(val) for val in parts[1:]])
        
        w2v = {k: v for k, v in w2v.items() if v is not None}
        z_sem = next(iter(w2v.values())).shape[0]

        # 获取数据集中所有的基类和新类 ID
        base_ids, novel_ids = set(), set()
        # 一次性读取所有 split 文件，确保 ID 列表是完整的
        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(self.data_root, f"{split}.csv")
            with open(csv_path, 'r') as f:
                next(f)  # 跳过表头
                for line in f:
                    class_id = line.strip().split(',')[1]
                    if split == 'train':
                        base_ids.add(class_id)
                    else:
                        novel_ids.add(class_id)
        
        base_ids, novel_ids = sorted(list(base_ids)), sorted(list(novel_ids))
        

        # 定义一个辅助函数来安全地获取词向量
        def get_embedding(word_id):
            phrase = self.word_map.get(word_id, '').replace('_', ' ').split()
            vectors = [w2v[word] for word in phrase if word in w2v]
            return np.mean(vectors, axis=0) if vectors else np.zeros(z_sem)

        emb_base = torch.from_numpy(np.stack([get_embedding(wid) for wid in base_ids])).float()
        emb_novel = torch.from_numpy(np.stack([get_embedding(wid) for wid in novel_ids])).float()
        
        return emb_base.to(self.device), emb_novel.to(self.device), z_sem

    def set_forward(self, batch):
        """框架调用的测试/验证前向传播接口。"""
        return self._forward_main(batch, is_train=False)

    def set_forward_loss(self, batch):
        """框架调用的训练前向传播接口。"""
        return self._forward_main(batch, is_train=True)

    def _forward_main(self, batch, is_train):
        """
        统一的训练和测试主流程。
        """
        image, _ = batch
        image = image.to(self.device)

        # 1. 使用预训练的骨干网络提取特征
        with torch.no_grad():
            feat_all = self.emb_func(image)

        # 2. 将特征划分为支持集和查询集
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat_all, mode=1)
        
        episode_size = support_feat.size(0)
        output_list, loss_list, acc_list = [], [], []

        # 3. 遍历-一个 batch 中的所有-任务 (episode)
        for i in range(episode_size):
            # 将当前任务的数据传入 SIFTLayer 进行处理
            output = self.sift_layer(support_feat[i], query_feat[i], support_target[i])
            current_q_target = query_target[i]
            
            # 计算准确率
            acc = accuracy(output, current_q_target)
            
            output_list.append(output)
            acc_list.append(acc)

            # 如果是训练阶段，计算损失
            if is_train:
                loss = self.loss_func(output, current_q_target)
                loss_list.append(loss)
                
        final_output = torch.cat(output_list, dim=0)
        final_acc = np.mean(acc_list)
        
        # 根据阶段返回不同的结果
        if is_train:
            final_loss = torch.stack(loss_list).mean()
            return final_output, final_acc, final_loss
        else:
            return final_output, final_acc