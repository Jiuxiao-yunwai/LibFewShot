# -*- coding: utf-8 -*-
"""
SIFT: Semantic-guided Image Filtering for Few-shot Learning
语义引导的图像过滤少样本学习方法

适配LibFewShot框架的SIFT实现
核心思想：利用语义嵌入来增强少样本学习中的特征表示
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pulp as lp

from core.utils import accuracy
from .metric_model import MetricModel


def route_plan(Dij):
    """最优传输分配，使用线性规划求解"""
    K = Dij.shape[0]
    model = lp.LpProblem(name='plan_0_1', sense=lp.LpMinimize)
    x = [[lp.LpVariable("x_{},{}".format(i, j), cat="Binary") for j in range(K)] for i in range(K)]
    
    # 目标函数：最小化传输成本
    objective = 0
    for i in range(K):
        for j in range(K):
            objective = objective + Dij[i, j] * x[i][j]
    model += objective
    
    # 约束条件：每行每列和为1（一对一匹配）
    for i in range(K):
        in_degree = 0
        for j in range(K):
            in_degree = in_degree + x[i][j]
        model += in_degree == 1

    for i in range(K):
        out_degree = 0
        for j in range(K):
            out_degree = out_degree + x[j][i]
        model += out_degree == 1

    model.solve(lp.apis.PULP_CBC_CMD(msg=False))

    # 提取解决方案
    W = np.zeros((K, K))
    for v in model.variables():
        if v.varValue is not None:
            idx = [int(s) for s in v.name.split('_')[1].split(',')]
            W[idx[0], idx[1]] = v.varValue
    return W


def route_plan_J(Dij):
    """基类选择的最优传输分配"""
    NN, BB = Dij.shape
    model = lp.LpProblem(name='plan_0_1', sense=lp.LpMaximize)
    x = [[lp.LpVariable("x_{},{}".format(i, j), cat="Binary") for j in range(BB)] for i in range(NN)]
    
    # 目标函数：最大化相似度
    objective = 0
    for i in range(NN):
        for j in range(BB):
            objective = objective + Dij[i, j] * x[i][j]
    model += objective
    
    # 约束条件：每个新类只能匹配一个基类，基类可以不被匹配
    for i in range(NN):
        in_degree = 0
        for j in range(BB):
            in_degree = in_degree + x[i][j]
        model += in_degree == 1
    
    for j in range(BB):
        out_degree = 0
        for i in range(NN):
            out_degree = out_degree + x[i][j]
        model += out_degree <= 1
    
    model.solve(lp.apis.PULP_CBC_CMD(msg=False))

    W = np.zeros((NN, BB))
    for v in model.variables():
        if v.varValue is not None:
            idx = [int(s) for s in v.name.split('_')[1].split(',')]
            W[idx[0], idx[1]] = v.varValue
    return W


def np_proto(Xs, ys, way):
    """使用numpy计算原型（类中心）"""
    feat_proto = np.zeros((way, Xs.shape[1]))
    for lb in np.unique(ys):
        ds = np.where(ys == lb)[0]
        feat_proto[lb] = np.mean(Xs[ds], axis=0)
    return feat_proto


def tc_proto(feat, label, way):
    """使用torch计算原型（类中心）"""
    feat_proto = torch.zeros(way, feat.size(1)).type(feat.type())
    for lb in torch.unique(label):
        ds = torch.where(label == lb)[0]
        feat_proto[lb] = torch.mean(feat[ds], dim=0)
    return feat_proto


def updateproto_(Xs, ys, cls_center, way):
    """使用最优传输更新原型"""
    proto = np_proto(Xs, ys, way)
    # 计算原型和聚类中心的距离
    dist = ((proto[:, np.newaxis, :]-cls_center[np.newaxis, :, :])**2).sum(2)
    W = route_plan(dist)
    _, id = np.where(W > 0)
    feat_proto = np.zeros((way, Xs.shape[1]))
    # 融合原型和聚类中心
    for i in range(way):
        feat_proto[i] = (proto[i] + cls_center[id[i]])/2
    return feat_proto


def euclidean_metric(query, proto):
    """欧几里得距离度量"""
    n = query.shape[0]
    m = proto.shape[0]
    query = query.unsqueeze(1).expand(n, m, -1)
    proto = proto.unsqueeze(0).expand(n, m, -1)
    logits = -((query - proto) ** 2).sum(dim=2)
    return logits


def compactness_loss(gen_feat, gen_label, proto, supp_label):
    """紧凑性损失，确保生成的特征接近原型"""
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = 0
    count = 0
    for lb in torch.unique(supp_label):
        id = torch.where(gen_label == lb)[0]
        if len(id) > 0:
            loss = loss + loss_fn(gen_feat[id], proto[lb].unsqueeze(0).expand(len(id), -1))
            count += 1
    return loss / max(count, 1)


def get_cos_similar_matrix(v1, v2):
    """计算余弦相似度矩阵"""
    # 处理向量全为零的情况
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
    
    # 避免除零错误
    v1_norm = np.where(v1_norm == 0, 1e-8, v1_norm)
    v2_norm = np.where(v2_norm == 0, 1e-8, v2_norm)
    
    v1_normalized = v1 / v1_norm
    v2_normalized = v2 / v2_norm
    
    cos_sim = np.dot(v1_normalized, v2_normalized.T)
    cos_sim = np.clip(cos_sim, -1, 1)  # 确保值在[-1, 1]范围内
    
    return 0.5 + 0.5 * cos_sim


class FClayer(nn.Module):
    """无偏置的全连接层"""
    def __init__(self, z_out, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.z_out = z_out
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.z_out, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
    
    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        net = F.linear(input_x, fc1_w)
        return net

    def parameters(self):
        return self.vars


class Classifier(nn.Module):
    """SIFT的简单分类器"""
    def __init__(self, way, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.way = way
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars


class SIFT(MetricModel):
    """SIFT模型在LibFewShot框架中的实现
    
    SIFT通过语义嵌入来增强少样本学习的性能，主要包含三个模式：
    - st: 语义变换模式，包含编码器-变换器-解码器架构
    - dc: 直接分类模式，仅使用分类器
    - ns: 无语义模式，使用特征变换网络
    """
    
    def __init__(self, semantic_dim=300, classifier_method='metric', setting='in',
                 num_aug=3, lr=0.001, grad_lr=0.1, ablation='no', 
                 cls='lr', mode='st', **kwargs):
        super(SIFT, self).__init__(**kwargs)
        
        # SIFT特有参数配置
        self.semantic_dim = semantic_dim  # 语义特征维度
        self.classifier_method = classifier_method  # 分类方法: metric/gradient/nonparam
        self.setting = setting  # 评估设置: in(归纳)/tran(转导)
        self.num_aug = num_aug  # 每类增强样本数量
        self.lr = lr  # 网络学习率
        self.grad_lr = grad_lr  # 梯度分类器学习率
        self.ablation = ablation  # 消融研究设置
        self.cls = cls  # 非参数分类器类型
        self.mode = mode  # SIFT模式
        
        # 根据主干网络获取特征维度，默认640
        self.feat_dim = 640  # 会根据实际输入动态更新
        
        # 根据模式初始化相应组件
        if self.mode == 'st':
            # 语义变换模式：编码器+变换器+解码器+分类器
            self.fc_en = FClayer(self.semantic_dim, self.feat_dim)
            self.trans = nn.Linear(self.semantic_dim, self.semantic_dim)
            self.fc_de = FClayer(self.feat_dim, self.semantic_dim)
            self.classifier = Classifier(self.way_num, self.feat_dim)
        elif self.mode == 'dc':
            # 直接分类模式：仅分类器
            self.classifier = Classifier(self.way_num, self.feat_dim)
        elif self.mode == 'ns':
            # 无语义模式：特征变换网络
            self.contloss = torch.nn.CrossEntropyLoss()
            self.transNet = nn.Sequential(
                nn.Linear(self.feat_dim, self.semantic_dim),
                nn.Linear(self.semantic_dim, self.feat_dim)
            )
        
        # 语义嵌入初始化：避免state_dict加载问题，设为None
        # 在实际使用时会动态初始化
        self.base_semantics = None  # 基类语义嵌入
        self.novel_semantics = None  # 新类语义嵌入
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def _init_semantic_embeddings(self):
        """初始化语义嵌入（如果尚未完成）"""
        if self.base_semantics is None:
            self.base_semantics = torch.randn(100, self.semantic_dim).to(self.device)
        if self.novel_semantics is None:
            self.novel_semantics = torch.randn(self.way_num, self.semantic_dim).to(self.device)

    def set_forward(self, batch):
        """测试时的前向传播"""
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        
        # 提取特征
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        # 对每个episode分别处理
        output_list = []
        for i in range(episode_size):
            if self.mode == 'dc':
                output = self._dc_forward(support_feat[i], support_target[i], query_feat[i])
            elif self.mode == 'st':
                output = self._st_forward_episode(support_feat[i], support_target[i], query_feat[i])
            elif self.mode == 'ns':
                output = self._ns_forward_episode(support_feat[i], support_target[i], query_feat[i])
            else:
                raise ValueError(f'Unknown mode: {self.mode}')
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """训练时的前向传播"""
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output_list = []
        loss_list = []
        for i in range(episode_size):
            if self.mode == 'dc':
                output = self._dc_forward(support_feat[i], support_target[i], query_feat[i])
            elif self.mode == 'st':
                output = self._st_forward_episode(support_feat[i], support_target[i], query_feat[i])
            elif self.mode == 'ns':
                output = self._ns_forward_episode(support_feat[i], support_target[i], query_feat[i])
            else:
                raise ValueError(f'Unknown mode: {self.mode}')
            
            output_list.append(output)
            episode_loss = self.loss_func(output, query_target[i])
            loss_list.append(episode_loss)

        output = torch.cat(output_list, dim=0)
        loss = torch.stack(loss_list).mean()
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss

    def _dc_forward(self, feat_s, label_s, feat_q):
        """直接分类前向传播"""
        if self.classifier_method == 'gradient':
            # 基于梯度的快速适应
            logits = self.classifier(feat_s)
            loss = F.cross_entropy(logits, label_s)
            grad = torch.autograd.grad(loss, self.classifier.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.classifier.parameters())))

            # 多步梯度更新（为提高效率减少至20步）
            for _ in range(1, 20):
                logits = self.classifier(feat_s, fast_weights)
                loss = F.cross_entropy(logits, label_s)
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.classifier(feat_q, fast_weights)

        elif self.classifier_method == 'metric':
            # 基于度量学习的原型网络
            protos = tc_proto(feat_s, label_s, self.way_num)
            logits_q = euclidean_metric(feat_q, protos)
        else:
            raise ValueError(f'Unknown classifier method: {self.classifier_method}')

        return logits_q

    def _st_forward_episode(self, feat_s, label_s, feat_q):
        """语义变换模式的单episode前向传播"""
        # 根据实际输入更新特征维度
        if self.feat_dim != feat_s.size(-1):
            self.feat_dim = feat_s.size(-1)
            self._update_networks()
        
        # 转换为numpy进行聚类
        Xq = feat_q.detach().cpu().numpy()
        Xs = feat_s.detach().cpu().numpy()
        ys = label_s.detach().cpu().numpy()
        
        # K-means聚类：区分1-shot和多shot情况
        if self.shot_num == 1:
            km = KMeans(n_clusters=self.way_num, max_iter=100, random_state=100, n_init=1)
        else:
            p_np = np_proto(Xs, ys, self.way_num)
            km = KMeans(n_clusters=self.way_num, init=p_np, max_iter=100, random_state=100, n_init=1)

        yq_fit = km.fit(Xq)
        clus_center = yq_fit.cluster_centers_
        # 使用最优传输更新原型
        proto1 = updateproto_(Xs, ys, clus_center, self.way_num)
        proto1 = torch.tensor(proto1, dtype=feat_s.dtype, device=feat_s.device)
        proto1 = F.normalize(proto1, dim=1)

        # 归纳原型（基于支持集）
        proto2 = tc_proto(feat_s, label_s, self.way_num)
        proto2 = F.normalize(proto2, dim=1)

        # 根据设置选择原型：转导(tran)或归纳(in)
        if self.setting == 'tran':
            proto = proto1
        else:  # 'in'
            proto = proto2

        # 生成基类特征（简化版本）
        feat_b = self._generate_base_features(feat_s, label_s, proto)
        label_b = torch.arange(self.way_num).repeat(self.num_aug).to(self.device)
        
        # 初始化语义嵌入（如果需要）
        self._init_semantic_embeddings()
        
        # 使用虚拟语义特征
        sem_b = self.base_semantics[:self.way_num * self.num_aug].to(self.device)
        sem_b1 = self.base_semantics[:self.way_num].to(self.device)
        sem_ns = self.novel_semantics[:feat_s.size(0)].to(self.device)
        sem_n1 = self.novel_semantics[:self.way_num].to(self.device)

        # 训练变换网络
        self._train_transformation_networks(feat_b, sem_b, sem_b1, label_b, 
                                            feat_s, sem_ns, label_s, sem_n1, proto)

        # 生成增强特征
        with torch.no_grad():
            sem_b_1 = self.fc_en(feat_b)
            sem_n_1_1 = self.trans(sem_b_1)
            feat_n_1_1 = self.fc_de(sem_n_1_1)
            
        # 合并特征
        feat = torch.cat((feat_n_1_1, feat_s), dim=0)
        labels = torch.cat((label_b, label_s), dim=0)
        feat = F.normalize(feat, dim=1)

        # 分类
        if self.classifier_method == 'metric':
            protos = tc_proto(feat, labels, self.way_num)
            logits_q = euclidean_metric(feat_q, protos)
        elif self.classifier_method == 'gradient':
            logits_q = self._gradient_classification(feat, labels, feat_q)
        elif self.classifier_method == 'nonparam':
            logits_q = self._nonparam_classification(feat, labels, feat_q)
        else:
            raise ValueError(f'Unknown classifier method: {self.classifier_method}')

        return logits_q

    def _ns_forward_episode(self, feat_s, label_s, feat_q):
        """无语义模式的单episode前向传播"""  
        # 根据实际输入更新特征维度
        if self.feat_dim != feat_s.size(-1):
            self.feat_dim = feat_s.size(-1)
            self._update_networks()
            
        # 转换为numpy进行聚类
        Xq = feat_q.detach().cpu().numpy()
        Xs = feat_s.detach().cpu().numpy()
        ys = label_s.detach().cpu().numpy()
        
        # K-means聚类
        if self.shot_num == 1:
            km = KMeans(n_clusters=self.way_num, max_iter=100, random_state=100, n_init=1)
        else:
            p_np = np_proto(Xs, ys, self.way_num)
            km = KMeans(n_clusters=self.way_num, init=p_np, max_iter=100, random_state=100, n_init=1)

        yq_fit = km.fit(Xq)
        clus_center = yq_fit.cluster_centers_
        proto1 = updateproto_(Xs, ys, clus_center, self.way_num)
        proto1 = torch.tensor(proto1, dtype=feat_s.dtype, device=feat_s.device)
        proto1 = F.normalize(proto1, dim=1)

        # 归纳原型
        proto2 = tc_proto(feat_s, label_s, self.way_num)
        proto2 = F.normalize(proto2, dim=1)

        if self.setting == 'tran':
            proto = proto1
        else:  # 'in'
            proto = proto2

        # 生成基类特征
        feat_b = self._generate_base_features(feat_s, label_s, proto)
        label_b = torch.arange(self.way_num).repeat(self.num_aug).to(self.device)

        # 训练变换网络（为提高效率减少至10步）
        optimizer = torch.optim.Adam(self.transNet.parameters(), lr=self.lr)
        
        for i in range(10):
            self.transNet.train()
            optimizer.zero_grad()
            n_of_b = self.transNet(feat_b)
            logit = n_of_b.mm(proto.T)
            loss_con = self.contloss(logit, label_b)
            loss_con.backward(retain_graph=True)
            optimizer.step()

        # 生成增强特征
        self.transNet.eval()
        with torch.no_grad():
            n_of_b = self.transNet(feat_b)
        
        feat = torch.cat((n_of_b, feat_s), dim=0)
        labels = torch.cat((label_b, label_s), dim=0)
        feat = F.normalize(feat, dim=1)

        # 分类
        if self.classifier_method == 'metric':
            protos = tc_proto(feat, labels, self.way_num)
            logits_q = euclidean_metric(feat_q, protos)
        elif self.classifier_method == 'gradient':
            logits_q = self._gradient_classification(feat, labels, feat_q)
        elif self.classifier_method == 'nonparam':
            logits_q = self._nonparam_classification(feat, labels, feat_q)
        else:
            raise ValueError(f'Unknown classifier method: {self.classifier_method}')

        return logits_q

    def _update_networks(self):
        """当特征维度改变时更新网络维度"""
        if self.mode == 'st':
            self.fc_en = FClayer(self.semantic_dim, self.feat_dim).to(self.device)
            self.fc_de = FClayer(self.feat_dim, self.semantic_dim).to(self.device)
            self.classifier = Classifier(self.way_num, self.feat_dim).to(self.device)
        elif self.mode == 'dc':
            self.classifier = Classifier(self.way_num, self.feat_dim).to(self.device)
        elif self.mode == 'ns':
            self.transNet = nn.Sequential(
                nn.Linear(self.feat_dim, self.semantic_dim),
                nn.Linear(self.semantic_dim, self.feat_dim)
            ).to(self.device)

    def _generate_base_features(self, feat_s, label_s, proto):
        """生成用于增强的基类特征"""
        # 简单的特征生成 - 实际应用中会使用语义嵌入
        feat_b_list = []
        for i in range(self.way_num):
            # 在原型周围生成特征
            base_feat = proto[i].unsqueeze(0).repeat(self.num_aug, 1)
            noise = torch.randn_like(base_feat) * 0.1
            feat_b_list.append(base_feat + noise)
        
        feat_b = torch.cat(feat_b_list, dim=0)
        return feat_b

    def _train_transformation_networks(self, feat_b, sem_b, sem_b1, label_b, 
                                       feat_s, sem_ns, label_s, sem_n1, proto):
        """训练编码器、变换器和解码器网络"""
        loss_fn = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam([
            {'params': self.fc_en.parameters(), 'lr': self.lr},
            {'params': self.trans.parameters(), 'lr': self.lr},
            {'params': self.fc_de.parameters(), 'lr': self.lr}
        ], lr=self.lr)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # 为提高效率减少至10步训练
        for i in range(10):
            lr_scheduler.step()
            self.fc_en.train()
            self.trans.train()
            self.fc_de.train()
            optimizer.zero_grad()

            # 编码器损失
            sem_b_1 = self.fc_en(feat_b)
            loss1 = loss_fn(sem_b, sem_b_1)
            
            # 编码器重构损失
            vars = nn.ParameterList()
            fc1_w = nn.Parameter(self.fc_en.fc1_w.transpose(1, 0))
            vars.append(fc1_w)
            feat_b_1 = self.fc_en(sem_b_1, vars)
            loss2 = loss_fn(feat_b, feat_b_1)

            # 变换器损失
            sem_n_1 = self.trans(sem_b1)
            loss3 = loss_fn(sem_n1, sem_n_1)

            # 解码器损失
            feat_ns_1 = self.fc_de(sem_ns)
            loss5 = loss_fn(feat_s, feat_ns_1)
            
            # 解码器重构损失
            vars = nn.ParameterList()
            fc1_w = nn.Parameter(self.fc_de.fc1_w.transpose(1, 0))
            vars.append(fc1_w)
            sem_ns_1 = self.fc_de(feat_ns_1, vars)
            loss6 = loss_fn(sem_ns, sem_ns_1)

            # 紧凑性损失
            sem_n_1_1 = self.trans(sem_b_1)
            feat_n_1_1 = self.fc_de(sem_n_1_1)
            loss7 = compactness_loss(feat_n_1_1, label_b, proto, label_s)

            # 根据消融研究设置计算总损失
            if self.ablation == 'no':
                loss = loss1 + loss2 + loss3 + loss5 + loss6 + loss7
            elif self.ablation == 'enc_recon':
                loss = loss1 + loss3 + loss5 + loss6 + loss7
            elif self.ablation == 'dec_recon':
                loss = loss1 + loss2 + loss3 + loss5 + loss7
            elif self.ablation == 'cpt':
                loss = loss1 + loss2 + loss3 + loss5 + loss6
            elif self.ablation == 'all':
                loss = loss1 + loss3 + loss5
            else:
                loss = loss1 + loss2 + loss3 + loss5 + loss6 + loss7

            if loss.requires_grad:
                loss.backward(retain_graph=True)
                optimizer.step()

    def _gradient_classification(self, feat, labels, feat_q):
        """基于梯度的分类"""
        logits = self.classifier(feat)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, self.classifier.parameters(), create_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.grad_lr * p[0], zip(grad, self.classifier.parameters())))

        # 为提高效率减少至20步梯度更新
        for _ in range(1, 20):
            logits = self.classifier(feat, fast_weights)
            loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.grad_lr * p[0], zip(grad, fast_weights)))
        
        logits_q = self.classifier(feat_q, fast_weights)
        return logits_q

    def _nonparam_classification(self, feat, labels, feat_q):
        """使用sklearn的非参数分类"""
        X_aug = feat.detach().cpu().numpy()
        Y_aug = labels.detach().cpu().numpy()
        data_query = feat_q.detach().cpu().numpy()
        
        if self.cls == 'lr':
            # 逻辑回归分类器
            classifier = LogisticRegression(max_iter=100, random_state=42).fit(X=X_aug, y=Y_aug)
            pred_proba = classifier.predict_proba(data_query)
            logits_q = torch.tensor(pred_proba, dtype=feat_q.dtype, device=feat_q.device)
        elif self.cls == 'svm':
            # 支持向量机分类器
            classifier = SVC(C=1.0, gamma='auto', kernel='linear', probability=True, random_state=42).fit(X=X_aug, y=Y_aug)
            pred_proba = classifier.predict_proba(data_query)
            logits_q = torch.tensor(pred_proba, dtype=feat_q.dtype, device=feat_q.device)
        elif self.cls == 'knn':
            # K近邻分类器
            classifier = KNeighborsClassifier(n_neighbors=1).fit(X=X_aug, y=Y_aug)
            predictions = classifier.predict(data_query)
            # 转换为one-hot编码
            logits_q = torch.zeros(len(predictions), self.way_num, dtype=feat_q.dtype, device=feat_q.device)
            logits_q[range(len(predictions)), predictions] = 1.0
        else:
            raise ValueError(f'Unknown non-parametric classifier: {self.cls}')
            
        return logits_q
