# -*- coding: utf-8 -*-
"""
SIFT: Semantic-aware Interactive Feature Transfer for Few-shot Learning

Adapted from the original SIFT implementation for LibFewShot framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    import pulp as lp
except ImportError:
    print("Warning: pulp is not installed. Route planning will be disabled.")
    lp = None

from core.utils import accuracy
from .metric_model import MetricModel


class Classifier(nn.Module):
    """Classifier for SIFT model compatibility"""
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


def np_proto(feat, label, way):
    """Calculate prototypes using numpy"""
    feat_proto = np.zeros((way, feat.shape[1]))
    for lb in np.unique(label):
        ds = np.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = np.mean(feat_, axis=0)
    return feat_proto


def tc_proto(feat, label, way):
    """Calculate prototypes using torch"""
    feat_proto = torch.zeros(way, feat.size(1))
    for lb in torch.unique(label):
        ds = torch.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = torch.mean(feat_, dim=0)
    if torch.cuda.is_available():
        feat_proto = feat_proto.type(feat.type())
    return feat_proto


def euclidean_metric(a, b):
    """Calculate euclidean distance metric"""
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class SIFTLayer(nn.Module):
    """SIFT layer for few-shot learning"""
    
    def __init__(self, setting='in'):
        super(SIFTLayer, self).__init__()
        self.setting = setting  # 'in' for inductive, 'tran' for transductive
        
    def forward(self, query_feat, support_feat, support_target, way_num, shot_num, query_num):
        """
        Args:
            query_feat: [episode_size, way_num * query_num, feat_dim]
            support_feat: [episode_size, way_num * shot_num, feat_dim]
            support_target: [episode_size, way_num * shot_num]
            way_num: number of ways
            shot_num: number of shots
            query_num: number of queries
        
        Returns:
            logits: [episode_size, way_num * query_num, way_num]
        """
        episode_size = query_feat.size(0)
        logits_list = []
        
        for ep in range(episode_size):
            support_feat_ep = support_feat[ep]  # [way_num * shot_num, feat_dim]
            query_feat_ep = query_feat[ep]      # [way_num * query_num, feat_dim]
            support_target_ep = support_target[ep]  # [way_num * shot_num]
            
            if self.setting == 'tran':
                # Transductive setting with clustering
                logits_ep = self._transductive_forward(
                    query_feat_ep, support_feat_ep, support_target_ep, way_num, shot_num
                )
            else:
                # Inductive setting
                logits_ep = self._inductive_forward(
                    query_feat_ep, support_feat_ep, support_target_ep, way_num
                )
                
            logits_list.append(logits_ep)
            
        output = torch.stack(logits_list, dim=0)  # [episode_size, way_num * query_num, way_num]
        return output
    
    def _inductive_forward(self, query_feat, support_feat, support_target, way_num):
        """Standard inductive prototypical network forward"""
        protos = tc_proto(support_feat, support_target, way_num)
        protos = F.normalize(protos, dim=1)
        query_feat = F.normalize(query_feat, dim=1)
        logits = euclidean_metric(query_feat, protos)
        return logits
    
    def _transductive_forward(self, query_feat, support_feat, support_target, way_num, shot_num):
        """Transductive forward with clustering"""
        # Convert to numpy for clustering
        Xq = query_feat.detach().cpu().numpy()
        Xs = support_feat.detach().cpu().numpy()
        ys = support_target.detach().cpu().numpy()
        
        try:
            if shot_num == 1:
                km = KMeans(n_clusters=way_num, max_iter=1000, random_state=100, n_init=10)
            else:
                p_np = np_proto(Xs, ys, way_num)
                km = KMeans(n_clusters=way_num, init=p_np, max_iter=1000, random_state=100, n_init=1)

            yq_fit = km.fit(Xq)
            clus_center = yq_fit.cluster_centers_
            
            # Update prototypes
            proto = self._updateproto(Xs, ys, clus_center, way_num)
            proto = torch.tensor(proto, dtype=query_feat.dtype, device=query_feat.device)
            proto = F.normalize(proto, dim=1)
            
        except Exception:
            # Fallback to inductive setting if clustering fails
            proto = tc_proto(support_feat, support_target, way_num)
            proto = F.normalize(proto, dim=1)

        query_feat = F.normalize(query_feat, dim=1)
        logits = euclidean_metric(query_feat, proto)
        return logits
    
    def _updateproto(self, Xs, ys, cls_center, way):
        """Update prototypes using clustering centers"""
        proto = np_proto(Xs, ys, way)
        dist = ((proto[:, np.newaxis, :] - cls_center[np.newaxis, :, :])**2).sum(2)
        id = dist.argmin(1)
        feat_proto = np.zeros((way, Xs.shape[1]))
        for i in range(way):
            feat_proto[i] = (cls_center[id[i]] + proto[i]) / 2
        return feat_proto


class SIFT(MetricModel):
    """SIFT model implementation for LibFewShot"""
    
    def __init__(self, setting='in', **kwargs):
        super(SIFT, self).__init__(**kwargs)
        
        self.setting = setting  # 'in' for inductive, 'tran' for transductive
        self.sift_layer = SIFTLayer(setting=setting)
        self.loss_func = nn.CrossEntropyLoss()
        
        # Add classifier for backward compatibility
        # This will be properly initialized when we know the feature dimension
        self.classifier = None
        self._classifier_initialized = False

    def _initialize_classifier(self, feat_dim):
        """Initialize classifier for backward compatibility"""
        if not self._classifier_initialized:
            self.classifier = Classifier(self.way_num, feat_dim).to(self.device)
            self._classifier_initialized = True

    def set_forward(self, batch):
        """Forward pass for evaluation"""
        images, global_targets = batch
        images = images.to(self.device)
        
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(images)
        
        # Initialize classifier if needed for backward compatibility
        self._initialize_classifier(feat.size(-1))
        
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        
        output = self.sift_layer(
            query_feat, support_feat, support_target, 
            self.way_num, self.shot_num, self.query_num
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        """Forward pass for training"""
        images, global_targets = batch
        images = images.to(self.device)
        
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(images)
        
        # Initialize classifier if needed for backward compatibility
        self._initialize_classifier(feat.size(-1))
        
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        
        output = self.sift_layer(
            query_feat, support_feat, support_target, 
            self.way_num, self.shot_num, self.query_num
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc, loss
def route_plan(Dij):
    """Route planning using linear programming"""
    if lp is None:
        raise ImportError("pulp is required for route planning. Install with: pip install pulp")
    
    K = Dij.shape[0]
    model = lp.LpProblem(name='plan_0_1', sense=lp.LpMinimize)
    x = [[lp.LpVariable("x_{},{}".format(i, j), cat="Binary") for j in range(K)] for i in range(K)]
    
    # objective
    objective = 0
    for i in range(K):
        for j in range(K):
            objective = objective + Dij[i, j] * x[i][j]
    model += objective
    
    # constraints
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

    W = np.zeros((K, K))
    for v in model.variables():
        idex = [int(s) for s in v.name.split('_')[1].split(',')]
        W[idex[0], idex[1]] = v.varValue
    return W


def np_proto(feat, label, way):
    """Calculate prototypes using numpy"""
    feat_proto = np.zeros((way, feat.shape[1]))
    for lb in np.unique(label):
        ds = np.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = np.mean(feat_, axis=0)
    return feat_proto


def tc_proto(feat, label, way):
    """Calculate prototypes using torch"""
    feat_proto = torch.zeros(way, feat.size(1))
    for lb in torch.unique(label):
        ds = torch.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = torch.mean(feat_, dim=0)
    if torch.cuda.is_available():
        feat_proto = feat_proto.type(feat.type())
    return feat_proto


def updateproto_(Xs, ys, cls_center, way):
    """Update prototypes with route planning"""
    proto = np_proto(Xs, ys, way)
    dist = ((proto[:, np.newaxis, :]-cls_center[np.newaxis, :, :])**2).sum(2)
    W = route_plan(dist)
    _, id = np.where(W > 0)
    feat_proto = np.zeros((way, Xs.shape[1]))
    for i in range(way):
        feat_proto[i] = (proto[i] + cls_center[id[i]])/2
    return feat_proto


def euclidean_metric(a, b):
    """Calculate euclidean distance metric"""
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def compactness_loss(gen_feat, gen_label, proto, supp_label):
    """Calculate compactness loss"""
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = 0
    for lb in torch.unique(supp_label):
        id = torch.where(gen_label == lb)[0]
        if len(id) > 0:
            loss = loss + loss_fn(gen_feat[id], proto[lb])
    return loss


class Classifier(nn.Module):
    """Classifier for SIFT"""
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


class FClayer(nn.Module):
    """Fully connected layer for SIFT"""
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


class SIFT(MetricModel):
    """SIFT model implementation for LibFewShot"""
    
    def __init__(self, mode='dc', classifier_method='metric', setting='in', 
                 lr=0.001, grad_lr=0.01, ablation='no', cls='lr', **kwargs):
        super(SIFT, self).__init__(**kwargs)
        
        self.mode = mode
        self.classifier_method = classifier_method
        self.setting = setting
        self.lr = lr
        self.grad_lr = grad_lr
        self.ablation = ablation
        self.cls = cls
        
        # For semantic features - this might need adjustment based on dataset
        if hasattr(self, 'dataset') and self.dataset == 'cub':
            z_sem = 312
        else:
            z_sem = 300
            
        # We'll set z_dim after first forward pass
        self.z_dim = None
        self.z_sem = z_sem
        self.components_initialized = False
        
        self.loss_func = nn.CrossEntropyLoss()

    def _initialize_components(self, feat_dim):
        """Initialize SIFT components after knowing feature dimension"""
        if self.components_initialized:
            return
            
        self.z_dim = feat_dim
        
        # Initialize components based on mode
        if self.mode == 'st':
            self.fc_en = FClayer(self.z_sem, self.z_dim).to(self.device)
            self.trans = nn.Linear(self.z_sem, self.z_sem).to(self.device)
            self.fc_de = FClayer(self.z_dim, self.z_sem).to(self.device)
            self.classifier = Classifier(self.way_num, self.z_dim).to(self.device)
        elif self.mode == 'dc':
            self.classifier = Classifier(self.way_num, self.z_dim).to(self.device)
        elif self.mode == 'ns':
            self.contloss = torch.nn.CrossEntropyLoss()
            self.transNet = nn.Sequential(
                nn.Linear(self.z_dim, self.z_sem), 
                nn.Linear(self.z_sem, self.z_dim)
            ).to(self.device)
            
        self.components_initialized = True

    def set_forward(self, batch):
        """Forward pass for evaluation"""
        images, global_targets = batch
        images = images.to(self.device)
        
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(images)
        
        # Initialize components if needed
        self._initialize_components(feat.size(-1))
        
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        
        # Process each episode
        logits_list = []
        for ep in range(episode_size):
            support_feat_ep = support_feat[ep]  # way_num * shot_num, feat_dim
            query_feat_ep = query_feat[ep]      # way_num * query_num, feat_dim
            support_target_ep = support_target[ep]  # way_num * shot_num
            
            if self.mode == 'dc':
                logits_ep = self.dc_forward(support_feat_ep, support_target_ep, query_feat_ep)
            else:
                # For other modes, use metric-based approach
                logits_ep = self.metric_forward(support_feat_ep, support_target_ep, query_feat_ep)
                
            logits_list.append(logits_ep)
            
        output = torch.stack(logits_list, dim=0)  # episode_size, way_num * query_num, way_num
        output = output.view(-1, self.way_num)    # (episode_size * way_num * query_num), way_num
        
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        """Forward pass for training"""
        images, global_targets = batch
        images = images.to(self.device)
        
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(images)
        
        # Initialize components if needed
        self._initialize_components(feat.size(-1))
        
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        
        # Process each episode
        logits_list = []
        for ep in range(episode_size):
            support_feat_ep = support_feat[ep]
            query_feat_ep = query_feat[ep]
            support_target_ep = support_target[ep]
            
            if self.mode == 'dc':
                logits_ep = self.dc_forward(support_feat_ep, support_target_ep, query_feat_ep)
            else:
                logits_ep = self.metric_forward(support_feat_ep, support_target_ep, query_feat_ep)
                
            logits_list.append(logits_ep)
            
        output = torch.stack(logits_list, dim=0)
        output = output.view(-1, self.way_num)
        
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc, loss

    def dc_forward(self, feat_s, label_s, feat_q):
        """Domain conversion forward pass"""
        if self.classifier_method == 'gradient':
            logits = self.classifier(feat_s)
            loss = F.cross_entropy(logits, label_s)
            grad = torch.autograd.grad(loss, self.classifier.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.grad_lr * p[0], zip(grad, self.classifier.parameters())))

            for _ in range(1, 100):
                logits = self.classifier(feat_s, fast_weights)
                loss = F.cross_entropy(logits, label_s)
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.grad_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.classifier(feat_q, fast_weights)

        elif self.classifier_method == 'metric':
            protos = tc_proto(feat_s, label_s, self.way_num)
            logits_q = euclidean_metric(feat_q, protos)

        return logits_q

    def metric_forward(self, feat_s, label_s, feat_q):
        """Standard metric-based forward pass"""
        # Use transductive clustering if specified
        if self.setting == 'tran':
            # Convert to numpy for clustering
            Xq = feat_q.detach().cpu().numpy()
            Xs = feat_s.detach().cpu().numpy()
            ys = label_s.detach().cpu().numpy()
            
            if self.shot_num == 1:
                km = KMeans(n_clusters=self.way_num, max_iter=1000, random_state=100)
            else:
                p_np = np_proto(Xs, ys, self.way_num)
                km = KMeans(n_clusters=self.way_num, init=p_np, max_iter=1000, random_state=100)

            yq_fit = km.fit(Xq)
            clus_center = yq_fit.cluster_centers_
            proto1 = updateproto_(Xs, ys, clus_center, self.way_num)
            proto1 = torch.tensor(proto1, dtype=feat_s.dtype, device=feat_s.device)
            proto1 = F.normalize(proto1, dim=1)
            protos = proto1
        else:
            # Inductive setting
            protos = tc_proto(feat_s, label_s, self.way_num)
            protos = F.normalize(protos, dim=1)

        logits_q = euclidean_metric(feat_q, protos)
        return logits_q
