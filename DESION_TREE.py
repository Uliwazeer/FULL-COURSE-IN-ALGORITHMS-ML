from dataclasses import replace
import numpy as np 
from collections import Counter
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([P *np.log2(P) for P in ps if P > 0])



class Node:
    def __init__(self,feature=None,threshold=None,right=None,left=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value
    def is_leaf_node(self):
        return self.value is not None
    
    
class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root =None
    def fit(self,X,y):
        #grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root = self._grow_tree(X,y)
        
    def _grow_tree(self,X,y,depth=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(y))
        # stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self.__most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idx = np.random.choice(n_features,self.n_feats,replace=False)
        #greedy search 
        best_feat,best_thresh = self._best_criteria(X,y,feat_idx)
        left_idxs , right_ixdxs = self._split(X[:,best_feat],best_thresh)
        left = self._grow_tree(X[left_idxs, :],y[left_idxs], depth=+1)
        right = self._grow_tree(X[right_idxs, :],y[right_idxs], depth=+1)
        return Node(best_feat,best_thresh,left,right)
    def _best_criteria(self,X,y,feat_idx):
            best_gain = -1
            split_idx,split_threh =None , None
            for feat_idx in feat_idx:
                X_column = X[:,feat_idx]
                thresholds = np.unique(X_column)
                for thresholds in thresholds:
                    gain = self._information_gain(y,X_column,thresholds)
                    
                    if gain> best_gain:
                        best_gain = gain
                        split_idx = feat_idx 
                        split_threh = thresholds 
                        
            return split_idx, split_threh
        
    def _information_gain(self,y ,X_column,split_threh):
            #parent E
            parent_entropy = entropy(y)
            #generate split
            left_idxs,right_ixdxs = self._split(X_column,split_threh)
            if len(left_idxs) == 0 or len(right_ixdxs) == 0:
                return 0
            #weight average child E
            n = len(y)
            n_l ,n_r =len(left_idxs),len(right_idxs)
            e_l,e_r = entropy(y[left_idxs]),entropy(y[right_ixdxs])
            child_entropy = (n_l/n) * e_l + (n_r/n)* e_r
            
            #return ig 
            ig = parent_entropy - child_entropy
            return ig 
    def _split(self,X_column,split_threh):
        left_idxs = np.argwhere(X_column <= split_threh).flatten()        
        right_idxs = np.argwhere(X_column > split_threh).flatten() 
        return left_idxs, right_idxs       
            
    def predict(self,X):
        #traverse tree
        return np.array([self._traverse_tree(x,self.root) for x in X])
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
            return self._traverse_tree(x,node.right)
        
    def __most_common_label(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        
    