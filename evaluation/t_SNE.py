import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets


class t_sne():
    def __init__(self,n_components,perplexity,early_exaggeration,learning_rate,n_iter):
        super(t_sne, self).__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.tsne = manifold.TSNE(n_components=n_components,perplexity=perplexity,early_exaggeration=early_exaggeration,
                                  learning_rate=learning_rate,n_iter=n_iter,init='pca')

    def run_t_sne(self,X):
        X_rteshape = X.reshape(X.shape[0], -1)
        self.n_samples, self.n_features = X_rteshape.shape
        self.X_tsne = self.tsne.fit_transform(X_rteshape)

    def visulization(self,y):
        y = np.squeeze(y)
        x_min, x_max = self.X_tsne.min(0), self.X_tsne.max(0)
        X_norm = (self.X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(12, 12))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], "*", color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def run_2_t_sne(self, X1, X2):
        X1_rteshape = X1.reshape(X1.shape[0], -1)
        X2_rteshape = X2.reshape(X2.shape[0], -1)
        X_rteshape = np.concatenate((X1_rteshape,X2_rteshape))
        self.n_samples, self.n_features = X_rteshape.shape
        self.X_tsne = self.tsne.fit_transform(X_rteshape)

    def visulization2(self,y1,y2):
        y1 = np.squeeze(y1)
        y2 = np.squeeze(y2)
        y = np.concatenate((y1,y2))
        x_min, x_max = self.X_tsne.min(0), self.X_tsne.max(0)
        X_norm = (self.X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(12, 12))
        for i in range(y1.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], "*", color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        for i in range(y1.shape[0],y2.shape[0]+y1.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], ".", color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()

if __name__ == "__main__":
    x = np.array([[1,22,3],[2,3,3],[4,23,23],[233,3,4]])
    y = np.array([1,2,2,1])
    t = t_sne(n_components=2,perplexity=30,early_exaggeration=12,learning_rate=200,n_iter=250)
    t.run_t_sne(x)
    t.visulization(y)



