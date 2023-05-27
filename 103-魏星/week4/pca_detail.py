##Python实现PCA
import numpy as np
def pca(X,k):
  #mean of each feature
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])

  #normalization
  norm_X=X-mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)

  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)

  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])

  #get new data
  data = np.dot(norm_X,np.transpose(feature))
  return data

'10样本3特征的样本集, 行为样例，列为特征维度'
X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9,  35],
              [42, 45, 11],
              [9,  48, 5],
              [11, 21, 14],
              [8,  5,  15],
              [11, 12, 21],
              [21, 20, 25]])
K = np.shape(X)[1] - 1
print(pca(X,K))




