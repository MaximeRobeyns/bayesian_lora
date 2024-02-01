#!/usr/bin/env python
# coding: utf-8

# # Incremental SVD

# In[1]:


import torch as t
from torch import Tensor
from jaxtyping import Float


# In[2]:


device = "cuda:0"
dtype = t.float32


# In[9]:


d, n_kfac, batch = 1024, 1000, 16


# ## Testing the Incrementatl SVD

# Import our library method

# In[10]:


from bayesian_lora.kfac import incremental_svd


# Create a ground-truth, full-rank matrix

# In[11]:


true_B = t.randn(d, d).to(device, dtype)
true_B = true_B@true_B.T


# Calculate the full SVD to get a low-rank factor

# In[12]:


U_full, S_full, _ = t.linalg.svd(true_B, full_matrices=False)
full_B = U_full[:, :n_kfac]@t.diag(S_full[:n_kfac])


# In[13]:


t.norm(true_B - (full_B@full_B.T))


# In[7]:


t.norm(true_B - (full_B@full_B.T))


# In[8]:


assert t.allclose(true_B, full_B@full_B.T)


# ## Using Eigendecomposition

# In[25]:


# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = t.linalg.eigh(true_B)


# In[34]:


# Choose the rank for the approximation
N = 1000

# Select the top N eigenvalues and eigenvectors
top_eigenvalues = eigenvalues[-N:]
top_eigenvectors = eigenvectors[:, -N:]

# Reconstruct the low-rank approximation of the matrix
B_approx = top_eigenvectors @ t.diag(top_eigenvalues) @ top_eigenvectors.T


# In[35]:


t.norm(true_B - B_approx)


# #### Dribs and Drabs

# In[5]:


A = t.randn(d, n_kfac).to(device, dtype)
a = t.randn(d, batch).to(device, dtype)

U, S, _ = t.linalg.svd(A, full_matrices=False)

