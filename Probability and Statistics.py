#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torch.distributions.multinomial import Multinomial
import matplotlib.pyplot as plt


# In[2]:


# sampling 100 of heads and tails by tossing a fair coin
probs = torch.tensor([0.5, 0.5])
sample = Multinomial(100, probs).sample()
sample


# In[3]:


# computing the frequencies
sample / 100


# In[4]:


# simulating 10000 tosses 
Multinomial(10000, probs).sample() / 10000 


# As you can see as the number of tosses is big, the frequencies of heads and tails get closer, this phenomenon is called the law of large numbers. And by definition of the central limit theorem, as the sample size (n) grows, the error goes down by at a rate of 1/sqrt(n).

# In[5]:


# studying the effect of sampling a large sample on a coin of 2 outcomes (heads, tails)
sample = Multinomial(1, probs).sample((10000,))
sample_counts = sample.cumsum(dim=0)
sample_counts, sample_counts.shape


# In[6]:


# computing an estimator
estimates = sample_counts / sample_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()
estimates,  estimates.shape


# In[7]:


# drawing sample numbers vs the estimated probability of each outcome
plt.figure(figsize=(4.5, 4.5))
plt.plot(estimates[:, 0], label="P(coin=heads)")
plt.plot(estimates[:, 1], label="P(coin=tails)")
plt.axhline(y=0.5, color='black', linestyle='dashed') # line of the expected probability
plt.xlabel('Samples')
plt.ylabel('Estimated probability')
plt.legend()

