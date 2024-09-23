#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/eunhyechoe/psyc60/blob/main/1_data_visualization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # **1. Data Visualization**
#   
# #### Plot time-series, response patterns, and functional connectivity profiles

# ## 0. Setup environment for Colab
# 
# Colab's virtual environment is temporary, so you need to install any non-default packages each time you run the script. Run the following cell first to set up the environment.

# In[1]:


get_ipython().run_cell_magic('capture', '', 'pip install -U neuroboros')


# ## 1. Import Packages
# Colab comes with pre-installed packages like `numpy` and `matplotlib`, so you can import them without installation. We also import `neuroboros` under the simplified alias `nb` for convenience.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import pearsonr

import neuroboros as nb


# ## 2. Import searchlight indices
# 
# The brain has a complex 3D structure. This can be simplified by representing its surface as a mesh of small triangles. Using the `nb.sls` function, you can extract coordinates from a spherical region on this surface mesh.
# 
# `nb.sls` returns the vertex indices for each searchlight as a list of `numpy` arrays. In this step, you will focus on the right hemisphere with a 15 mm radius, using the `onavg` (OpenNeuro Average) template at ico8 resolution.

# In[3]:


# rh, radius = 15 mm
SLS_ico8_15mm_rh = nb.sls('r', 15, space='onavg-ico32', center_space='onavg-ico8', return_dists=False)


# In[4]:


SLS_ico8_15mm_rh[0]


# In[5]:


# rh, radius = 20 mm
SLS_ico8_20mm_rh, Dists_ico8_20mm_rh = nb.sls('r', 20, space='onavg-ico32', center_space='onavg-ico8', return_dists=True)


# In[6]:


len(SLS_ico8_20mm_rh)


# In[7]:


SLS_ico8_20mm_rh[0]


# In[8]:


Dists_ico8_20mm_rh[0]


# ## 3. Load the data
# 
# We will load the [StudyForrest](https://www.studyforrest.org/) data in `numpy` format using the `neuroboros` package. For detailed explanations of the code, refer to the [Neuroboros](https://neuroboros.github.io/tutorials/neuroimaging_data_matrix.html) tutorial.

# In[9]:


dset = nb.Forrest()
sids = dset.subjects
type(sids), len(sids)


# In[10]:


sids[:3]


# We set the first subject ID of the list to `sid`, and get the data as `dm` for the following parameters: `l`eft hemisphere, from the `1`st run of the `forrest` task, for `sid`.

# In[11]:


sid = sids[0]
dm = dset.get_data(sid, 'forrest', 1, 'l')


# In[25]:


type(dm), dm.dtype


# In[26]:


dm.shape


# 

# ## 4. Plot Time-series
# Each row of the data represents the time points (TR), and the columns correspond to the cortical vertices (9675). You can plot a response time series of a cortical vertex. Here, we can plot one for the first vertex index.

# In[24]:


fig, ax = plt.subplots(1, 1, figsize=(4, 1), dpi=200)
ax.plot(dm[:, 0])
ax.set_xlabel('Time points (TR)')
plt.show()


# ## 5. Plot Response Pattern
# You can also check the spatial response pattern for a specific time point, for example, TR = 0.

# In[30]:


nb.plot([dm[0], dset.get_data(sid, 'forrest', 1, 'r')[0]],
       cmap='bwr', vmax=4, vmin=-4, title='Response pattern')


# calculate searchlight means for one ico8 searchlight (262 for FFA)

# In[39]:


# lh, radius = 20 mm
sls = nb.sls('l', radius=20)
type(sls), len(sls)


# In[40]:


type(sls[0]), len(sls[0])


# The first vertex has 119 vertex indices of the searchlight.

# In[41]:


sls[0]


# We can extract the data for the first searchlight.

# In[42]:


sl_dm = dm[:, sls[0]]
sl_dm.shape


# ## 6. Funcional Connectivity Profile
# 
# We can also quantify how brain regions are synchronized in time series by computing functional connectivity between two vertices using the Pearson correlation along the time series.

# In[62]:


dm.shape


# To simplify the calculation, we downsample the data matrix from ico32 resolution to ico8 resolution.

# In[44]:


xfm = nb.mapping('l', 'onavg-ico32', 'onavg-ico8', mask=True)
xfm.shape


# In[52]:


dm_ico32 = dm
dm_ico8 = dm_ico32 @ xfm


# In[54]:


print(dm_ico32.shape, xfm.shape, dm_ico8.shape)


# Now the data consist of a total of 603 vertices. By calculating the correlation along the time series for all pairs of vertices, we can generate the connectivity matrix.

# In[60]:


d = pdist(dm_ico8.T, 'correlation')
mat = 1 - squareform(d)
print(d.shape, mat.shape)


# In[61]:


fig, ax = plt.subplots(1, 1, figsize=[_/2.54 for _ in [6, 6]], dpi=300)
im = ax.imshow(mat, vmax=1, vmin=-1, cmap='viridis')

ax.set_title('Functional connectivity matrix', size=7, pad=3)
ax.tick_params('both', labelsize=5, size=2, pad=1)
ax.set_xlabel('Vertices', size=7, labelpad=1)
ax.set_ylabel('Vertices', size=7, labelpad=1)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.ax.tick_params(labelsize=5, size=2, pad=1)
cb.ax.set_ylabel('Correlation', size=7)
plt.show()

