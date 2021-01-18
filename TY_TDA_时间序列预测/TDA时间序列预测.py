#!/usr/bin/env python
# coding: utf-8

# # Topology in time series forecasting
# 
# This notebook shows how ``giotto-tda`` can be used to create topological features for time series forecasting tasks, and how to integrate them into ``scikit-learn``–compatible pipelines.
# 
# In particular, we will concentrate on topological features which are created from consecutive **sliding windows** over the data. In sliding window models, a single time series array ``X`` of shape ``(n_timestamps, n_features)`` is turned into a time series of windows over the data, with a new shape ``(n_windows, n_samples_per_window, n_features)``. There are two main issues that arise when building forecasting models with sliding windows:
# 1. ``n_windows`` is smaller than ``n_timestamps``. This is because we cannot have more windows than there are timestamps without padding ``X``, and this is not done by ``giotto-tda``. ``n_timestamps - n_windows`` is even larger if we decide to pick a large stride between consecutive windows. 
# 2. The target variable ``y`` needs to be properly "aligned" with each window so that the forecasting problem is meaningful and e.g. we don't "leak" information from the future. In particular, ``y`` needs to be "resampled" so that it too has length ``n_windows``.
# 
# To deal with these issues, ``giotto-tda`` provides a selection of transformers with ``resample``, ``transform_resample`` and ``fit_transform_resample`` methods. These are inherited from a ``TransformerResamplerMixin`` base class. Furthermore, ``giotto-tda`` provides a drop-in replacement for ``scikit-learn``'s ``Pipeline`` which extends it to allow chaining ``TransformerResamplerMixin``s with regular ``scikit-learn`` estimators.
# 
# If you are looking at a static version of this notebook and would like to run its contents, head over to [GitHub](https://github.com/giotto-ai/giotto-tda/blob/master/examples/time_series_forecasting.ipynb) and download the source.
# 
# **License: AGPLv3**
# 该笔记本显示了如何使用giotto-tda创建时间序列预测任务的拓扑特征，以及如何将其集成到scikit-learn兼容的管道中。 
# 特别是，我们将专注于从数据上连续滑动的窗口创建的拓扑特征。在滑动窗口模型中，形状为（n_timestamps，n_features）的单个时间序列数组X变为数据上具有新形状（n_windows，n_samples_per_window，n_features）的窗口的时间序列。使用滑动窗口构建预测模型时会出现两个主要问题： 
# 1.n_windows小于n_timestamps。这是因为我们不能在没有填充X的情况下拥有超过时间戳的窗口，并且giotto-tda不能做到这一点。如果我们决定在连续的窗口之间选择较大的跨度，则n_timestamps-n_windows会更大。 
# 2.目标变量y需要与每个窗口适当地“对齐”，以便预测问题是有意义的，例如。我们不会从未来“泄漏”信息。特别是，需要对y进行“重新采样”，使其长度也为n_windows。 
# 为了解决这些问题，giotto-tda提供了带有resample，transform_resample和fit_transform_resample方法的转换器选择。这些是从TransformerResamplerMixin基类继承的。此外，giotto-tda替代了scikit-learn的Pipeline，从而扩展了它，以允许将TransformerResamplerMixins与常规的scikit-learn估计器链接在一起。 

# ## ``SlidingWindow``
# 
# Let us start with a simple example of a "time series" ``X`` with a corresponding target ``y`` of the same length.
# 让我们从具有相同长度的相应目标y的“时间序列” X的简单示例开始。

# In[1]:


import numpy as np

n_timestamps = 10
X, y = np.arange(n_timestamps), np.arange(n_timestamps) - n_timestamps
X, y


# We can instantiate our sliding window transformer-resampler and run it on the pair ``(X, y)``:

# 我们可以实例化滑动窗口转换器-重采样器，然后在（X，y）对上运行它

# In[2]:


from gtda.time_series import SlidingWindow

window_size = 3
stride = 2

SW = SlidingWindow(size=window_size,stride=stride)
X_sw, yr = SW.fit_transform_resample(X, y)
X_sw, yr


# We note a couple of things:
# - ``fit_transform_resample`` returns a pair: the window-transformed ``X`` and the resampled and aligned ``y``.
# - ``SlidingWindow`` has made a choice for us on how to resample ``y`` and line it up with the windows from ``X``: a window on ``X`` corresponds to the *last* value in a corresponding window over ``y``. This is common in time series forecasting where, for example, ``y`` could be a shift of ``X`` by one timestamp.
# - Some of the initial values of ``X`` may not be found in ``X_sw``. This is because ``SlidingWindow`` only ensures the *last* value is represented in the last window, regardless of the stride. 
# 
# 注意以下几点： 
# fit_transform_resample返回一对：经过窗口转换的X和经过重新采样并对齐的y。
# SlidingWindow已为我们选择了如何对y重新采样并将其与X中的窗口对齐：X上的窗口与y上相应窗口中的最后一个值相对应。这在时间序列预测中很常见，例如，y可能是X偏移一个时间戳。 
# X的某些初始值可能在X_sw中找不到。这是因为SlidingWindow仅确保最后一个值在最后一个窗口中表示，而不管步幅如何。

# ## Multivariate time series example: Sliding window + topology ``Pipeline``

# ``giotto-tda``'s topology transformers expect 3D input. But our ``X_sw`` above is 2D. How do we capture interesting properties of the topology of input time series then? For univariate time series, it turns out that a good way is to use the "time delay embedding" or "Takens embedding" technique explained in the first part of [Topology of time series](https://github.com/giotto-ai/giotto-tda/blob/master/examples/time_series_classification.ipynb). But as this involves an extra layer of complexity, we leave it for later and concentrate for now on an example with a simpler API which also demonstrates the use of a ``giotto-tda`` ``Pipeline``.
# Surprisingly, this involves multivariate time series input!
# 
# giotto-tda的拓扑转换器需要3D输入。但是我们上面的X_sw是2D。那么，我们如何捕获输入时间序列拓扑的有趣属性？对于单变量时间序列，事实证明，一种好的方法是使用时间序列拓扑第一部分中介绍的“时间延迟嵌入”或“令牌嵌入”技术。但这涉及额外的一层复杂性，因此我们将其留待以后，现在集中讨论一个具有更简单API的示例，该示例还演示了giotto-tda Pipeline的用法。 令人惊讶的是，这涉及到多元时间序列输入！

# In[3]:


from numpy.random import default_rng


# In[4]:


rng = np.random.default_rng(42)

n_features = 2

X = rng.integers(0, high=20, size=(n_timestamps, n_features), dtype=int)
X


# We are interpreting this input as a time series in two variables, of length ``n_timestamps``. The target variable is the same ``y`` as before.

# In[5]:


SW = SlidingWindow(size=window_size, stride=stride)
X_sw, yr = SW.fit_transform_resample(X, y)
X_sw, yr


# ``X_sw`` is now a complicated-looking array, but it has a simple interpretation. Again, ``X_sw[i]`` is the ``i``-th window on ``X``, and it contains ``window_size`` samples from the original time series. This time, the samples are not scalars but 1D arrays.
# 
# What if we suspect that the way in which the **correlations** between the variables evolve over time can help forecast the target ``y``? This is a common situation in neuroscience, where each variable could be data from a single EEG sensor, for instance.
# 
# ``giotto-tda`` exposes a ``PearsonDissimilarity`` transformer which creates a 2D dissimilarity matrix from each window in ``X_sw``, and stacks them together into a single 3D object. This is the correct format (and information content!) for a typical topological transformer in ``gtda.homology``. See also [Topological feature extraction from graphs](https://github.com/giotto-ai/giotto-tda/blob/master/examples/persistent_homology_graphs.ipynb) for an in-depth look. Finally, we can extract simple scalar features using a selection of transformers in ``gtda.diagrams``.

# In[6]:


from gtda.time_series import PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude

PD = PearsonDissimilarity()
X_pd = PD.fit_transform(X_sw)
VR = VietorisRipsPersistence(metric="precomputed")
X_vr = VR.fit_transform(X_pd)  # "precomputed" required on dissimilarity data
Ampl = Amplitude()
X_a = Ampl.fit_transform(X_vr)
X_vr


# Notice that we are not acting on ``y`` above. We are simply creating features from each window using topology! *Note*: it's two features per window because we used the default value for ``homology_dimensions`` in ``VietorisRipsPersistence``, not because we had two variables in the time series initially!
# 
# We can now put this all together into a ``giotto-tda`` ``Pipeline`` which combines both the sliding window transformation on ``X`` and resampling of ``y`` with the feature extraction from the windows on ``X``.
# 
# *Note*: while we could import the ``Pipeline`` class and use its constructor, we use the convenience function ``make_pipeline`` instead, which is a drop-in replacement for [scikit-learn's](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html).
# 
# 请注意，我们没有对上面的“y”采取行动。我们只是在使用拓扑从每个窗口创建特征！ *注意*：每个窗口有两个特征，因为我们在“VietorisRipsPersistence”中使用了“ homology_dimensions”的默认值，而不是因为我们最初在时间序列中有两个变量！ 
# 
# 现在我们可以将所有这些放到giotto-tda“Pipeline”中，它将“X”上的滑动窗口转换和“y”的重采样与从“X”的Windows窗口上提取的特征结合在一起。 
# 
# *注意*：虽然我们可以导入“ Pipeline”类并使用其构造函数，但我们使用便利功能“ make_pipeline”代替，它是[scikit-learn's]（https：// scikit-learn.org/stable/modules/generation/sklearn.pipeline.make_pipeline.html）。

# In[7]:


from sklearn import set_config
set_config(display='diagram')  # For HTML representations of pipelines

from gtda.pipeline import make_pipeline

pipe = make_pipeline(SW, PD, VR, Ampl)
pipe


# Finally, if we have a *regression* task on ``y`` we can add a final estimator such as scikit-learn's ``RandomForestRegressor`` as a final step in the previous pipeline, and fit it!
# 
# 最后，如果在y上有回归任务，我们可以添加最终估计量（例如scikit-learn的RandomForestRegressor）作为上一个管道中的最后一步，并将其拟合！

# In[8]:


from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor()

pipe = make_pipeline(SW, PD, VR, Ampl, RFR)
pipe


# In[9]:


pipe.fit(X, y)
y_pred = pipe.predict(X)
score = pipe.score(X, y)
y_pred, score


# ## Univariate time series – ``TakensEmbedding`` and ``SingleTakensEmbedding``
# 
# The first part of [Topology of time series](https://github.com/giotto-ai/giotto-tda/blob/master/examples/time_series_classification.ipynb) explains a commonly used technique for converting a univariate time series into a single **point cloud**. Since topological features can be extracted from any point cloud, this is a gateway to time series analysis using topology. The second part of that notebook shows how to transform a *batch* of time series into a batch of point clouds, and how to extract topological descriptors from each of them independently. While in that notebook this is applied to a time series classification task, in this notebook we are concerned with topology-powered *forecasting* from a single time series.
# 
# Reasoning by analogy with the multivariate case above, we can look at sliding windows over ``X`` as small time series in their own right and track the evolution of *their* topology against the variable of interest (or against itself, if we are interested in unsupervised tasks such as anomaly detection).
# 
# There are two ways in which we can implement this idea in ``giotto-tda``:
# 1. We can first apply a ``SlidingWindow``, and then an instance of ``TakensEmbedding``.
# 2. We can *first* compute a global Takens embedding of the time series via ``SingleTakensEmbedding``, which takes us from 1D/column data to 2D data, and *then* partition the 2D data of vectors into sliding windows via ``SlidingWindow``.
# 
# The first route ensures that we can run our "topological feature extraction track" in parallel with other feature-generation pipelines from sliding windows, without experiencing shape mismatches. The second route seems a little upside-down and it is not generally recommended, but it has the advantange that globally "optimal" parameters for the "time delay" and "embedding dimension" parameters can be computed automatically by ``SingleTakensEmbedding``. 
# 
# Below is what each route would look like.
# 
# *Remark:* In the presence of noise, a small sliding window size is likely to reduce the reliability of the estimate of the time series' local topology.

# ### Option 1: ``SlidingWindow`` + ``TakensEmbedding``
# 
# ``TakensEmbedding`` is not a ``TransformerResamplerMixin``, but this is not a problem in the context of a ``Pipeline`` when we order things in this way.

# In[10]:


from gtda.time_series import TakensEmbedding

X = np.arange(n_timestamps)

window_size = 5
stride = 2

SW = SlidingWindow(size=window_size, stride=stride)
X_sw, yr = SW.fit_transform_resample(X, y)
X_sw, yr


# In[11]:


X,y


# In[12]:


time_delay = 1
dimension = 2

TE = TakensEmbedding(time_delay=time_delay, dimension=dimension)
X_te = TE.fit_transform(X_sw)
X_te


# In[13]:


VR = VietorisRipsPersistence()  # No "precomputed" for point clouds
Ampl = Amplitude()
RFR = RandomForestRegressor()

pipe = make_pipeline(SW, TE, VR, Ampl, RFR)
pipe


# In[14]:


pipe.fit(X, y)
y_pred = pipe.predict(X)
score = pipe.score(X, y)
y_pred, score


# In[15]:


X


# In[ ]:





# ### Option 2: ``SingleTakensEmbeding`` + ``SlidingWindow``
# 
# Note that ``SingleTakensEmbedding`` is also a ``TransformerResamplerMixin``, and that the logic for resampling/aligning ``y`` is the same as in ``SlidingWindow``.

# In[16]:


from gtda.time_series import SingleTakensEmbedding

X = np.arange(n_timestamps)

STE = SingleTakensEmbedding(parameters_type="search", time_delay=2, dimension=3)
X_ste, yr = STE.fit_transform_resample(X, y)
X_ste, yr


# In[17]:


X


# In[18]:


window_size = 5
stride = 2

SW = SlidingWindow(size=window_size, stride=stride)
X_sw, yr = SW.fit_transform_resample(X_ste, yr)
X_sw, yr


# From here on, it is easy to push a very similar pipeline through as in the multivariate case:

# In[19]:


VR = VietorisRipsPersistence()  # No "precomputed" for point clouds
Ampl = Amplitude()
RFR = RandomForestRegressor()

pipe = make_pipeline(STE, SW, VR, Ampl, RFR)
pipe


# In[20]:


pipe.fit(X, y)
y_pred = pipe.predict(X)
score = pipe.score(X, y)
y_pred, score


# ### Integrating non-topological features
# 
# The best results are obtained when topological methods are used not in isolation but in **combination** with other methods. Here's an example where, in parallel with the topological feature extraction from local sliding windows using **Option 2** above, we also compute the mean and variance in each sliding window. A ``scikit-learn`` ``FeatureUnion`` is used to combine these very different sets of features into a single pipeline object.
# 
# 当不单独使用拓扑方法，而是与其他方法结合使用拓扑方法时，可获得最佳结果。这是一个示例，其中与使用上述选项2从本地滑动窗口中提取拓扑特征并行，我们还计算了每个滑动窗口中的均值和方差。 scikit学习的FeatureUnion用于将这些非常不同的功能集组合到单个管道对象中。

# In[21]:


from functools import partial
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.base import clone

mean = FunctionTransformer(partial(np.mean, axis=1, keepdims=True))
var = FunctionTransformer(partial(np.var, axis=1, keepdims=True))

pipe_topology = make_pipeline(TE, VR, Ampl)

feature_union = FeatureUnion([("window_mean", mean),
                              ("window_variance", var),
                              ("window_topology", pipe_topology)])
    
pipe = make_pipeline(SW, feature_union, RFR)
pipe


# In[22]:


pipe.fit(X, y)
y_pred = pipe.predict(X)
score = pipe.score(X, y)
y_pred, score


# ## Endogeneous target preparation with ``Labeller``
# 
# Let us say that we simply wish to predict the future of a time series from itself. This is very common in the study of financial markets for example. ``giotto-tda`` provides convenience classes for target preparation from a time series. This notebook only shows a very simple example: many more options are described in ``Labeller``'s documentation.
# 
# 我们希望预测一个时间序列的未来。例如，这在金融市场研究中非常普遍。 giotto-tda提供了方便的类，用于根据时间序列进行目标准备。该笔记本仅显示一个非常简单的示例：Labeller的文档中描述了更多选项。
# 
# If we wished to create a target ``y`` from ``X`` such that ``y[i]`` is equal to ``X[i + 1]``, while also modifying ``X`` and ``y`` so that they still have the same length, we could proceed as follows:
# 
# 如果我们希望从“ X”创建目标“ y”，以使“ y [i]”等于“ X [i + 1]”，同时还要修改“ X”和``y''以便它们仍然具有相同的长度，我们可以按照以下步骤进行操作：

# In[23]:


from gtda.time_series import Labeller

X = np.arange(10)

Lab = Labeller(size=1, func=np.max)
Xl, yl = Lab.fit_transform_resample(X, X)
Xl, yl

# 注意，在这种情况下，我们将X的两个副本提供给fit_transform_resample！ 
# 这就是使用端到端管道为将来的拓扑预测做准备的样子。同样，鼓励您在混合中包括自己的非拓扑功能！

# In[24]:


SW = SlidingWindow(size=5)
TE = TakensEmbedding(time_delay=1, dimension=2)
VR = VietorisRipsPersistence()
Ampl = Amplitude()
RFR = RandomForestRegressor()

# Full pipeline including the regressor
pipe = make_pipeline(Lab, SW, TE, VR, Ampl, RFR)
pipe


# In[25]:


pipe.fit(X, X)
y_pred = pipe.predict(X)
y_pred




