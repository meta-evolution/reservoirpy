#!/usr/bin/env python
# coding: utf-8

# # General Introduction to Reservoir Computing

# ## Summary
# 
# - <a href="#chapter1">Chapter 1 : A simple task</a>
# - <a href="#chapter2">Chapter 2 : Generative models</a>
# - <a href="#chapter3">Chapter 3 : Online learning</a>
# - <a href="#chapter4">Chapter 4:  use case in the wild: robot falling</a>
# - <a href="#chapter5">Chapter 5: use case in the wild: canary song decoding</a>

# In[1]:


import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import reservoirpy as rpy

# just a little tweak to center the plots, nothing to worry about
from IPython.core.display import HTML
HTML("""
<style>
.img-center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    }
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
    }
</style>
""")

rpy.set_seed(42)


# ## Chapter 1 : Reservoir Computing for chaotic timeseries forecasting <span id="chapter1"/>

# **Mackey-Glass timeseries**
# 
# Mackey-Glass equation are a set of delayed differential equations
# describing the temporal behavior of different physiological signal,
# for example, the relative quantity of mature blood cells over time.

# The equations are defined as:
# 
# $$
# \frac{dP(t)}{dt} = \frac{a P(t - \tau)}{1 + P(t - \tau)^n} - bP(t)
# $$
# 
# where $a = 0.2$, $b = 0.1$, $n = 10$, and the time delay $\tau = 17$.
# $\tau$ controls the chaotic behavior of the equations (the higher it is,
# the more chaotic the timeseries becomes.
# $\tau=17$ already gives good chaotic results.)

# In[2]:


from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import nrmse, rsquare

timesteps = 2510
tau = 17
X = mackey_glass(timesteps, tau=tau)

# rescale between -1 and 1
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1


# In[3]:


def plot_mackey_glass(X, sample, tau):

    fig = plt.figure(figsize=(13, 5))
    N = sample

    ax = plt.subplot((121))
    t = np.linspace(0, N, N)
    for i in range(N-1):
        ax.plot(t[i:i+2], X[i:i+2], color=plt.cm.magma(255*i//N), lw=1.0)

    plt.title(f"Timeseries - {N} timesteps")
    plt.xlabel("$t$")
    plt.ylabel("$P(t)$")

    ax2 = plt.subplot((122))
    ax2.margins(0.05)
    for i in range(N-1):
        ax2.plot(X[i:i+2], X[i+tau:i+tau+2], color=plt.cm.magma(255*i//N), lw=1.0)

    plt.title(f"Phase diagram: $P(t) = f(P(t-\\tau))$")
    plt.xlabel("$P(t-\\tau)$")
    plt.ylabel("$P(t)$")

    plt.tight_layout()
    plt.show()


# In[4]:


plot_mackey_glass(X, 500, tau)


# - Not completely unpredictable... (not random)
# - ...but not easily predictable (not periodic)
# - Similar to ECG rhythms, stocks, weather...

# ### 1.1. Task 1: 10 timesteps ahead forecast

# Predict $P(t + 10)$ given $P(t)$.

# #### Data preprocessing

# In[5]:


def plot_train_test(X_train, y_train, X_test, y_test):
    sample = 500
    test_len = X_test.shape[0]
    fig = plt.figure(figsize=(15, 5))
    plt.plot(np.arange(0, 500), X_train[-sample:], label="Training data")
    plt.plot(np.arange(0, 500), y_train[-sample:], label="Training ground truth")
    plt.plot(np.arange(500, 500+test_len), X_test, label="Testing data")
    plt.plot(np.arange(500, 500+test_len), y_test, label="Testing ground truth")
    plt.legend()
    plt.show()


# In[6]:


from reservoirpy.datasets import to_forecasting

x, y = to_forecasting(X, forecast=10)
X_train1, y_train1 = x[:2000], y[:2000]
X_test1, y_test1 = x[2000:], y[2000:]

plot_train_test(X_train1, y_train1, X_test1, y_test1)


# ### Build your first Echo State Network

# In[7]:


units = 100
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
regularization = 1e-8
seed = 1234


# In[8]:


def reset_esn():
    from reservoirpy.nodes import Reservoir, Ridge

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                          lr=leak_rate, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, seed=seed)
    readout   = Ridge(1, ridge=regularization)

    return reservoir >> readout


# In[9]:


from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = Ridge(1, ridge=regularization)

esn = reservoir >> readout


# In[10]:


y = esn(X[0])  # initialisation
reservoir.Win is not None, reservoir.W is not None, readout.Wout is not None


# In[11]:


np.all(readout.Wout == 0.0)


# #### ESN training
# 
# Training is performed *offline*: it happens only once on the whole dataset.

# In[12]:


esn = esn.fit(X_train1, y_train1)


# In[13]:


def plot_readout(readout):
    Wout = readout.Wout
    bias = readout.bias
    Wout = np.r_[bias, Wout]

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(111)
    ax.grid(axis="y")
    ax.set_ylabel("Coefs. of $W_{out}$")
    ax.set_xlabel("reservoir neurons index")
    ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])

    plt.show()


# In[14]:


plot_readout(readout)


# #### ESN test

# In[15]:


def plot_results(y_pred, y_test, sample=500):

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")

    plt.legend()
    plt.show()


# In[16]:


y_pred1 = esn.run(X_test1)


# In[17]:


plot_results(y_pred1, y_test1)


# $R^2$ and NRMSE :

# In[18]:


rsquare(y_test1, y_pred1), nrmse(y_test1, y_pred1)


# ### 1.2 Make the task harder
# 
# Now, let's have a forecasting horizon of 100 timesteps.

# In[19]:


x, y = to_forecasting(X, forecast=100)
X_train2, y_train2 = x[:2000], y[:2000]
X_test2, y_test2 = x[2000:], y[2000:]

plot_train_test(X_train2, y_train2, X_test2, y_test2)


# In[20]:


y_pred2 = esn.fit(X_train2, y_train2).run(X_test2)


# In[21]:


plot_results(y_pred2, y_test2, sample=400)


# $R^2$ and NRMSE:

# In[22]:


rsquare(y_test2, y_pred2), nrmse(y_test2, y_pred2)


# ## Chapter 2 : Use generative mode <span id="chapter2"/>
# 
# - Train ESN on a one-timestep-ahead forecasting task.
# - Run the ESN on its own predictions (closed loop generative mode)

# In[23]:


units = 500
leak_rate = 0.3
spectral_radius = 0.99
input_scaling = 1.0
connectivity = 0.1      # - density of reservoir internal matrix
input_connectivity = 0.2  # and of reservoir input matrix
regularization = 1e-4
seed = 1234             # for reproducibility


# In[24]:


def plot_generation(X_gen, X_t, nb_generations, warming_out=None, warming_inputs=None, seed_timesteps=0):

    plt.figure(figsize=(15, 5))
    if warming_out is not None:
        plt.plot(np.vstack([warming_out, X_gen]), label="Generated timeseries")
    else:
        plt.plot(X_gen, label="Generated timeseries")

    plt.plot(np.arange(nb_generations)+seed_timesteps, X_t, linestyle="--", label="Real timeseries")

    if warming_inputs is not None:
        plt.plot(np.arange(seed_timesteps), warming_inputs, linestyle="--", label="Warmup")

    plt.plot(np.arange(nb_generations)+seed_timesteps, np.abs(X_t - X_gen),
             label="Absolute deviation")

    if seed_timesteps > 0:
        plt.fill_between([0, seed_timesteps], *plt.ylim(), facecolor='lightgray', alpha=0.5, label="Warmup")

    plt.plot([], [], ' ', label=f"$R^2 = {round(rsquare(X_t, X_gen), 4)}$")
    plt.plot([], [], ' ', label=f"$NRMSE = {round(nrmse(X_t, X_gen), 4)}$")
    plt.legend()
    plt.show()


# #### Training for one-timestep-ahead forecast

# In[25]:


esn = reset_esn()

x, y = to_forecasting(X, forecast=1)
X_train3, y_train3 = x[:2000], y[:2000]
X_test3, y_test3 = x[2000:], y[2000:]

esn = esn.fit(X_train3, y_train3)


# #### Generative mode
# 
# - 100 steps of the real timeseries used as warmup.
# - 300 steps generated by the reservoir, without external inputs.

# In[26]:


seed_timesteps = 100

warming_inputs = X_test3[:seed_timesteps]

warming_out = esn.run(warming_inputs, reset=True)  # warmup


# In[27]:


nb_generations = 400

X_gen = np.zeros((nb_generations, 1))
y = warming_out[-1]
for t in range(nb_generations):  # generation
    y = esn(y)
    X_gen[t, :] = y


# In[28]:


X_t = X_test3[seed_timesteps: nb_generations+seed_timesteps]
plot_generation(X_gen, X_t, nb_generations, warming_out=warming_out,
                warming_inputs=warming_inputs, seed_timesteps=seed_timesteps)


# ## Chapter 3 : Online learning <span id="chapter3"/>
# 
# Some learning rules allow to update the readout parameters at every timestep of input series.

# Using **FORCE** algorithm *(Sussillo and Abott, 2009)*

# <div>
#     <img src="./static/online.png" width="700">
# </div>

# In[29]:


units = 100
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
seed = 1234


# In[30]:


from reservoirpy.nodes import FORCE

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = FORCE(1)


esn_online = reservoir >> readout


# #### Step by step training

# In[31]:


outputs_pre = np.zeros(X_train1.shape)
for t, (x, y) in enumerate(zip(X_train1, y_train1)): # for each timestep of training data:
    outputs_pre[t, :] = esn_online.train(x, y)


# In[32]:


plot_results(outputs_pre, y_train1, sample=100)


# In[33]:


plot_results(outputs_pre, y_train1, sample=500)


# #### Training on a whole timeseries

# In[34]:


reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = FORCE(1)


esn_online = reservoir >> readout


# In[35]:


esn_online.train(X_train1, y_train1)

pred_online = esn_online.run(X_test1)  # Wout est maintenant figée


# In[36]:


plot_results(pred_online, y_test1, sample=500)


# $R^2$ and NRMSE:

# In[37]:


rsquare(y_test1, pred_online), nrmse(y_test1, pred_online)


# ## Other timeseries
# 
# Try out the other chaotic timeseries included in ReservoirPy: Lorenz chaotic attractor, Hénon map, Logistic map, Double scroll attractor...

# ## Chapter 4:  use case in the wild: robot falling <span id="chapter4"/>
# 
# Data for this use case can be found on Zenodo: https://zenodo.org/record/5900966

# <div>
#     <img src="./static/sigmaban.gif" width="500">
# </div>

# #### Loading and data pre-processing

# In[38]:


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from joblib import delayed, Parallel
from tqdm import tqdm


# In[39]:


features = ['com_x', 'com_y', 'com_z', 'trunk_pitch', 'trunk_roll', 'left_x', 'left_y',
            'right_x', 'right_y', 'left_ankle_pitch', 'left_ankle_roll', 'left_hip_pitch',
            'left_hip_roll', 'left_hip_yaw', 'left_knee', 'right_ankle_pitch',
            'right_ankle_roll', 'right_hip_pitch', 'right_hip_roll',
            'right_hip_yaw', 'right_knee']

prediction = ['fallen']
force = ['force_orientation', 'force_magnitude']


# In[40]:

# Note: External data files not available, creating mock data for demonstration
print("Creating mock robot data for demonstration...")

# Create mock data
np.random.seed(42)
num_sequences = 5
sequence_length = 2000

X = []
Y = []
F = []

for i in range(num_sequences):
    # Create mock features data
    x_data = np.random.randn(sequence_length, len(features)) * 0.1
    # Add some structure to make it more realistic
    x_data[:, 0] += np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 0.5  # com_x
    x_data[:, 1] += np.cos(np.linspace(0, 4*np.pi, sequence_length)) * 0.3  # com_y
    
    # Create mock fall indicator (binary)
    y_data = np.zeros((sequence_length, 1))
    # Simulate a fall event
    fall_start = np.random.randint(1000, 1500)
    y_data[fall_start:] = 1.0
    
    # Create mock force data
    f_data = np.random.exponential(0.5, sequence_length)
    # Add a force spike before the fall
    f_data[fall_start-100:fall_start] += np.random.exponential(2.0, 100)
    
    X.append(x_data)
    Y.append(y_data)
    F.append(f_data)

# In[41]:

# (Data already created above)

# In[42]:

Y_train = []
for y in Y:
    y_shift = np.roll(y, -500)
    y_shift[-500:] = y[-500:]
    Y_train.append(y_shift)

# In[43]:

def plot_robot(Y, Y_train, F):
    plt.figure(figsize=(10, 7))
    plt.plot(Y_train[1], label="Objective")
    plt.plot(Y[1], label="Fall indicator") 
    plt.plot(F[1], label="Applied force")
    plt.legend()
    plt.show()

# In[44]:

plot_robot(Y, Y_train, F)


# #### Training the ESN

# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, Y_train, test_size=0.2, random_state=42)


# In[46]:

if __name__ == '__main__':
    from reservoirpy.nodes import ESN

    reservoir = Reservoir(300, lr=0.5, sr=0.99, input_bias=False)
    readout   = Ridge(1, ridge=1e-3)
    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)  # version distribuée


    # In[47]:


    esn = esn.fit(X_train, y_train)


    # In[48]:


    res = esn.run(X_test)


    # In[49]:


    from reservoirpy.observables import rmse
    scores = []
    for y_t, y_p in zip(y_test, res):
        # Ensure compatible shapes
        if y_t.shape != y_p.shape:
            min_len = min(len(y_t), len(y_p))
            y_t_trimmed = y_t[:min_len]
            y_p_trimmed = y_p[:min_len]
        else:
            y_t_trimmed = y_t
            y_p_trimmed = y_p
        
        # Ensure both arrays have the same shape
        y_t_trimmed = np.asarray(y_t_trimmed).reshape(-1, 1)
        y_p_trimmed = np.asarray(y_p_trimmed).reshape(-1, 1)
        
        score = rmse(y_t_trimmed, y_p_trimmed)
        scores.append(score)


    filt_scores = []
    for y_t, y_p in zip(y_test, res):
        # Ensure compatible shapes
        if y_t.shape != y_p.shape:
            min_len = min(len(y_t), len(y_p))
            y_t_trimmed = y_t[:min_len]
            y_p_trimmed = y_p[:min_len]
        else:
            y_t_trimmed = y_t
            y_p_trimmed = y_p
        
        # Ensure both arrays have the same shape
        y_t_trimmed = np.asarray(y_t_trimmed).reshape(-1, 1)
        y_p_trimmed = np.asarray(y_p_trimmed).reshape(-1, 1)
        
        y_f = y_p_trimmed.copy()
        y_f[y_f > 0.5] = 1.0
        y_f[y_f <= 0.5] = 0.0
        score = rmse(y_t_trimmed, y_f)
        filt_scores.append(score)


    # In[50]:


    def plot_robot_results(y_test, y_pred):
        for y_t, y_p in zip(y_test, y_pred):
            if y_t.max() > 0.5:
                y_shift = np.roll(y_t, 500)
                y_shift[:500] = 0.0

                plt.figure(figsize=(7, 5))
                plt.plot(y_t, label="Objective")
                plt.plot(y_shift, label="Fall")
                plt.plot(y_p, label="Prediction")
                plt.legend()
                plt.show()
                break


    # In[51]:


    plot_robot_results(y_test, res)


    # In[52]:


    print("Averaged RMSE :", f"{np.mean(scores):.4f}", "±", f"{np.std(scores):.5f}")
    print("Averaged RMSE (with threshold) :", f"{np.mean(filt_scores):.4f}", "±", f"{np.std(filt_scores):.5f}")


# ## Chapter 5: use case in the wild: canary song decoding <span id="chapter5"/>
# 
# Data for this use case can be found on Zenodo :
# https://zenodo.org/record/4736597

# <div>
#     <img src="./static/canary.png" width="500">
# </div>

# In[53]:

# Note: External audio files not available, skipping audio playback
print("Audio file not available for demonstration")

# In[54]:

# Skipped audio display

# Several temporal motifs to classify: the *phrases*
# 
# 
# - There is one label per phrase type.
# - A *SIL* label denotes silence. Silence also needs to be detected to segment songs properly.

# In[55]:

# Note: External image files not available, skipping image display
print("Canary outputs image not available for demonstration")

# #### Loading and data preprocessing

# In[56]:

# Note: External data files and librosa not available, creating mock data for demonstration
print("Creating mock canary song data for demonstration...")

from sklearn.preprocessing import OneHotEncoder

# Create mock MFCC data
np.random.seed(42)
n_mfcc = 13
n_features = n_mfcc * 3  # MFCC + delta + delta2
vocab = ['SIL', 'A', 'B', 'C', 'D']
num_songs = 100

X = []
Y = []

for i in range(num_songs):
    # Create mock MFCC features (variable length sequences)
    seq_length = np.random.randint(100, 500)
    x_data = np.random.randn(seq_length, n_features) * 0.5
    
    # Create mock labels (phrase classification)
    y_data = []
    current_pos = 0
    while current_pos < seq_length:
        # Random phrase length
        phrase_length = np.random.randint(10, 50)
        phrase_length = min(phrase_length, seq_length - current_pos)
        
        # Random phrase type
        phrase_type = np.random.choice(vocab)
        
        # Add labels for this phrase
        y_data.extend([[phrase_type]] * phrase_length)
        current_pos += phrase_length
    
    X.append(x_data)
    Y.append(y_data[:seq_length])  # Ensure same length

# #### One-hot encoding of phrase labels

# In[57]:

one_hot = OneHotEncoder(categories=[vocab], sparse_output=False)

Y = [one_hot.fit_transform(np.array(y)) for y in Y]

# We will conduct a first preliminary trial on 100 songs (90 for training, 10 for testing).
# 
# The dataset contains 459 songs in total. You may improve your results by adding more data and performing cross validation.

# In[58]:

X_train, y_train = X[:-10], Y[:-10]
X_test, y_test = X[-10:], Y[-10:]

# #### ESN training
# 
# We use the special node `ESN` to train our model. This node allows parallelization of states computations, which will speed up the training on this large dataset.

# In[59]:

from reservoirpy.nodes import ESN

units = 1000
leak_rate = 0.05
spectral_radius = 0.5
inputs_scaling = 0.001
connectivity = 0.1
input_connectivity = 0.1
regularization = 1e-5
seed = 1234

reservoir = Reservoir(units, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout = Ridge(ridge=regularization)

esn = ESN(reservoir=reservoir, readout=readout, workers=-1)

# In[60]:

if __name__ == '__main__':
    esn = esn.fit(X_train, y_train)

    # In[61]:

    outputs = esn.run(X_test)

    # In[62]:

    from sklearn.metrics import accuracy_score

    scores = []
    for y_t, y_p in zip(y_test, outputs):
        targets = np.vstack(one_hot.inverse_transform(y_t)).flatten()

        top_1 = np.argmax(y_p, axis=1)
        top_1 = np.array([vocab[t] for t in top_1])

        accuracy = accuracy_score(targets, top_1)

        scores.append(accuracy)

    # In[63]:

    print("Accuracy scores for each test song:", scores)  # for each song in the testing set

    # In[64]:

    print("Average accuracy :", f"{np.mean(scores):.4f}", "±", f"{np.std(scores):.5f}")
