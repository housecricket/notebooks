# %% [markdown]
# Let assume that you are a doctor, you evaluating data for ten people and predicting if somebody could get coronavirus.
# In our example. You predict six people will get coronavirus.

# %% [code]
y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# %% [markdown]
# By the end of season, you find only five people had coronavirus.

# %% [code]
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# %% [markdown]
# Building a confusion matrix.

# %% [code]
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# %% [markdown]
# Disply the confusion matrix.

# %% [code]
print('Here is our confusion matrix.')

import pandas as pd
cmtx = pd.DataFrame(
    confusion_matrix(y_true, y_pred, labels=[1, 0]), 
    index=['true:1', 'true:0'], 
    columns=['pred:1', 'pred:0']
)
print(cmtx)

# %% [markdown]
# We calculate the percentage of sick people who are correctly identified as having the condition (also called sensitivity).

# %% [code]
sensitivity = tp / (tp+fn)
print('The percentage of sick people who are correctly identified as having the condition')
print('Sensitivity : %7.3f %%' % (sensitivity*100),'\n')

# %% [markdown]
# We also the percentage of healthy people who are correctly identified as not having the condition (also called specificity).

# %% [code]
specificity = tn / (tn+fp)
print('The percentage of healthy people who are correctly identified as not having the condition')
print('Specificity : %7.3f %%' % (specificity*100))

# %% [markdown]
# Next, we calculate the precision of this algorithm.

# %% [code]
from sklearn.metrics import precision_score
print('The ratio of properly predicted positive clarifications to the total predicted positive clarifications.')
print(precision_score(y_true, y_pred, average=None))

# %% [code]
npv = tn / (tn+fn)
print('The probability that records with a negative predicted result truly should be negative: %7.3f %%' % (npv*100))

# %% [markdown]
# We calculate the proportion of positives that yield negative prediction outcomes with the specific model (also called miss rate or FNR).

# %% [code]
fnr = fp / (fn+tp)
print('The proportion of positives that yield negative prediction outcomes with the specific model: %7.3f %%' % (fnr*100))

# %% [markdown]
# Finally, we calculate the false positive rate (also called FPR).

# %% [code]
fdr = fp / (fp+tp)
print('False discovery rate: %7.3f %%' % (fdr*100))

# %% [markdown]
# We calculate statistical bias, as these cause a difference between a result and a "true" value.

# %% [code]
acc = (tp + tn) / (tp + tn + fp + fn)
print('Accuracy: %7.3f %%' % (acc*100))
