#!/usr/bin/env python
# coding: utf-8

# # RoBERTa Testing

# In[1]:


model_id = "FacebookAI/roberta-base"
# model_id = "gpt2"


# In[2]:


import os
import torch as t
import bayesian_lora
try:
    assert(_SETUP)
except NameError:
    os.chdir(os.path.split(bayesian_lora.__path__[0])[0])
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    _SETUP = True


# In[3]:


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
from examples.utils import dsets
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)


# Example dataset usage

# In[6]:


dset_class: dsets.ClassificationDataset = getattr(dsets, "cola")
dset = dset_class(tokenizer, add_space=True, max_len=50)


# In[7]:


sc_loader = dset.loader(is_sc=True, drop_last=False)


# In[12]:


for (prompts, classes, _) in sc_loader:
    ps = tokenizer.batch_decode(prompts['input_ids'])
    for p in ps:
        print(p)
    break


# ## Sequence Classification (single label)

# In[13]:


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)


# In[14]:


model


# In[35]:


inputs = tokenizer("Hello, this is my dog, Java. Woof.", return_tensors="pt")


# In[36]:


with torch.no_grad():
    logits = model(**inputs).logits


# In[37]:


predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]


# In[38]:


# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = RobertaForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)


# ## Sequence Classification (multi label)

# In[48]:


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, problem_type="multi_label_classification", num_labels=3
)


# In[49]:


inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")


# In[50]:


with torch.no_grad():
    logits = model(**inputs).logits


# In[51]:


predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]


# In[43]:


predicted_class_ids


# In[ ]:


# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = RobertaForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-emotion", num_labels=num_labels, problem_type="multi_label_classification"
)

labels = torch.sum(
    torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
).to(torch.float)
loss = model(**inputs, labels=labels).loss

