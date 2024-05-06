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


# In[27]:


import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
from examples.utils import dsets
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")


# Example dataset usage

# In[6]:


dset_class: dsets.ClassificationDataset = getattr(dsets, "boolq")
dset = dset_class(tokenizer, add_space=True, max_len=50)


# In[7]:


print(f"The dataset has {dset.n_labels} labels")


# # Single-label classification example

# In[ ]:


import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)


# In[ ]:


import torch
from transformers import AutoTokenizer, BertForSequenceClassification

model_id = "google/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = BertForSequenceClassification.from_pretrained(model_id)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)


# ## Sequence Classification (single label)

# In[8]:


model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    # low_cpu_mem_usage=True,
    torch_dtype=t.bfloat16,
    num_labels=dset.n_labels,
)


# In[24]:


model = model.to(0).train()
opt = t.optim.AdamW(model.parameters(), lr=5e-4)


# In[25]:


loader = dset.loader(is_sc=True)


# In[34]:


def class_to_label(classes, num_labels):
    # dset.n_labels
    problem_type="multi_label_classification"

    labels = t.sum(
        F.one_hot(classes[:, None], num_classes=num_labels), dim=1
    ).to(torch.float)
    return labels


# In[ ]:


loss = model(**inputs, labels=labels).loss


# In[40]:


losses = []
for epoch in range(1):
    for batch in tqdm(loader):
        prompts, classes, _ = batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(0)
        # labels = class_to_label(classes, dset.n_labels).to(0)
        outputs = model(**inputs, labels=classes)
        opt.zero_grad()
        outputs.loss.backward()
        opt.step()
        break


# In[ ]:


test_loader = dset.loader(is_sc=True, split="test")
for batch in test_loader:
    prompts, classes, _ = batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(0)
    outputs = model(**inputs, labels=classes.to(0))
    predicted_class_ids = outputs.logits.argmax(0).item()
    print(predicted_class_ids)
    model.config.id2label[predicted_class_ids]
    break


# In[ ]:


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


# In[51]:


with torch.no_grad():
    logits = model(**inputs).logits

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

