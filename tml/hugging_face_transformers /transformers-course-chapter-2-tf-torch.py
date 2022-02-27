# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] papermill={"duration": 0.088395, "end_time": "2022-01-24T19:37:53.159206", "exception": false, "start_time": "2022-01-24T19:37:53.070811", "status": "completed"} tags=[]
# <img src="https://repository-images.githubusercontent.com/155220641/a16c4880-a501-11ea-9e8f-646cf611702e"></img>
#
# ---
#
# <p style="font-family: Georgia;">This collection of notebooks will walk you through the entire <a src="https://huggingface.co/course/chapter0/1?fw=tf" style="font-weight: bold;"><b>Hugging Face Transformers Course:</b></a></p>
#
# ---
#
# <b style="font-family: Georgia;">Links To Chapter Notebooks&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color: red; font-size: 11px;">(This Notebook Will Cover Chapter 2)</span></b>
# <ul style="font-family: Georgia;">
#     <li><a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-1-tf-torch#chapter_0"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">0</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">SEtUP</span></b></a></li>
#     <li><a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-1-tf-torch#chapter_1"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">1</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">Transformer Models</span></b></a></li>
#     <li><a href="#toc"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">2</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">Using 🤗 Transformers</span></b></a></li>
#     <li><a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-3-tf-torch"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">3</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">Fine-Tuning A Pretrained Model</span></b></a></li>
#     <li><a href="#"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">4</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">Sharing Models And Tokenizers</span></b></a></li>
#     <li><a href="#"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">5</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">The 🤗 Datasets Library</span></b></a></li>
#     <li><a href="#"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">6</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">The 🤗 Tokenizers Library</span></b></a></li>
#     <li><a href="#"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">7</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">Main NLP Tasks</span></b></a></li>
#     <li><a href="#"><b>Chapter <span style="font-family: Courier New; font-size: 16px;">8</span> &nbsp;&nbsp; - &nbsp;&nbsp;<span style="letter-spacing: 0.15em; text-transform: uppercase;">How To Ask For Help</span></b></a></li>    
# </ul>
#
# ---
#
# <p style="font-family: Georgia;">To find the original course please <a src="https://huggingface.co/course/chapter0/1?fw=tf" style="font-weight: bold; text-decoration: underline;">>>>click here<<<</a></p>
#     
# ---
#     
# <br>
#
# <center><div class="alert alert-block alert-info" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px;">
#     <br><b>⚠️&nbsp;&nbsp;The vast majority of the text in this notebook will come directly from the HuggingFace Transformers course. If I would like to add in (or change anything), I will insert the information in a blue-box similar to this one.</b><br><br>
# </div></center>
#    
#     
# <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px;">
#     <br><b>⚠️&nbsp;&nbsp;Hugging Face uses green blocks to inject notation into their course.</b><br><br>
# </div></center>
#

# + _kg_hide-input=true _kg_hide-output=true papermill={"duration": 47.889894, "end_time": "2022-01-24T19:38:41.131925", "exception": false, "start_time": "2022-01-24T19:37:53.242031", "status": "completed"} tags=[]
#### PIP INSTALLS ####

#################################################################
######################## Light Version ##########################
#################################################################
# - This installs a very light version of 🤗 Transformers. 
# - In particular, no specific machine learning frameworks are installed. 
# - Since we’ll be using a lot of different features of the library, 
#   we recommend installing the development version, which comes 
#   with all the required dependencies for pretty much 
#   any imaginable use case.
# - Therefore we use transformers[sentencepiece] instead of transformers
# - Note*: We use a -q argument to quiet the output that is displayed
#################################################################
# # !pip install --upgrade transformers

# Full Development Version
# !pip install -q --upgrade transformers[sentencepiece]

# Install Flair NLP library - https://github.com/flairNLP/flair
# !pip install -q --upgrade flair

# + [markdown] papermill={"duration": 0.083326, "end_time": "2022-01-24T19:38:41.299720", "exception": false, "start_time": "2022-01-24T19:38:41.216394", "status": "completed"} tags=[]
# <a id="chapter_2"></a>
#
# <p id="toc"></p>
#
# <h1 style="font-family: Georgia; font-size: 30px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: black; background-color: #ffffff;">TABLE OF CONTENTS</h1>
#
# ---
#
# <h2 style="text-indent: 10vw; font-family: Georgia; font-size: 24px; font-style: normal; font-weight: bolder; text-decoration: none; text-transform: none; letter-spacing: 2px; color:  navy; background-color: #ffffff;"><a href="#toc">CHAPTER &nbsp;#2&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;USING 🤗 TRANSFORMERS</a></h2>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_0">2.0 INTRODUCTION</a></h3>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_1">2.1 BEHIND THE PIPELINE</a></h3>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_2">2.2 MODELS</a></h3>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_3">2.3 TOKENIZER BACKGROUND</a></h3>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_4">2.4 TOKENIZER BASICS</a></h3>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_5">2.5 HANDLING MULTIPLE SEQUENCES</a></h3>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_6">2.6 PUTTING IT ALL TOGETHER</a></h3>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_7">2.7 BASIC USAGE COMPLETED</a></h3>
#
# <h3 style="text-indent: 10vw; font-family: Georgia; font-size: 18px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp; <a href="#2_8">2.8 CHAPTER QUIZ RECAP</a></h3>

# + [markdown] papermill={"duration": 0.08462, "end_time": "2022-01-24T19:38:41.469163", "exception": false, "start_time": "2022-01-24T19:38:41.384543", "status": "completed"} tags=[]
# <a id="2_0"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.0 INTRODUCTION&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# <p style="font-family: Georgia;">As you saw in <b><a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-1-tf-torch">Chapter 1</a></b>, Transformer models are usually very large. With millions to tens of billions of parameters, training and deploying these models is a complicated undertaking. Furthermore, with new models being released on a near-daily basis and each having its own implementation, trying them all out is no easy task.</p>
#
# <p style="font-family: Georgia;">The 🤗 Transformers library was created to solve this problem. Its goal is to provide a single API through which any Transformer model can be loaded, trained, and saved. The library’s main features are:</p>
#     
# <ul style="font-family: Georgia;">
#     <li>
#         <b>Ease of Use:</b> Downloading, loading, and using a state-of-the-art NLP model for inference can be done in just two lines of code.
#     </li>
#     <li>
#         <b>Flexibility:</b> At their core, all models are simple PyTorch nn.Module or TensorFlow tf.keras.Model classes and can be handled like any other models in their respective machine learning (ML) frameworks.
#     </li>
#     <li>
#         <b>Simplicity:</b> Hardly any abstractions are made across the library. The “All in one file” is a core concept: a model’s forward pass is entirely defined in a single file, so that the code itself is understandable and hackable.
#     </li>
# </ul>
#
# <p style="font-family: Georgia;">This last feature makes 🤗 Transformers quite different from other ML libraries. The models are not built on modules that are shared across files; instead, each model has its own layers. In addition to making the models more approachable and understandable, this allows you to easily experiment on one model without affecting others.</p>
#
# <p style="font-family: Georgia;">This chapter will begin with an end-to-end example where we use a model and a tokenizer together to replicate the <b><code>pipeline()</code></b> function introduced in <b><a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-1-tf-torch">Chapter 1</a></b>. Next, we’ll discuss the model API: we’ll dive into the model and configuration classes, and show you how to load a model and how it processes numerical inputs to output predictions.</p>
#
# <p style="font-family: Georgia;">Then we’ll look at the tokenizer API, which is the other main component of the pipeline() function. Tokenizers take care of the first and last processing steps, handling the conversion from text to numerical inputs for the neural network, and the conversion back to text when it is needed. Finally, we’ll show you how to handle sending multiple sentences through a model in a prepared batch, then wrap it all up with a closer look at the high-level <b><code>tokenizer()</code></b> function.</p>
#
# <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px; ">
#     <br><b>⚠️&nbsp;&nbsp;In order to benefit from all features available with the Model Hub and 🤗 Transformers, we recommend <a href="https://huggingface.co/join">creating an account</a>.</b><br><br>
# </div></center>
#  
#
#

# + [markdown] papermill={"duration": 0.083974, "end_time": "2022-01-24T19:38:41.638163", "exception": false, "start_time": "2022-01-24T19:38:41.554189", "status": "completed"} tags=[]
# <a id="2_1"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.1 BEHIND THE PIPELINE&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# <center><div class="alert alert-block alert-info" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px; ">
#     <br><b>⚠️&nbsp;&nbsp;This is the first section where the content is slightly different depending on whether you use PyTorch and TensorFlow. In the course you can toggle to see both versions of the code... in this notebook we will use both libraries simultaneously as a learning tool to compare and contrast Tensorflow and PyTorch</b><br><br>
# </div></center>
#
# - <a href="https://www.youtube.com/watch?v=wVN12smEvqg&t=1s" style="font-family: Georgia; color: #ff6f00; font-weight: bold;">[TENSORFLOW] &nbsp;&nbsp; VIDEO LINK - WHAT HAPPENS INSIDE THE PIPELINE FUNCTION - HUGGING FACE CHANNEL</a>
# - <a href="https://www.youtube.com/watch?v=1pedAIvTWXk" style="font-family: Georgia; color: #EE4C2C; font-weight: bold;">[PYTORCH] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; VIDEO LINK - WHAT HAPPENS INSIDE THE PIPELINE FUNCTION - HUGGING FACE CHANNEL</a><br>
#
#
#
# <p style="font-family: Georgia;">Let’s start with a complete example, taking a look at what happened behind the scenes when we executed the following code in <a href="#chapter_1"><b>Chapter 1</b></a>:</p>

# + papermill={"duration": 22.629254, "end_time": "2022-01-24T19:39:04.351890", "exception": false, "start_time": "2022-01-24T19:38:41.722636", "status": "completed"} tags=[]
# Import the pipeline module
from transformers import pipeline

# Instantiate a classifier for sentiment analysis 
classifier = pipeline("sentiment-analysis")
print(classifier([
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]))

# + [markdown] papermill={"duration": 0.087289, "end_time": "2022-01-24T19:39:04.528252", "exception": false, "start_time": "2022-01-24T19:39:04.440963", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase;">PIPELINE BUILDING BLOCKS</b>
#
#
# <p style="font-family: Georgia;">As we saw in <a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-1-tf-torch"><b>Chapter 1</b></a>, this pipeline groups together three steps:</p>
# <ul style="font-family: Georgia;">
#     <li>
#         Preprocessing
#     </li>
#     <li>
#         Passing the Inputs Through the Model
#     </li>
#     <li>
#         Postprocessing
#     </li>
# </ul>
#
# <center><img src="https://huggingface.co/course/static/chapter2/full_nlp_pipeline.png" width=80%></center>
#
# <br><p style="font-family: Georgia;">Let’s quickly go over each of these steps:</p>

# + [markdown] papermill={"duration": 0.087092, "end_time": "2022-01-24T19:39:04.703158", "exception": false, "start_time": "2022-01-24T19:39:04.616066", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">PREPROCESSING WITH A TOKENIZER</b>
#
# <p style="font-family: Georgia;">Like other neural networks, Transformer models can’t process raw text directly, so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of. To do this we use a tokenizer, which will be responsible for:</p>
#
# <ul style="font-family: Georgia;">
#     <li>
#         Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
#     </li>
#     <li>
#         Mapping each token to an integer
#     </li>
#     <li>
#         Adding additional inputs that may be useful to the model
#     </li>
# </ul>
#
#
#
# <p style="font-family: Georgia;">All this preprocessing needs to be done in exactly the same way as when the model was pretrained, so we first need to download that information from the <b><a href="https://huggingface.co/models">Model Hub</a></b>. To do this, we use the <b><code>AutoTokenizer</code></b> class and its <b><code>from_pretrained()</code></b> method. Using the checkpoint name of our model, it will automatically fetch the data associated with the model’s tokenizer and cache it (so it’s only downloaded the first time you run the code below).</p>
#
# <p style="font-family: Georgia;">Since the default checkpoint of the sentiment-analysis pipeline is <b><code>distilbert-base-uncased-finetuned-sst-2-english</code></b> (you can see its model card <b><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">here</a></b>), we will pass that checkpoint. Once we have the tokenizer, we can directly pass our sentences to it and we’ll get back a dictionary that’s ready to feed to our model! The only thing left to do is to convert the list of input IDs to tensors.</p>
#
# <p style="font-family: Georgia;">You can use 🤗 Transformers without having to worry about which ML framework is used as a backend; it might be <b style="color: #EE4C2C;">PyTorch</b> or <b style="color: #ff6f00;">TensorFlow</b>, <b style="color: darkgreen;">Trax</b>, <b style="color: darkred;">Flax</b>, etc. for some models. However, Transformer models only accept <b>tensors</b> as input. If this is your first time hearing about <b>tensors</b>, you can think of them as <b>NumPy arrays</b> instead. A <b>NumPy array</b> can be a scalar (0D), a vector (1D), a matrix (2D), or have more dimensions. It’s effectively a <b>tensor</b>; other ML frameworks’ <b>tensors</b> behave similarly, and are usually as simple to instantiate as <b>NumPy arrays</b>.</p>
#
# <p style="font-family: Georgia;">To specify the type of tensors we want to get back (<b style="color: #EE4C2C;">PyTorch</b>, <b style="color: #ff6f00;">TensorFlow</b>, or plain <b style="color: navy;">NumPy</b>), we use the return_tensors argument. Don’t worry about padding and truncation just yet; we’ll explain those later. The main things to remember here are that you can pass one sentence or a list of sentences, as well as specifying the type of tensors you want to get back (if no type is passed, you will get a list of lists as a result).</p>
#
# <p style="font-family: Georgia;">The output itself is a dictionary containing two keys, <b><code>input_ids</code></b> and <b><code>attention_mask</code></b>. <b><code>input_ids</code></b> contains two rows of integers (one for each sentence) that are the unique identifiers of the tokens in each sentence. We’ll explain what the <b><code>attention_mask</code></b> is later in this chapter.</p>

# + [markdown] papermill={"duration": 0.088256, "end_time": "2022-01-24T19:39:04.878946", "exception": false, "start_time": "2022-01-24T19:39:04.790690", "status": "completed"} tags=[]
# <br><b style="color: gray; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">LIBRARY AGNOSTIC</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 1.724859, "end_time": "2022-01-24T19:39:06.691771", "exception": false, "start_time": "2022-01-24T19:39:04.966912", "status": "completed"} tags=[]
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

print("\n\n... TOKENIZER ...\n")
print(tokenizer)

print("\n\n\n... RAW INPUTS ...\n")
print(raw_inputs)

# + [markdown] papermill={"duration": 0.057231, "end_time": "2022-01-24T19:39:06.806852", "exception": false, "start_time": "2022-01-24T19:39:06.749621", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.099805, "end_time": "2022-01-24T19:39:06.964093", "exception": false, "start_time": "2022-01-24T19:39:06.864288", "status": "completed"} tags=[]
tf_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")

print("\n\n... TF INPUTS ...\n")
print(tf_inputs)

# + [markdown] papermill={"duration": 0.090807, "end_time": "2022-01-24T19:39:07.146086", "exception": false, "start_time": "2022-01-24T19:39:07.055279", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.101055, "end_time": "2022-01-24T19:39:07.337045", "exception": false, "start_time": "2022-01-24T19:39:07.235990", "status": "completed"} tags=[]
pt_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

print("\n\n... PT INPUTS ...\n")
print(pt_inputs)

# + [markdown] papermill={"duration": 0.088298, "end_time": "2022-01-24T19:39:07.515698", "exception": false, "start_time": "2022-01-24T19:39:07.427400", "status": "completed"} tags=[]
# <br><b style="color: navy; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">NUMPY</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>
#
# We won't normally show the <b style="color: navy;">NumPy</b> version. However, I wanted to highlight that you can pass this option. You can also occasionally pass other options for other respective libraries (<b style="color: darkgreen;">Trax</b>, <b style="color: darkred;">Flax</b>, etc.)

# + papermill={"duration": 0.099724, "end_time": "2022-01-24T19:39:07.704449", "exception": false, "start_time": "2022-01-24T19:39:07.604725", "status": "completed"} tags=[]
np_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="np")
print("\n\n... NP INPUTS ...\n")
print(np_inputs)

# + [markdown] papermill={"duration": 0.090021, "end_time": "2022-01-24T19:39:07.885314", "exception": false, "start_time": "2022-01-24T19:39:07.795293", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">GOING THROUGH THE MODEL</b>
#
# <p style="font-family: Georgia;">We can download our pretrained model the same way we did with our tokenizer. 🤗 Transformers provides an <b>[<b style="color: #ff6f00;">TFAutoModel</b>|<b style="color: #EE4C2C;">AutoModel</b>]</b> class which also has a <b><code>from_pretrained</code></b> method:</p>
#
# <p style="font-family: Georgia;">In *[the code below, we download the same checkpoint] we used in our pipeline before (it should actually *[be]  cached already) and *[instantiate] a model with it.</p>
#
# <p style="font-family: Georgia;">This architecture contains only the base Transformer module: given some inputs, it will output what we call <b><i>hidden states</i></b>, also known as <b><i>features</i></b>. For each model input, we’ll retrieve a high-dimensional vector representing the <b>contextual understanding of that input by the Transformer model</b>.</p>
#
# <p style="font-family: Georgia;">While these hidden states can be useful on their own, they’re usually inputs to another part of the model, known as the <b><i>head</i></b>. In <b><a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-1-tf-torch">Chapter 1</a></b>, the different tasks could have been performed with the same architecture, but each of these tasks will have a different head associated with it.</p>

# + [markdown] papermill={"duration": 0.09006, "end_time": "2022-01-24T19:39:08.065993", "exception": false, "start_time": "2022-01-24T19:39:07.975933", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 10.428748, "end_time": "2022-01-24T19:39:18.563290", "exception": false, "start_time": "2022-01-24T19:39:08.134542", "status": "completed"} tags=[]
# Import Tensorflow version of AutoModel
from transformers import TFAutoModel

# Specify the model checkpoint name
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the tensorflow version of the model from the checkpoint
tf_model = TFAutoModel.from_pretrained(checkpoint)

print("\n... TF MODEL ...\n")
print(tf_model)

# + [markdown] papermill={"duration": 0.059866, "end_time": "2022-01-24T19:39:18.684959", "exception": false, "start_time": "2022-01-24T19:39:18.625093", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 1.869598, "end_time": "2022-01-24T19:39:20.614711", "exception": false, "start_time": "2022-01-24T19:39:18.745113", "status": "completed"} tags=[]
# Import PyTorch version of AutoModel
from transformers import AutoModel

# Specify the model checkpoint name
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the pytorch version of the model from the checkpoint
pt_model = AutoModel.from_pretrained(checkpoint)

print("\n... PT MODEL ...\n")
print(pt_model)

# + [markdown] papermill={"duration": 0.059913, "end_time": "2022-01-24T19:39:20.733797", "exception": false, "start_time": "2022-01-24T19:39:20.673884", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">A high-Dimensional vector?</b>
#
# <p style="font-family: Georgia;">The vector output by the Transformer module is usually large. It generally has three dimensions:</p>
#
# <ol style="font-family: Georgia;">
#     <li>
#         <b>Batch Size:</b> The number of sequences processed at a time (2 in our example).
#     </li>
#     <li>
#         <b>Sequence Length:</b> The length of the numerical representation of the sequence (16 in our example).
#     </li>
#     <li>
#         <b>Hidden Size:</b> The vector dimension of each model input.
#     </li>
# </ol>
#
# <p style="font-family: Georgia;">It is said to be <b><i>“high dimensional”</i></b> because of the last value. The hidden size can be very large (<b><code>768</code></b> is common for smaller models, and in larger models this can reach <b><code>3072</code></b> or more).</p>
#
# <p style="font-family: Georgia;">We can see this below when we feed the inputs we preprocessed to our model. Note that the outputs of 🤗 Transformers models behave like <a href="https://www.geeksforgeeks.org/namedtuple-in-python/"><b><code>namedtuples</code></b></a> or <a href="https://www.w3schools.com/python/python_dictionaries.asp"><b><code>dictionaries</code></b></a>. You can access the elements by attributes (like we did) or by key (<b><code>outputs["last_hidden_state"]</code></b>), or even by index if you know exactly where the thing you are looking for is (<b><code>outputs[0]</code></b>).</p>

# + [markdown] papermill={"duration": 0.091885, "end_time": "2022-01-24T19:39:20.918405", "exception": false, "start_time": "2022-01-24T19:39:20.826520", "status": "completed"} tags=[]
# <!-- <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b><b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;"> & </b><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;"> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b> -->
#
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.228801, "end_time": "2022-01-24T19:39:21.238876", "exception": false, "start_time": "2022-01-24T19:39:21.010075", "status": "completed"} tags=[]
# `Call` the tensorflow model
tf_outputs = tf_model(tf_inputs)

# Print the output shape
print("\n\n... TF OUTPUT SHAPE ...\n")
print(tf_outputs.last_hidden_state.shape)

# Print the raw outputs
print("\n\n\n... TF RAW OUTPUTS ...\n")
print(tf_outputs)

# + [markdown] papermill={"duration": 0.093492, "end_time": "2022-01-24T19:39:21.426411", "exception": false, "start_time": "2022-01-24T19:39:21.332919", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.209726, "end_time": "2022-01-24T19:39:21.729341", "exception": false, "start_time": "2022-01-24T19:39:21.519615", "status": "completed"} tags=[]
# `Call` the pytorch model
pt_outputs = pt_model(**pt_inputs)

# Print the output shape
print("\n\n... PT OUTPUT SHAPE ...\n")
print(pt_outputs.last_hidden_state.shape)

# Print the raw outputs
print("\n\n\n... PT RAW OUTPUTS ...\n")
print(pt_outputs)

# + [markdown] papermill={"duration": 0.093947, "end_time": "2022-01-24T19:39:21.918810", "exception": false, "start_time": "2022-01-24T19:39:21.824863", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Model heads: Making sense out of numbers</b>
#
# <p style="font-family: Georgia;">The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension. They are usually composed of one or a few linear layers.</p>
#
# <br><center><img src="https://huggingface.co/course/static/chapter2/transformer_and_head.png" width=80%></center><br>
#
# <p style="font-family: Georgia;">The output of the Transformer model is sent directly to the model head to be processed.</p>
#
# <p style="font-family: Georgia;">In this diagram, the model is represented by its embeddings layer and the subsequent layers. The embeddings layer converts each input ID in the tokenized input into a vector that represents the associated token. The subsequent layers manipulate those vectors using the attention mechanism to produce the final representation of the sentences.</p>
#
# <p style="font-family: Georgia;">There are many different architectures available in 🤗 Transformers, with each one designed around tackling a specific task. Here is a non-exhaustive list:</p>
#
# <ul style="font-family: Georgia;">
#     <li>
#         <b><code>Model</code></b> (retrieve the hidden states)
#     </li>
#     <li>
#         <b><code>ForCausalLM</code></b>
#     </li>
#     <li>
#         <b><code>ForMaskedLM</code></b>
#     </li>
#     <li>
#         <b><code>ForMultipleChoice</code></b>
#     </li>
#     <li>
#         <b><code>ForQuestionAnswering</code></b>
#     </li>
#     <li>
#         <b><code>ForSequenceClassification</code></b>
#     </li>
#     <li>
#         <b><code>ForTokenClassification</code></b>
#     </li>
#     <li>
#         and others 🤗
#     </li>
# </ul>
#
# <p style="font-family: Georgia;">For our example, we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative). So, we won’t actually use the <b>[<b style="color: #ff6f00;">TFAutoModel</b>|<b style="color: #EE4C2C;">AutoModel</b>]</b> class, but <b>[<b style="color: #ff6f00;">TFAutoModelForSequenceClassification</b>|<b style="color: #EE4C2C;">AutoModelForSequenceClassification</b>]</b></p>
#
# <p style="font-family: Georgia;">Now if we look at the shape of our inputs, the dimensionality will be much lower: the model head takes as input the high-dimensional vectors we saw before, and outputs vectors containing two values (one per label). Since we have just two sentences and two labels, the result we get from our model is of shape 2 x 2.</p>
#
#

# + [markdown] papermill={"duration": 0.094208, "end_time": "2022-01-24T19:39:22.108503", "exception": false, "start_time": "2022-01-24T19:39:22.014295", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 1.888513, "end_time": "2022-01-24T19:39:24.090755", "exception": false, "start_time": "2022-01-24T19:39:22.202242", "status": "completed"} tags=[]
# Import tensorflow version of the module required 
# to generate the sequence classification model 
from transformers import TFAutoModelForSequenceClassification

# Specify the checkpoint for which we will load the required model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the tensorflow model for sequence classification
tf_model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

# `Call` the model
tf_outputs = tf_model(tf_inputs)

print("\n\n... TF OUTPUT LOGITS' SHAPE ...\n")
print(tf_outputs.logits.shape)

print("\n\n\n... TF RAW OUTPUTS ...\n")
print(tf_outputs)

# + [markdown] papermill={"duration": 0.096689, "end_time": "2022-01-24T19:39:24.284960", "exception": false, "start_time": "2022-01-24T19:39:24.188271", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 1.94162, "end_time": "2022-01-24T19:39:26.322871", "exception": false, "start_time": "2022-01-24T19:39:24.381251", "status": "completed"} tags=[]
# Import pytorch version of the module required 
# to generate the sequence classification model 
from transformers import AutoModelForSequenceClassification

# Specify the checkpoint for which we will load the required model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the pytorch model for sequence classification
pt_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# `Call` the model
pt_outputs = pt_model(**pt_inputs)

print("\n\n... PT OUTPUT LOGITS' SHAPE ...\n")
print(pt_outputs.logits.shape)

print("\n\n\n... PT RAW OUTPUTS ...\n")
print(pt_outputs)

# + [markdown] papermill={"duration": 0.094763, "end_time": "2022-01-24T19:39:26.514294", "exception": false, "start_time": "2022-01-24T19:39:26.419531", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Post-Processing The Output</b>
#
# <p style="font-family: Georgia;">The values we get as output from our model don’t necessarily make sense by themselves. Our model predicted <b><code>[-1.5607, 1.6123]</code></b> for the first sentence and <b><code>[4.1692, -3.3464]</code></b> for the second one. <b>Those are not probabilities but logits</b>, the raw, unnormalized scores outputted by the last layer of the model. To be converted to probabilities, they need to go through a <b><i>SoftMax Layer</i></b> (all 🤗 Transformers models output the logits, as the loss function for training will generally fuse the last activation function, such as <b><i>SoftMax</i></b>, with the actual loss function, such as <b><i>Cross Entropy</i></b>).</p>
#
# <p style="font-family: Georgia;">*After we run the output through a softmax function, we will see that the model predicts <b><code>[0.0402, 0.9598]</code></b> for the first sentence and <b><code>[0.9995, 0.0005]</code></b> for the second one. These are recognizable <b><i>probability scores</i></b>.
#
# <p style="font-family: Georgia;">To get the labels corresponding to each position, we can inspect the <b><code>id2label</code></b> attribute of the model config (more on this in the next section). Using this mapping, we can conclude that the model predicted the following. After the demonstrating this below,we will  have successfully reproduced the three steps of the pipeline: preprocessing with tokenizers, passing the inputs through the model, and postprocessing!  We will now take some time to dive deeper into each of those steps.</p>
#
# <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px; ">
#     <br><b>✏️&nbsp;&nbsp;<b style="color: black;">TRY IT OUT!</b>&nbsp;&nbsp;&nbsp;&nbsp; Choose two (or more) texts of your own and run them through the sentiment-analysis pipeline. Then replicate the steps you saw here yourself and check that you obtain the same results!</b><br><br>
# </div></center>

# + [markdown] papermill={"duration": 0.094756, "end_time": "2022-01-24T19:39:26.703567", "exception": false, "start_time": "2022-01-24T19:39:26.608811", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.119155, "end_time": "2022-01-24T19:39:26.918317", "exception": false, "start_time": "2022-01-24T19:39:26.799162", "status": "completed"} tags=[]
# Import Tensorflow
import tensorflow as tf

# Calculate the softmax of the output logits
tf_predictions = tf.math.softmax(tf_outputs.logits, axis=-1)

# Print the label map
print("\n\n\n... MODEL LABEL MAP ...\n")
print(tf_model.config.id2label)

# Print the prediction probabilities
print("\n\n... TF OUTPUT PROBABILITIES ...\n")
print(tf_predictions)

# Print the label map
print("\n\n... TF OUTPUT PROBABILITIES WITH LABEL TITLES ...\n")
for i, pred in enumerate(tf_predictions): print(f"Sentence #{i+1} - Classification Probabilities:\n\t{tf_model.config.id2label[0]}: %{100*pred[0]:.3f}\n\t{tf_model.config.id2label[1]}: %{100*pred[1]:.3f}\n")


# + papermill={"duration": 0.244214, "end_time": "2022-01-24T19:39:27.258350", "exception": false, "start_time": "2022-01-24T19:39:27.014136", "status": "completed"} tags=[]
# AutoModelForSequenceClassification
def tf_sentiment_classification(model, tokenizer, sentences, return_lbl_map=True):
    """ Function to perform sentiment analysis with pipeline in Tensorflow 
    
    Args:
        model (TFAutoModelForSequenceClassification): Pretrained model to use 
            for sentiment classification.
        tokenizer (PreTrainedTokenizerFast): Tokenizer (agnostic to TF or PT)
        sentences (list of strs): Sentences to perform sentiment classification on
        return_lbl_map (bool, optional): Dictionary mapping id to string
        
    Returns:
        Tensor containing probabilities of each particular sentiment 
        for the respective sentences. Optionally return the model label map.
    """
    
    # Create the pipeline
    #     1. Preprocessing (tokenizers)
    #     2. Model Inference
    #     3. Postprocessing (softmax)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="tf")
    raw_outputs = model(inputs)
    predictions = tf.math.softmax(raw_outputs.logits, axis=-1)
    
    if return_lbl_map:
        return predictions, model.config.id2label
    else:
        return predictions
    
ex_sentences = ["I really hate this song. You can't dance to it!",
                "You're definitely not a bad guy.",
                "The capital of California is Sacremento.",
                "The capital of Canada is Ottawa."]
tf_ex_preds, ex_lbl_map = tf_sentiment_classification(tf_model, tokenizer, ex_sentences)
for i, tf_pred in enumerate(tf_ex_preds): 
    print(f"Sentence #{i+1} - '{ex_sentences[i]}'\n\t" \
          f"--> {ex_lbl_map[0]}: %{100*tf_pred[0]:.3f}\n\t" \
          f"--> {ex_lbl_map[1]}: %{100*tf_pred[1]:.3f}\n")

# + [markdown] papermill={"duration": 0.099786, "end_time": "2022-01-24T19:39:27.454065", "exception": false, "start_time": "2022-01-24T19:39:27.354279", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.111201, "end_time": "2022-01-24T19:39:27.663403", "exception": false, "start_time": "2022-01-24T19:39:27.552202", "status": "completed"} tags=[]
# Import PyTorch
import torch

# Calculate the softmax of the output logits
pt_predictions = torch.nn.functional.softmax(pt_outputs.logits, dim=-1)

# Print the label map
print("\n\n\n... MODEL LABEL MAP ...\n")
print(pt_model.config.id2label)

# Print the prediction probabilities
print("\n\n... PT OUTPUT PROBABILITIES ...\n")
print(pt_predictions)

# Print the label map
print("\n\n... PT OUTPUT PROBABILITIES WITH LABEL TITLES ...\n")
for i, pred in enumerate(pt_predictions): print(f"Sentence #{i+1} - Classification Probabilities:\n\t{pt_model.config.id2label[0]}: %{100*pred[0]:.3f}\n\t{pt_model.config.id2label[1]}: %{100*pred[1]:.3f}\n")


# + papermill={"duration": 0.246894, "end_time": "2022-01-24T19:39:28.008254", "exception": false, "start_time": "2022-01-24T19:39:27.761360", "status": "completed"} tags=[]
def pt_sentiment_classification(model, tokenizer, sentences, return_lbl_map=True):
    """ Function to perform sentiment analysis with pipeline in PyTorch 
    
    Args:
        model (AutoModelForSequenceClassification): Pretrained model to use 
            for sentiment classification.
        tokenizer (PreTrainedTokenizerFast): Tokenizer (agnostic to TF or PT)
        sentences (list of strs): Sentences to perform sentiment classification on
        return_lbl_map (bool, optional): Dictionary mapping id to string
        
    Returns:
        Tensor containing probabilities of each particular sentiment 
        for the respective sentences. Optionally return the model label map.
    """
    
    # Create the pipeline
    #     1. Preprocessing (tokenizers)
    #     2. Model Inference
    #     3. Postprocessing (softmax)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    raw_outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(raw_outputs.logits, dim=-1)
    if return_lbl_map:
        return predictions, model.config.id2label
    else:
        return predictions
    
pt_ex_preds, ex_lbl_map = pt_sentiment_classification(pt_model, tokenizer, ex_sentences)
for i, pt_pred in enumerate(pt_ex_preds): 
    print(f"Sentence #{i+1} - '{ex_sentences[i]}'\n\t" \
          f"--> {ex_lbl_map[0]}: %{100*pt_pred[0]:.3f}\n\t" \
          f"--> {ex_lbl_map[1]}: %{100*pt_pred[1]:.3f}\n")

# + [markdown] papermill={"duration": 0.096619, "end_time": "2022-01-24T19:39:28.203017", "exception": false, "start_time": "2022-01-24T19:39:28.106398", "status": "completed"} tags=[]
# <a id="2_2"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.2 MODELS&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# - <a href="https://www.youtube.com/watch?v=d3JVgghSOew" style="font-family: Georgia; color: #ff6f00; font-weight: bold;">[TENSORFLOW] &nbsp;&nbsp; VIDEO LINK - INSTANTIATE A TRANSFORMER MODEL - HUGGING FACE CHANNEL</a>
# - <a href="https://www.youtube.com/watch?v=AhChOFRegn4" style="font-family: Georgia; color: #EE4C2C; font-weight: bold;">[PYTORCH] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; VIDEO LINK - INSTANTIATE A TRANSFORMER MODEL - HUGGING FACE CHANNEL</a><br>
#
# <p style="font-family: Georgia;">In this section we’ll take a closer look at creating and using a model. We’ll use the <b>[<b style="color: #ff6f00;">TFAutoModel</b>|<b style="color: #EE4C2C;">AutoModel</b>]</b> class, which is handy when you want to instantiate any model from a checkpoint.</p>
#
# <p style="font-family: Georgia;">The <b>[<b style="color: #ff6f00;">TFAutoModel</b>|<b style="color: #EE4C2C;">AutoModel</b>]</b> class and all of its relatives are actually simple wrappers over the wide variety of models available in the library. It’s a clever wrapper as it can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture.</p>
#
# <p style="font-family: Georgia;">However, if you know the type of model you want to use, you can use the class that defines its architecture directly. Let’s take a look at how this works with a <b><a href="https://arxiv.org/abs/1810.04805">BERT</a></b> model.</p><br>
#
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Creating a Transformer</b>
#
# <p style="font-family: Georgia;">The first thing we’ll need to do to initialize a BERT model is load a configuration object. The configuration contains many attributes that are used to build the model. While you haven’t seen what all of these attributes do yet, you should recognize some of them.</p>
#
# <ul style="font-family: Georgia;">
#     <li>The <b><code>hidden_size</code></b> attribute defines the size of the hidden_states</code></b> vector</li>
#     <li>The <b><code>num_hidden_layers</code></b> defines the number of layers the Transformer model has.</li>
# </ul>

# + [markdown] papermill={"duration": 0.097907, "end_time": "2022-01-24T19:39:28.398238", "exception": false, "start_time": "2022-01-24T19:39:28.300331", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.660074, "end_time": "2022-01-24T19:39:29.156984", "exception": false, "start_time": "2022-01-24T19:39:28.496910", "status": "completed"} tags=[]
# Imports
from transformers import BertConfig, TFBertModel

# Building the config
config = BertConfig()

# Building the model from the config
tf_model = TFBertModel(config)

print("\n\n\n... TF BERT MODEL ...\n")
print(tf_model)

print("\n\n... BERT CONFIG ...\n")
print(config)

# + [markdown] papermill={"duration": 0.105073, "end_time": "2022-01-24T19:39:29.325801", "exception": false, "start_time": "2022-01-24T19:39:29.220728", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 4.251852, "end_time": "2022-01-24T19:39:33.739858", "exception": false, "start_time": "2022-01-24T19:39:29.488006", "status": "completed"} tags=[]
# Imports
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the pt model from the config
pt_model = BertModel(config)

print("\n\n\n... PT BERT MODEL ...\n")
print(pt_model)

print("\n\n... BERT CONFIG ...\n")
print(config)

# + [markdown] papermill={"duration": 0.099119, "end_time": "2022-01-24T19:39:33.939704", "exception": false, "start_time": "2022-01-24T19:39:33.840585", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Different Loading Methods</b>
#
# <p style="font-family: Georgia;">Creating a model from the default configuration initializes it with random values.</p>
#
# <p style="font-family: Georgia;">The model can be used in this state, but it will output gibberish; it needs to be trained first. We could train the model from scratch on the task at hand, but as you saw in <b><a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-1-tf-torch">Chapter 1</a></b>, this would require a long time and a lot of data, and it would have a non-negligible environmental impact. To avoid unnecessary and duplicated effort, it’s imperative to be able to share and reuse models that have already been trained.</p>
#
# <p style="font-family: Georgia;">Loading a Transformer model that is already trained is simple — we can do this using the <b><code>from_pretrained()</code></b> method:</p>
#
# <p style="font-family: Georgia;">As you saw earlier, we could replace <b>[<b style="color: #ff6f00;">TFBertModel</b>|<b style="color: #EE4C2C;">BertModel</b>]</b> with the equivalent <b>[<b style="color: #ff6f00;">TFAutoModel</b>|<b style="color: #EE4C2C;">AutoModel</b>]</b> class. We’ll do this from now on as this produces checkpoint-agnostic code; if your code works for one checkpoint, it should work seamlessly with another. This applies even if the architecture is different, as long as the checkpoint was trained for a similar task (for example, a sentiment analysis task).</p>
#
# <p style="font-family: Georgia;">In the code sample *below we don’t use <b><code>BertConfig</code></b>, and instead loaded a pretrained model via the <b><code>bert-base-cased</code></b> identifier. This is a model checkpoint that was trained by the authors of <b><a href="https://arxiv.org/abs/1810.04805">BERT</a></b> themselves; you can find more details about it in its <b><a href="https://huggingface.co/bert-base-cased">model card</a></b>.</p>
#
# <p style="font-family: Georgia;">After the model is initialized with the weights from the checkpoint, it can be used directly for inference on the tasks it was trained on or it can be fine-tuned on a new task. By training with pretrained weights rather than from scratch, we can quickly achieve good results.</p>
#
# <p style="font-family: Georgia;">The weights have been downloaded and cached (so future calls to the <b><code>from_pretrained()</code></b> method won’t re-download them) in the cache folder, which defaults to <code><i>~/.cache/huggingface/transformers</i></code>. You can customize your cache folder by setting the <b><code>HF_HOME</code></b> environment variable.</p>
#
# <p style="font-family: Georgia;">The identifier used to load the model can be the identifier of any model on the <b><a href="https://huggingface.co/models">Model Hub</a></b>, as long as it is compatible with the <b><a href="https://arxiv.org/abs/1810.04805">BERT</a></b> architecture. The entire list of available <b><a href="https://arxiv.org/abs/1810.04805">BERT</a></b> checkpoints can be found <b><a href="https://huggingface.co/models?filter=bert">here</a></b>.</p>

# + [markdown] papermill={"duration": 0.098852, "end_time": "2022-01-24T19:39:34.138025", "exception": false, "start_time": "2022-01-24T19:39:34.039173", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 18.426031, "end_time": "2022-01-24T19:39:52.663377", "exception": false, "start_time": "2022-01-24T19:39:34.237346", "status": "completed"} tags=[]
from transformers import BertConfig, TFBertModel

# Model is randomly initialized!
config = BertConfig()
tf_model = TFBertModel(config)

# Checkpoint is downloaded and model loads from it (no config needed)
tf_model = TFBertModel.from_pretrained("bert-base-cased")

print("\n\n\n... CONENTS OF `~/.cache/huggingface/transformers` DIRECTORY ...\n")
# !ls -sh ~/.cache/huggingface/transformers

# + [markdown] papermill={"duration": 0.101861, "end_time": "2022-01-24T19:39:52.867936", "exception": false, "start_time": "2022-01-24T19:39:52.766075", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 23.846553, "end_time": "2022-01-24T19:40:16.817122", "exception": false, "start_time": "2022-01-24T19:39:52.970569", "status": "completed"} tags=[]
from transformers import BertConfig, BertModel

# Model is randomly initialized!
config = BertConfig()
pt_model = BertModel(config)

# Checkpoint is downloaded and model loads from it (no config needed)
pt_model = BertModel.from_pretrained("bert-base-cased")

print("\n\n\n... CONENTS OF `~/.cache/huggingface/transformers` DIRECTORY ...\n")
# !ls -sh ~/.cache/huggingface/transformers

# + [markdown] papermill={"duration": 0.067262, "end_time": "2022-01-24T19:40:16.953419", "exception": false, "start_time": "2022-01-24T19:40:16.886157", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Saving Methods</b>
#
# <p style="font-family: Georgia;">Saving a model is as easy as loading one — we use the <b><code>save_pretrained()</code></b> method, which is analogous to the <b><code>from_pretrained()</code></b> method. This method will save two files to your disk, <code><b>config.json</b></code> and <b>[<b style="color: #ff6f00;">tf_model.h5</b>|<b style="color: #EE4C2C;">pytorch_model.bin</b>]</b>. These two files go hand in hand; the configuration is necessary to know your model’s architecture, while the model weights are your model’s parameters.</p>
#
# <ul style="font-family: Georgia;">
#     <li><b><code>config.json</code></b><ul>
#         <li>This file contains the attributes necessary to build the model architecture.</li> 
#         <li>This file also contains some metadata, such as where the checkpoint originated and what 🤗 Transformers version you were using when you last saved the checkpoint.</li></ul>
#     <li><b>[<b style="color: #ff6f00;">tf_model.h5</b>|<b style="color: #EE4C2C;">pytorch_model.bin</b>]</b><ul>
#         <li>This file is the <b><i>state dictionary</i></b>; it contains all your model’s weights. </li>
# </ul>
#
#

# + [markdown] papermill={"duration": 0.067003, "end_time": "2022-01-24T19:40:17.089288", "exception": false, "start_time": "2022-01-24T19:40:17.022285", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b><b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;"> &nbsp;&&nbsp; </b><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 2.139522, "end_time": "2022-01-24T19:40:19.296338", "exception": false, "start_time": "2022-01-24T19:40:17.156816", "status": "completed"} tags=[]
import os

###############################################
#      Tensorflow Code (same as pytorch)      #
###############################################
os.makedirs("/kaggle/working/tf_models", exist_ok=True)
tf_model.save_pretrained("/kaggle/working/tf_models")
print("\n\n\n... CONTENTS OF `/kaggle/working/tf_models` ...\n")
# !ls -sh /kaggle/working/tf_models
###############################################


###############################################
#      PyTorch Code (same as tensorflow)      #
###############################################
os.makedirs("/kaggle/working/pt_models", exist_ok=True)
pt_model.save_pretrained("/kaggle/working/pt_models")
print("\n\n\n... CONTENTS OF `/kaggle/working/pt_models` ...\n")
# !ls -sh /kaggle/working/pt_models
###############################################

# + [markdown] papermill={"duration": 0.068506, "end_time": "2022-01-24T19:40:19.467106", "exception": false, "start_time": "2022-01-24T19:40:19.398600", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">USING A TRANSFORMER MODEL FOR INFERENCE</b>
#
# <p style="font-family: Georgia;">Now that you know how to load and save a model, let’s try using it to make some predictions. Transformer models can only process numbers — numbers that the tokenizer generates. But before we discuss tokenizers, let’s explore what inputs the model accepts.</p>
#
# <p style="font-family: Georgia;">Tokenizers can take care of casting the inputs to the appropriate framework’s tensors, but to help you understand what’s going on, we’ll take a quick look at what must be done before sending the inputs to the model.</p>
#
# <p style="font-family: Georgia;">Let’s say we have a couple of sequences. The tokenizer converts these to vocabulary indices which are typically called <b><i>input IDs.</i></b> Each sequence will now be a list of numbers! The resulting output is a list of encoded sequences: a list of lists. Tensors only accept rectangular shapes (think matrices). This “array” is already of rectangular shape, so converting it to a tensor is easy.</p>

# + [markdown] papermill={"duration": 0.07782, "end_time": "2022-01-24T19:40:19.614360", "exception": false, "start_time": "2022-01-24T19:40:19.536540", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b><b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;"> &nbsp;&&nbsp; </b><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.092065, "end_time": "2022-01-24T19:40:19.795869", "exception": false, "start_time": "2022-01-24T19:40:19.703804", "status": "completed"} tags=[]
# Define the initial sequences (raw)
sequences = ["Hello!", "Cool.", "Nice!"]
print(f"\n\n\n... SEQUENCES --> {sequences}")
for seq in sequences: print(f"\t--> {seq}")

# Tokenize the sequences (encoded)
encoded_sequences = tokenizer(sequences)["input_ids"]
print(f"\n\n\n... ENCODED SEQUENCES --> {encoded_sequences}")
for enc_seq in encoded_sequences: print(f"\t--> {enc_seq}")

###############################################
#               Tensorflow Code               #
###############################################
tf_model_inputs = tf.constant(encoded_sequences)
print("\n\n\n... TENSORFLOW MODEL INPUTS - TENSORS ...\n")
print(tf_model_inputs)
###############################################
    
###############################################
#                 PyTorch Code                #
###############################################
pt_model_inputs = torch.tensor(encoded_sequences)
print("\n\n\n... PYTORCH MODEL INPUTS - TENSORS ...\n")
print(pt_model_inputs)
###############################################

# + [markdown] papermill={"duration": 0.085253, "end_time": "2022-01-24T19:40:19.952674", "exception": false, "start_time": "2022-01-24T19:40:19.867421", "status": "completed"} tags=[]
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Using the tensors as inputs to the model</b>
#
# <p style="font-family: Georgia;">Making use of the tensors with the model is extremely simple — we just call the model with the inputs.</p> 
#
# <p style="font-family: Georgia;">While the model accepts a lot of different arguments, only the <b><i>input IDs</i></b> are necessary. We’ll explain what the other arguments do and when they are required later, but first we need to take a closer look at the tokenizers that build the inputs that a Transformer model can understand.</p>

# + [markdown] papermill={"duration": 0.105613, "end_time": "2022-01-24T19:40:20.158106", "exception": false, "start_time": "2022-01-24T19:40:20.052493", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b><b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;"> &nbsp;&&nbsp; </b><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.39839, "end_time": "2022-01-24T19:40:20.662795", "exception": false, "start_time": "2022-01-24T19:40:20.264405", "status": "completed"} tags=[]
###############################################
#               Tensorflow Code               #
###############################################
tf_output = tf_model(tf_model_inputs)
print("\n\n\n... TENSORFLOW MODEL OUTPUTS ...\n")
print(tf_output)
###############################################

###############################################
#                 PyTorch Code                #
###############################################
pt_output = pt_model(pt_model_inputs)
print("\n\n\n... PYTORCH MODEL OUTPUTS ...\n")
print(pt_output)
###############################################

# + [markdown] papermill={"duration": 0.107074, "end_time": "2022-01-24T19:40:20.876397", "exception": false, "start_time": "2022-01-24T19:40:20.769323", "status": "completed"} tags=[]
# <a id="2_3"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.3 TOKENIZER BACKGROUND&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# - <a href="https://www.youtube.com/watch?v=VFp38yj8h3A" style="font-family: Georgia; color: darkred; font-weight: bold;">VIDEO LINK - TOKENIZERS OVERVIEW - HUGGING FACE CHANNEL</a><br>
#
# <p style="font-family: Georgia;">Tokenizers are one of the core components of the NLP pipeline. They serve one purpose: to translate text into data that can be processed by the model. <b><i>Models can only process numbers, so tokenizers need to convert our text inputs to numerical data</i></b>. In this section, we’ll explore exactly what happens in the tokenization pipeline.</p>
#
# <p style="font-family: Georgia;">In NLP tasks, the data that is generally processed is raw text. Here’s an example of such text:</p>
#
# <pre style="font-weight: bold; white-space: pre-wrap; background-color: #eee; border: 1px dashed #999; display: block; padding: 16px; line-height:160%">
#     Jim Henson was a puppeteer
# </pre>
#     
# <p style="font-family: Georgia;">However, models can only process numbers, so we need to find a way to convert the raw text to numbers. That’s what the tokenizers do, and there are a lot of ways to go about this. The goal is to find the most meaningful representation — that is, the one that makes the most sense to the model — and, if possible, the smallest representation.</p>
#
# <p style="font-family: Georgia;">Let’s take a look at some examples of tokenization algorithms, and try to answer some of the questions you may have about tokenization.</p><br>
#
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">WORD BASED TOKENIZERS</b>
#
# - <a href="https://www.youtube.com/watch?v=nhJxYji1aho" style="font-family: Georgia; color: darkred; font-weight: bold;">VIDEO LINK - WORD BASED TOKENIZERS - HUGGING FACE CHANNEL</a><br>
#
# <p style="font-family: Georgia;">The first type of tokenizer that comes to mind is word-based. It’s generally very easy to set up and use with only a few rules, and it often yields decent results. For example, in the image below, the goal is to split the raw text into words and find a numerical representation for each of them:</p> 
#
# <br><center><img src="https://huggingface.co/course/static/chapter2/word_based_tokenization.png" width=95%></center><br>
#
# <p style="font-family: Georgia;">There are different ways to split the text. For example, we could could use whitespace to tokenize the text into words by applying Python’s <b><code>split()</code></b> function</p>
#
# <pre style="font-weight: bold; white-space: pre-wrap; background-color: #eee; border: 1px dashed #999; display: block; padding: 16px; line-height:160%">
#     tokenized_text = "Jim Henson was a puppeteer".split()
#     print(tokenized_text)
# </pre>
#
# <pre style="font-weight: bold; white-space: pre-wrap; background-color: #eee; border: 1px dashed #999; display: block; padding: 16px; line-height:160%">
#     ['Jim', 'Henson', 'was', 'a', 'puppeteer']
# </pre><br>
#
# <p style="font-family: Georgia;">There are also variations of word tokenizers that have extra rules for punctuation. With this kind of tokenizer, we can end up with some pretty large <b><i>“vocabularies,”</i></b> where a  <b><i>vocabulary</i></b> is defined by the total number of independent tokens that we have in our corpus.</p>
#
# <p style="font-family: Georgia;">Each word gets assigned an ID, starting from 0 and going up to the size of the vocabulary. The model uses these IDs to identify each word.</p>
#
# <p style="font-family: Georgia;">If we want to completely cover a language with a word-based tokenizer, we’ll need to have an identifier for each word in the language, which will generate a huge amount of tokens. For example, there are over 500,000 words in the English language, so to build a map from each word to an input ID we’d need to keep track of that many IDs. Furthermore, words like “dog” are represented differently from words like “dogs”, and the model will initially have no way of knowing that “dog” and “dogs” are similar: it will identify the two words as unrelated. The same applies to other similar words, like “run” and “running”, which the model will not see as being similar initially.</p>
#
# <p style="font-family: Georgia;">Finally, we need a custom token to represent words that are not in our vocabulary. This is known as the “unknown” token, often represented as ”[UNK]” or ””. It’s generally a bad sign if you see that the tokenizer is producing a lot of these tokens, as it wasn’t able to retrieve a sensible representation of a word and you’re losing information along the way. The goal when crafting the vocabulary is to do it in such a way that the tokenizer tokenizes as few words as possible into the unknown token.</p>
#
# <p style="font-family: Georgia;">One way to reduce the amount of unknown tokens is to go one level deeper, using a <b><i>character-based tokenizer</i></b>.</p><br>
#
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">CHARACTER-BASED TOKENIZERS</b>
#
# - <a href="https://www.youtube.com/watch?v=ssLq_EK2jLE" style="font-family: Georgia; color: darkred; font-weight: bold;">VIDEO LINK - CHARACTER-BASED TOKENIZERS - HUGGING FACE CHANNEL</a><br>
#
# <p style="font-family: Georgia;">Character-based tokenizers split the text into characters, rather than words. This has two primary benefits:</p>
#
# <ul style="font-family: Georgia;">
#     <li>The vocabulary is much smaller.</li>
#     <li>There are much fewer out-of-vocabulary (unknown) tokens, since every word can be built from characters.</li>
# </ul>
#
# <p style="font-family: Georgia;">But here too some questions arise concerning spaces and punctuation:</p> 
#
# <br><center><img src="https://huggingface.co/course/static/chapter2/character_based_tokenization.png" width=95%></center><br>
#
# <p style="font-family: Georgia;">This approach isn’t perfect either. Since the representation is now based on characters rather than words, one could argue that, intuitively, it’s less meaningful: each character doesn’t mean a lot on its own, whereas that is the case with words. However, this again differs according to the language; in Chinese, for example, each character carries more information than a character in a Latin language.</p>
#
# <p style="font-family: Georgia;">Another thing to consider is that we’ll end up with a very large amount of tokens to be processed by our model: whereas a word would only be a single token with a word-based tokenizer, it can easily turn into 10 or more tokens when converted into characters.</p>
#
# <p style="font-family: Georgia;">To get the best of both worlds, we can use a third technique that combines the two approaches: <b><i>subword tokenization</i></b>.</p><br>
#
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">SUBWORD-BASED TOKENIZERS</b>
#
# - <a href="https://www.youtube.com/watch?v=zHvTiHr506c" style="font-family: Georgia; color: darkred; font-weight: bold;">VIDEO LINK - SUBWORD-BASED TOKENIZERS - HUGGING FACE CHANNEL</a><br>
#
# <p style="font-family: Georgia;">Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.</p>
#
# <p style="font-family: Georgia;">For instance, “annoyingly” might be considered a rare word and could be decomposed into “annoying” and “ly”. These are both likely to appear more frequently as standalone subwords, while at the same time the meaning of “annoyingly” is kept by the composite meaning of “annoying” and “ly”.</p>
#
# <p style="font-family: Georgia;">Here is an example showing how a subword tokenization algorithm would tokenize the sequence “Let’s do tokenization!“:</p>
#
# <br><center><img src="https://huggingface.co/course/static/chapter2/bpe_subword.png" width=95%></center><br>
#
# <p style="font-family: Georgia;">These subwords end up providing a lot of semantic meaning: for instance, in the example above “tokenization” was split into “token” and “ization”, two tokens that have a semantic meaning while being space-efficient (only two tokens are needed to represent a long word). This allows us to have relatively good coverage with small vocabularies, and close to no unknown tokens.</p>
#
# <p style="font-family: Georgia;">This approach is especially useful in agglutinative languages such as Turkish, where you can form (almost) arbitrarily long complex words by stringing together subwords.</p><br>
#
# <b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">OTHER TOKENIZER TYPES</b>
#
# <p style="font-family: Georgia;">Unsurprisingly, there are many more techniques out there. To name a few:</p>
#
# <ul style="font-family: Georgia;">
#     <li><b>Byte-level BPE</b>, as used in <b><a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">GPT-2</a></b></li>
#     <li><b>WordPiece</b>, as used in <b><a href="https://arxiv.org/abs/1810.04805">BERT</a></b></li>
#     <li><b>SentencePiece</b> or <b>Unigram</b>, as used in several multilingual models</li>
# </ul>
#
# <p style="font-family: Georgia;">You should now have sufficient knowledge of how tokenizers work to get started with the API.</p>

# + papermill={"duration": 0.126199, "end_time": "2022-01-24T19:40:21.109822", "exception": false, "start_time": "2022-01-24T19:40:20.983623", "status": "completed"} tags=[]
# Define sentences to test on
text_1 = "Jim Henson was a puppeteer"
text_2 = "Let's do tokenization!"

print(f"\n\n\n\n... ORIGINAL TEXT 1\n\t--> '{text_1}'\n")
print(f"... ORIGINAL TEXT 2\n\t--> '{text_2}'\n")

wb_tokens_1 = text_1.split()
wb_tokens_2 = text_2.split()

print(f"\n\n... WORD-BASED TOKENS FROM TEXT 1\n\t--> {wb_tokens_1}\n")
print(f"... WORD-BASED TOKENS FROM TEXT 2\n\t--> {wb_tokens_2}\n")

cb_tokens_1 = [x for x in text_1 if x is not " "]
cb_tokens_2 = [x for x in text_2 if x is not " "]

print(f"\n\n... CHARACTER-BASED TOKENS FROM TEXT 1\n\t--> {cb_tokens_1}\n")
print(f"... CHARACTER-BASED TOKENS FROM TEXT 2\n\t--> {cb_tokens_2}\n")

sw_tokens_1 = tokenizer.tokenize(text_1)
sw_tokens_2 = tokenizer.tokenize(text_2)

print(f"\n\n... SUBWORD-BASED TOKENS FROM TEXT 1\n\t--> {sw_tokens_1}\n")
print(f"... SUBWORD-BASED TOKENS FROM TEXT 2\n\t--> {sw_tokens_2}\n")


# + [markdown] papermill={"duration": 0.107094, "end_time": "2022-01-24T19:40:21.323275", "exception": false, "start_time": "2022-01-24T19:40:21.216181", "status": "completed"} tags=[]
# <a id="2_4"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.4 TOKENIZER BASICS&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Tokenizer Loading and saving</b>
#
# <p style="font-family: Georgia;">Loading and saving tokenizers is as simple as it is with models. Actually, it’s based on the same two methods: <code><b>from_pretrained()</b></code> and <code><b>save_pretrained()</b></code>. These methods will load or save the algorithm used by the tokenizer (a bit like the <b><i>architecture</i></b> of the model) as well as its vocabulary (a bit like the <b><i>weights</i></b> of the model).</p>
# <ul style="font-family: Georgia;">
#     <li>Loading the <b><a href="https://arxiv.org/abs/1810.04805">BERT</a></b> tokenizer trained with the same checkpoint as <b><a href="https://arxiv.org/abs/1810.04805">BERT</a></b> is done the same way as loading the model, except we use the <b><code>BertTokenizer</code></b> class.</li>
#     <li>Similar to <b>[<b style="color: #ff6f00;">TFAutoModel</b>|<b style="color: #EE4C2C;">AutoModel</b>]</b>, the <b><code>AutoTokenizer</code></b> class will grab the proper tokenizer class in the library based on the checkpoint name, and can be used directly with any checkpoint.</li>
#     <li>Using the tokenizer is as simple as what was shown in the previous section</li>
#     <li>Saving a tokenizer is identical to saving a model</li>
# </ul>
#
# <p style="font-family: Georgia;">We’ll talk more about <b><code>token_type_ids</code></b> in <b><a href="">Chapter 3</a></b>, and we’ll explain the <b><code>attention_mask</code></b> key a little later. First, let’s see how the <b><code>input_ids</code></b> are generated. To do this, we’ll need to look at the intermediate methods of the tokenizer.</p>
#

# + papermill={"duration": 3.544822, "end_time": "2022-01-24T19:40:24.975190", "exception": false, "start_time": "2022-01-24T19:40:21.430368", "status": "completed"} tags=[]
from transformers import BertTokenizer
from transformers import AutoTokenizer

# Option 1
tokenizer_1 = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer_1.save_pretrained("/kaggle/working/tokenizer_1")
print("\n\n\n... OPTION 1 - BertTokenizer ...\n")
print(tokenizer_1("Using a Transformer network is simple"))
print("\n\n... AFTER SAVING - FILES LOCATED IN --> `/kaggle/working/tokenizer_1` ...\n")
# !ls -sh /kaggle/working/tokenizer_1

# Option 2
tokenizer_2 = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer_2.save_pretrained("/kaggle/working/tokenizer_2")
print("\n\n\n... OPTION 2 - AutoTokenizer ...\n")
print(tokenizer_2("Using a Transformer network is simple"))
print("\n\n... AFTER SAVING - FILES LOCATED IN --> `/kaggle/working/tokenizer_2` ...\n")
# !ls -sh /kaggle/working/tokenizer_2

# + [markdown] papermill={"duration": 0.124064, "end_time": "2022-01-24T19:40:25.212728", "exception": false, "start_time": "2022-01-24T19:40:25.088664", "status": "completed"} tags=[]
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Encoding</b>
#
# - <a href="https://youtu.be/Yffk5aydLzg" style="font-family: Georgia; color: darkred; font-weight: bold;">VIDEO LINK - THE TOKENIZATION PIPELINE - HUGGING FACE CHANNEL</a><br>
#
# <p style="font-family: Georgia;">Translating text to numbers is known as <b><i>encoding</i></b>. Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.</p>
#
# <p style="font-family: Georgia;">As we’ve seen, the first step is to split the text into words (or parts of words, punctuation symbols, etc.), usually called <b><i>tokens</i></b>. There are multiple rules that can govern that process, which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules that were used when the model was pretrained.</p>
#
# <p style="font-family: Georgia;">The second step is to convert those tokens into numbers, so we can build a tensor out of them and feed them to the model. To do this, the tokenizer has a <b><i>vocabulary</i></b>, which is the part we download when we instantiate it with the <b><code>from_pretrained()</code></b> method. Again, we need to use the same <b><i>vocabulary</i></b> used when the model was pretrained.</p>
#
# <p style="font-family: Georgia;">To get a better understanding of the two steps, we’ll explore them separately. Note that we will use some methods that perform parts of the tokenization pipeline separately to show you the intermediate results of those steps, but in practice, you should call the tokenizer directly on your inputs (as shown in the <b><a href="#2_1">Section 2.1 - Behind The Pipeline</a></b>).</p><br>
#
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Tokenization</b>
#
# <p style="font-family: Georgia;">The tokenization process is done by invoking the <b><code>tokenize()</code></b> method of the tokenizer. This will yield an output that is a list of strings, or tokens.
# </p>
#

# + papermill={"duration": 1.309962, "end_time": "2022-01-24T19:40:26.637813", "exception": false, "start_time": "2022-01-24T19:40:25.327851", "status": "completed"} tags=[]
from transformers import AutoTokenizer
# This tokenizer is a subword tokenizer: 
#      - it splits the words until it obtains tokens that can be represented by its vocabulary. 
#      - That’s the case here with 'transformer', which is split into two tokens: 
#           --> transform
#           --> ##er
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(f"\n\n\n... SEQUENCE\n\t--> '{sequence}'\n")
print(f"... TOKENS (SUBWORD)\n\t--> {tokens}\n")

# + [markdown] papermill={"duration": 0.111645, "end_time": "2022-01-24T19:40:26.861460", "exception": false, "start_time": "2022-01-24T19:40:26.749815", "status": "completed"} tags=[]
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">From tokens to input IDs</b>
#
# <p style="font-family: Georgia;">The conversion from tokens to input IDs is handled by the <code><b>convert_tokens_to_ids()</b></code> tokenizer method. This will yield outputs that, once converted to the appropriate framework tensor, can then be used as inputs to a model as seen earlier in this chapter.</p>
#
# <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px; ">
#     <br><b>✏️&nbsp;&nbsp;<b style="color: black;">TRY IT OUT!</b>&nbsp;&nbsp;&nbsp;&nbsp;<br><br>Replicate the two last steps (tokenization and conversion to input IDs) on the input sentences we used in <a href="#2_4"><b>Section 2.4</b></a> (“I’ve been waiting for a HuggingFace course my whole life.” and “I hate this so much!”). Check that you get the same input IDs we got earlier!</b><br><br>
# </div></center>
#

# + papermill={"duration": 1.347234, "end_time": "2022-01-24T19:40:28.321846", "exception": false, "start_time": "2022-01-24T19:40:26.974612", "status": "completed"} tags=[]
print(f"\n\n{'-'*85}\n\t\t\tCONTINUING EXAMPLE FROM PREVIOUS CELL\n{'-'*85}")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"\n\n... SEQUENCE\n\t--> '{sequence}'\n")
print(f"... TOKENS (SUBWORD)\n\t--> {tokens}\n")
print(f"... INPUT IDS\n\t--> {ids}\n")

print(f"\n\n{'-'*80}\n\t\t\tEXAMPLES FROM PREVIOUS SECTION\n{'-'*80}")
prev_seq_1 = "I’ve been waiting for a HuggingFace course my whole life."
prev_gen_tokens_1 = [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,  2026,  2878,  2166,  1012,   102]
prev_seq_2 = "I hate this so much!"
prev_gen_tokens_2 = [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
for i, (seq, pg_seq_ids) in enumerate(zip([prev_seq_1, prev_seq_2], [prev_gen_tokens_1, prev_gen_tokens_2])):
    seq_tokens = tokenizer.tokenize(seq)
    seq_ids = tokenizer.convert_tokens_to_ids(seq_tokens)
    print(f"\n\n... SEQUENCE\n\t--> '{seq}'\n")
    print(f"... TOKENS (SUBWORD)\n\t--> {seq}\n")
    print(f"... INPUT IDS (previously generated)\n\t--> {pg_seq_ids}\n")
    print(f"... INPUT IDS (newly generated)\n\t--> {seq_ids}\n")

# + [markdown] papermill={"duration": 0.113107, "end_time": "2022-01-24T19:40:28.555960", "exception": false, "start_time": "2022-01-24T19:40:28.442853", "status": "completed"} tags=[]
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">DECODING</b>
#
# <p style="font-family: Georgia;"><b><i>Decoding</i></b> is going the other way around: given vocabulary indices, we want to retrieve the representative string. This can be done with the <b><code>decode()</code></b> method of our tokenizer object.</p>
#
# <p style="font-family: Georgia;">Note that the decode method not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence. This behavior will be extremely useful when we use models that predict new text (either text generated from a prompt, or for sequence-to-sequence problems like translation or summarization).</p>
#
# <p style="font-family: Georgia;">By now you should understand the atomic operations a tokenizer can handle: tokenization, conversion to IDs, and converting IDs back to a string. However, we’ve just scraped the tip of the iceberg. In the following <b><a href="https://www.kaggle.com/dschettler8845/transformers-course-chapter-3-tf-torch">section (Chapter 3)</a></b>, we’ll take our approach to its limits and take a look at how to overcome them.</p>

# + papermill={"duration": 0.977373, "end_time": "2022-01-24T19:40:29.646264", "exception": false, "start_time": "2022-01-24T19:40:28.668891", "status": "completed"} tags=[]
print(f"\n\n{'-'*85}\n\t\t\tCONTINUING EXAMPLE FROM PREVIOUS CELL\n{'-'*85}")
decoded_sequence = tokenizer.decode(ids)
print(f"\n\n... SEQUENCE\n\t--> '{sequence}'\n")
print(f"... TOKENS (SUBWORD)\n\t--> {tokens}\n")
print(f"... INPUT IDS\n\t--> {ids}\n")
print(f"... DECODED SEQUENCE\n\t--> '{decoded_sequence}'\n")

# + [markdown] papermill={"duration": 0.112439, "end_time": "2022-01-24T19:40:29.871721", "exception": false, "start_time": "2022-01-24T19:40:29.759282", "status": "completed"} tags=[]
# <a id="2_5"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.5 HANDLING MULTIPLE SEQUENCES&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# - <a href="https://www.youtube.com/watch?v=ROxrFOEbsQE" style="font-family: Georgia; color: #ff6f00; font-weight: bold;">[TENSORFLOW] &nbsp;&nbsp; VIDEO LINK - INSTANTIATE A TRANSFORMER MODEL - HUGGING FACE CHANNEL</a>
# - <a href="https://www.youtube.com/watch?v=M6adb1j2jPI" style="font-family: Georgia; color: #EE4C2C; font-weight: bold;">[PYTORCH] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; VIDEO LINK - INSTANTIATE A TRANSFORMER MODEL - HUGGING FACE CHANNEL</a><br>
#
# <p style="font-family: Georgia;">In the previous section, we explored the simplest of use cases: doing inference on a single sequence of a small length. However, some questions emerge already:</p>
#
# <ul style="font-family: Georgia;">
#     <li>How do we handle multiple sequences?</li>
#     <li>How do we handle multiple sequences of <b><i>different lengths?</i></b></li>
#     <li>Are vocabulary indices the only inputs that allow a model to work well?</li>
#     <li>Is there such a thing as too long a sequence?</li>
# </ul>
#
# <p style="font-family: Georgia;">Let’s see what kinds of problems these questions pose, and how we can solve them using the 🤗 Transformers API.</p>
#
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">MODELS EXPECTS A BATCH OF INPUTS</b>
#
# <p style="font-family: Georgia;">In the previous exercise you saw how sequences get translated into lists of numbers. Let’s convert this list of numbers to a tensor and send it to the model:</p>
#
# <center><div class="alert alert-block alert-info" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px;">
#     <br><b>⚠️&nbsp;&nbsp;I think the authors expect the <span style="color: #ff6f00;">TensorFlow</span> version to fail (as the <span style="color: #EE4C2C;">PyTorch</span> version does). However, it appears that the library has been updated to allow for a single example to be passed without the additional batch dimension.</b><br><br>
# </div></center>

# + [markdown] papermill={"duration": 0.113419, "end_time": "2022-01-24T19:40:30.098739", "exception": false, "start_time": "2022-01-24T19:40:29.985320", "status": "completed"} tags=[]
# <br><b style="color: gray; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">LIBRARY AGNOSTIC</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 1.551763, "end_time": "2022-01-24T19:40:31.763293", "exception": false, "start_time": "2022-01-24T19:40:30.211530", "status": "completed"} tags=[]
# Import a tokenizer
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequence = "I've been waiting for a HuggingFace course my whole life."
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# + [markdown] papermill={"duration": 0.113925, "end_time": "2022-01-24T19:40:31.992997", "exception": false, "start_time": "2022-01-24T19:40:31.879072", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 2.125112, "end_time": "2022-01-24T19:40:34.233854", "exception": false, "start_time": "2022-01-24T19:40:32.108742", "status": "completed"} tags=[]
# Tensorflow specific import, input coercion and model instantiation
from transformers import TFAutoModelForSequenceClassification
tf_model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
tf_input_ids = tf.constant(ids)

# ********** This line will fail **********
#  - Call the model on the inputs.
# ********** This line will fail **********
try:
    print(tf_model(tf_input_ids))
except:
    print("You would have seen this error if we hadn't `try/except` captured it.\n\n")
    print("\tInvalidArgumentError: Input to reshape is a tensor with 14 values, but the requested shape has 196 [Op:Reshape]")
# ********** This line will fail **********

# + [markdown] papermill={"duration": 0.67938, "end_time": "2022-01-24T19:40:34.986187", "exception": false, "start_time": "2022-01-24T19:40:34.306807", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 1.795445, "end_time": "2022-01-24T19:40:36.855150", "exception": false, "start_time": "2022-01-24T19:40:35.059705", "status": "completed"} tags=[]
# PyTorch specific import, input coercion and model instantiation
from transformers import AutoModelForSequenceClassification
pt_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
pt_input_ids = torch.tensor(ids)

# ********** This line will fail **********
#  - Call the model on the inputs.
# ********** This line will fail **********
try:
    print(pt_model(pt_input_ids))
except:
    print("You would have seen this error if we hadn't `try/except` captured it.\n\n")
    print("\tIndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)\n\n")
# ********** This line will fail **********

# + [markdown] papermill={"duration": 0.073583, "end_time": "2022-01-24T19:40:37.002878", "exception": false, "start_time": "2022-01-24T19:40:36.929295", "status": "completed"} tags=[]
# <p style="font-family: Georgia;"><b>Oh no! Why did this fail?</b> We followed the steps from the pipeline in <a href="#2_4"><b>Section 2.4</b></a></p>
#
# <p style="font-family: Georgia;">The problem is that we sent a single sequence to the model, whereas 🤗 Transformers models expect multiple sentences by default. Here we tried to do everything the tokenizer did behind the scenes when we applied it to a sequence, but if you look closely, you’ll see that it didn’t just convert the list of input IDs into a tensor, it added a dimension on top of it.</p>
#     
# <p style="font-family: Georgia;">Let’s try again and add a new dimension. We will also print the new input IDs as well as the resulting logits for each modelling approach.</p>

# + [markdown] papermill={"duration": 0.074209, "end_time": "2022-01-24T19:40:37.152612", "exception": false, "start_time": "2022-01-24T19:40:37.078403", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.189276, "end_time": "2022-01-24T19:40:37.415386", "exception": false, "start_time": "2022-01-24T19:40:37.226110", "status": "completed"} tags=[]
tf_input_ids = tf.constant([ids])
tf_output = tf_model(tf_input_ids)

print("\n\n... TENSORFLOW MODEL ...")
print("\n\t--> Input IDs:\n\t\t", tf_input_ids)
print("\n\t--> Logits:\n\t\t", tf_output.logits)
print("\n\t--> Logits Shape:\n\t\t", tf_output.logits.shape)

# + [markdown] papermill={"duration": 0.074029, "end_time": "2022-01-24T19:40:37.564264", "exception": false, "start_time": "2022-01-24T19:40:37.490235", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.145384, "end_time": "2022-01-24T19:40:37.784563", "exception": false, "start_time": "2022-01-24T19:40:37.639179", "status": "completed"} tags=[]
pt_input_ids = torch.tensor([ids])
pt_output = pt_model(pt_input_ids)

print("\n\n... PYTORCH MODEL ...")
print("\n\t--> Input IDs:\n\t\t", pt_input_ids)
print("\n\t--> Logits:\n\t\t", pt_output.logits)
print("\n\t--> Logits Shape:\n\t\t", pt_output.logits.shape)

# + [markdown] papermill={"duration": 0.114163, "end_time": "2022-01-24T19:40:38.014140", "exception": false, "start_time": "2022-01-24T19:40:37.899977", "status": "completed"} tags=[]
# <p style="font-family: Georgia;"><b>Batching is the act of sending multiple sentences through the model, all at once.</b> If you only have one sentence, you can just build a batch with a single sequence:</p>
#
# <pre style="font-weight: bold; white-space: pre-wrap; background-color: #eee; border: 1px dashed #999; display: block; padding: 16px; line-height:160%">
#     batched_ids = [ids, ids]
# </pre>
#
# <p style="font-family: Georgia;">This is a batch of two identical sequences!</p>
#
# <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px; ">
#     <br><b>✏️&nbsp;&nbsp;<b style="color: black;">TRY IT OUT!</b>&nbsp;&nbsp;&nbsp;&nbsp;Convert this <code style="color:darkgreen;">batched_ids</code> list into a tensor and pass it through your model. Check that you obtain the same logits as before (but twice)!</b><br><br>
# </div></center>
#
# <p style="font-family: Georgia;">Batching allows the model to work when you feed it multiple sentences. Using multiple sequences is just as simple as building a batch with a single sequence. There’s a second issue, though. When you’re trying to batch together two (or more) sentences, they might be of different lengths. If you’ve ever worked with tensors before, you know that they need to be of rectangular shape, so you won’t be able to convert the list of input IDs into a tensor directly. To work around this problem, we usually <b><i>pad</i></b> the inputs.</p>
#
#

# + [markdown] papermill={"duration": 0.117293, "end_time": "2022-01-24T19:40:38.250830", "exception": false, "start_time": "2022-01-24T19:40:38.133537", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b><b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;"> &nbsp;&&nbsp; </b><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.354551, "end_time": "2022-01-24T19:40:38.720713", "exception": false, "start_time": "2022-01-24T19:40:38.366162", "status": "completed"} tags=[]
# Batch the ids as a list
batched_ids = [ids, ids]

###############################################
#               Tensorflow Code               #
###############################################
tf_input_ids = tf.constant(batched_ids)
tf_output = tf_model(tf_input_ids)

print("\n\n... TENSORFLOW MODEL ...")
print("\n\t--> Input IDs:\n\t\t", tf_input_ids)
print("\n\t--> Logits:\n\t\t", tf_output.logits)
print("\n\t--> Logits Shape:\n\t\t", tf_output.logits.shape)
###############################################
    
###############################################
#                 PyTorch Code                #
###############################################
pt_input_ids = torch.tensor(batched_ids)
pt_output = pt_model(pt_input_ids)

print("\n\n... PYTORCH MODEL ...")
print("\n\t--> Input IDs:\n\t\t", pt_input_ids)
print("\n\t--> Logits:\n\t\t", pt_output.logits)
print("\n\t--> Logits Shape:\n\t\t", pt_output.logits.shape)
###############################################

# + [markdown] papermill={"duration": 0.115999, "end_time": "2022-01-24T19:40:38.953088", "exception": false, "start_time": "2022-01-24T19:40:38.837089", "status": "completed"} tags=[]
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">PADDING THE INPUTS</b>
#
# <p style="font-family: Georgia;">The following list of lists cannot be converted to a tensor:</p>
#
# <pre style="font-weight: bold; white-space: pre-wrap; background-color: #eee; border: 1px dashed #999; display: block; padding: 16px; line-height:160%">
#     batched_ids = [
#         <span style="color: brown;">[200, 200, 200]</span>,
#         <span style="color: brown;">[200, 200]</span>,
#     ]
# </pre>
#
# <p style="font-family: Georgia;">In order to work around this, we’ll use padding to make our tensors have a rectangular shape. <b><i>Padding</i></b> makes sure all our sentences have the same length by adding a special word called the <b><i>padding token</i></b> to the sentences with fewer values. For example, if you have 10 sentences with 10 words and 1 sentence with 20 words, <b><i>padding</i></b> will ensure all the sentences have 20 words. In our example, the resulting tensor looks like this:</p>
#
# <pre style="font-weight: bold; white-space: pre-wrap; background-color: #eee; border: 1px dashed #999; display: block; padding: 16px; line-height:160%">
#     padding_id = <span style="color: brown;">100</span>
#     batched_ids = [
#         <span style="color: brown;">[200, 200, 200]</span>,
#         <span style="color: brown;">[200, 200, padding_id]</span>,
#     ]
# </pre>
#
#
# <p style="font-family: Georgia;">The padding token ID can be found in <b><code>tokenizer.pad_token_id</code></b> (0 in this case). Let’s use it and send our two sentences through the model individually and batched together.</p>

# + [markdown] papermill={"duration": 0.119775, "end_time": "2022-01-24T19:40:39.189861", "exception": false, "start_time": "2022-01-24T19:40:39.070086", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 2.031817, "end_time": "2022-01-24T19:40:41.339460", "exception": false, "start_time": "2022-01-24T19:40:39.307643", "status": "completed"} tags=[]
# Initialize the sequences as described above
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id], #tokenizer.pad_token_id=0
]

# Pass the various sequences through the model and compare the logits
print("\n\n\n... TENSORFLOW MODEL...")
tf_model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
print(f"\n\tSEQUENCE ONE LOGITS\n\t\t--> {tf_model(tf.constant(sequence1_ids)).logits}")
print(f"\n\tSEQUENCE TWO LOGITS\n\t\t--> {tf_model(tf.constant(sequence2_ids)).logits}")
print(f"\n\tBATCHED SEQUENCE LOGITS\n\t\t--> {tf_model(tf.constant(batched_ids)).logits}\n")

# + [markdown] papermill={"duration": 0.116182, "end_time": "2022-01-24T19:40:41.573645", "exception": false, "start_time": "2022-01-24T19:40:41.457463", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 1.905892, "end_time": "2022-01-24T19:40:43.596144", "exception": false, "start_time": "2022-01-24T19:40:41.690252", "status": "completed"} tags=[]
# Initialize the sequences as described above
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id], #tokenizer.pad_token_id=0
]

# Pass the various sequences through the model and compare the logits
print("\n\n\n... PYTORCH MODEL...")
pt_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
print(f"\n\tSEQUENCE ONE LOGITS\n\t\t--> {pt_model(torch.tensor(sequence1_ids)).logits}")
print(f"\n\tSEQUENCE TWO LOGITS\n\t\t--> {pt_model(torch.tensor(sequence2_ids)).logits}")
print(f"\n\tBATCHED SEQUENCE LOGITS\n\t\t--> {pt_model(torch.tensor(batched_ids)).logits}\n")

# + [markdown] papermill={"duration": 0.119349, "end_time": "2022-01-24T19:40:43.834120", "exception": false, "start_time": "2022-01-24T19:40:43.714771", "status": "completed"} tags=[]
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">ATTENTION MASKS</b>
#
# <p style="font-family: Georgia;"><b>Hmmm, wait!</b> There was something wrong with the logits in our batched predictions: the second row should be the same as the logits for the second sentence, but we got completely different values!</p>
#
# <p style="font-family: Georgia;">This is because the key feature of Transformer models is attention layers that <b><i>contextualize</i></b> each token. These will take into account the padding tokens since they attend to all of the tokens of a sequence. To get the same result when passing individual sentences of different lengths through the model or when passing a batch with the same sentences and padding applied, we need to tell those attention layers to ignore the padding tokens. This is done by using an <b><i>attention mask</i></b>.</p>
#
# <p style="font-family: Georgia;"><b><i>Attention masks</i></b> are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s:</p> 
# <ul style="font-family: Georgia;">
#     <li><b><code>1</code></b> indicates the corresponding tokens that should be attended to.</li>
#     <li><b><code>0</code></b> indicates the corresponding tokens that should not be attended to (i.e., they should be ignored by the attention layers of the model).</li>
# </ul>
#
# <p style="font-family: Georgia;">Now we will complete the previous example with an attention mask. Because we are using an attention mask, we will get the same logits for the second sentence in the batch. Pay attention to how the last value of the second sequence is a padding ID, (which is a 0 value in the attention mask.)</p>
#
# <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px; ">
#     <br><b>✏️&nbsp;&nbsp;<b style="color: black;">TRY IT OUT!</b><br><br>Apply the tokenization manually on the two sentences used in <a href="#2_4"><b>Section 2.4</b></a> (“I’ve been waiting for a HuggingFace course my whole life.” and “I hate this so much!”). Pass them through the model and check that you get the same logits as in <a href="#2_4"><b>Section 2.4</b></a>.<br><br>Now batch them together using the padding token, then create the proper attention mask. Check that you obtain the same results when going through the model!</b><br><br>
# </div></center>

# + [markdown] papermill={"duration": 0.119255, "end_time": "2022-01-24T19:40:44.072616", "exception": false, "start_time": "2022-01-24T19:40:43.953361", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b><b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;"> &nbsp;&&nbsp; </b><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 0.251526, "end_time": "2022-01-24T19:40:44.443869", "exception": false, "start_time": "2022-01-24T19:40:44.192343", "status": "completed"} tags=[]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

###############################################
#               Tensorflow Code               #
###############################################
tf_outputs = tf_model(tf.constant(batched_ids), attention_mask=tf.constant(attention_mask))
print("\n\n\n... TENSORFLOW OUTPUTS ...\n")
print(tf_outputs.logits)
###############################################
    
###############################################
#                 PyTorch Code                #
###############################################
pt_outputs = pt_model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print("\n\n... PYTORCH OUTPUTS ...\n")
print(pt_outputs.logits)
###############################################

# + papermill={"duration": 5.432789, "end_time": "2022-01-24T19:40:49.995157", "exception": false, "start_time": "2022-01-24T19:40:44.562368", "status": "completed"} tags=[]
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# tin = try_it_now
tin_seq_1 = "I’ve been waiting for a HuggingFace course my whole life." 
tin_seq_2 = "I hate this so much!"

tin_tokens_1 = tokenizer.tokenize(tin_seq_1)
tin_ids_1 = tokenizer.convert_tokens_to_ids(tin_tokens_1)

tin_tokens_2 = tokenizer.tokenize(tin_seq_2)
tin_ids_2 = tokenizer.convert_tokens_to_ids(tin_tokens_2)

# Pad either side respectively with the pad_token_id
batched_tin_ids = [
    tin_ids_1+[tokenizer.pad_token_id,]*max(0, (len(tin_ids_2)-len(tin_ids_1))),
    tin_ids_2+[tokenizer.pad_token_id,]*max(0, (len(tin_ids_1)-len(tin_ids_2))),
]

tin_attention_mask = [
    [1,]*len(tin_ids_1)+[0,]*max(0, (len(tin_ids_2)-len(tin_ids_1))),
    [1,]*len(tin_ids_2)+[0,]*max(0, (len(tin_ids_1)-len(tin_ids_2))),
]

print("\n\n\n... GENERAL INFO SEQUENCE 1 ...\n")
print(f"\n\tSEQUENCE ONE STRING\n\t\t--> '{tin_seq_1}'")
print(f"\n\tSEQUENCE ONE TOKENS\n\t\t--> {tin_tokens_1}")
print(f"\n\tSEQUENCE ONE INPUT IDS\n\t\t--> {tin_ids_1}\n")

print("\n\n... GENERAL INFO SEQUENCE 2 ...\n")
print(f"\n\tSEQUENCE TWO STRING\n\t\t--> '{tin_seq_2}'")
print(f"\n\tSEQUENCE TWO TOKENS\n\t\t--> {tin_tokens_2}")
print(f"\n\tSEQUENCE TWO INPUT IDS\n\t\t--> {tin_ids_2}\n")

print("\n\n... GENERAL INFO BATCHED SEQUENCES ...\n")
print(f"\n\tBATCHED SEQUENCES\n\t\t--> {[tin_seq_1, tin_seq_2]}")
print(f"\n\tBATCHED TOKENS\n\t\t--> {[tin_tokens_1, tin_tokens_2]}")
print(f"\n\tBATCHED & PADDED INPUT IDS\n\t\t--> {batched_tin_ids}")
print(f"\n\tATTENTION MASK FOR MODEL INPUT\n\t\t--> {tin_attention_mask}\n")

print("\n\n\n... MODEL INFORMATION ...\n")
###############################################
#               Tensorflow Code               #
###############################################
tf_model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
tf_tin_unbatched_output_1 = tf_model(tf.constant([tin_ids_1,])).logits
tf_tin_unbatched_output_2 = tf_model(tf.constant([tin_ids_2,])).logits
tf_tin_batched_output = tf_model(tf.constant(batched_tin_ids),  attention_mask=tf.constant(tin_attention_mask)).logits
print("\n\n... TENSORFLOW MODEL ...\n")
print(f"\n\tSEQUENCE ONE UNBATCHED - OUTPUT LOGITS\n\t\t--> {tf_tin_unbatched_output_1}")
print(f"\n\tSEQUENCE TWO UNBATCHED - OUTPUT LOGITS\n\t\t--> {tf_tin_unbatched_output_2}")
print(f"\n\tSEQUENCES BATCHED - OUTPUT LOGITS\n\t\t--> {tf_tin_batched_output}\n")
###############################################

# ###############################################
# #                 PyTorch Code                #
# ###############################################
pt_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
pt_tin_unbatched_output_1 = pt_model(torch.tensor([tin_ids_1,])).logits
pt_tin_unbatched_output_2 = pt_model(torch.tensor([tin_ids_2,])).logits
pt_tin_batched_output = pt_model(torch.tensor(batched_tin_ids), attention_mask=torch.tensor(tin_attention_mask)).logits
print("\n\n... PYTORCH MODEL ...\n")
print(f"\n\tSEQUENCE ONE UNBATCHED - OUTPUT LOGITS\n\t\t--> {pt_tin_unbatched_output_1}")
print(f"\n\tSEQUENCE TWO UNBATCHED - OUTPUT LOGITS\n\t\t--> {pt_tin_unbatched_output_2}")
print(f"\n\tSEQUENCES BATCHED - OUTPUT LOGITS\n\t\t--> {pt_tin_batched_output}\n")
# ###############################################

# + [markdown] papermill={"duration": 0.078382, "end_time": "2022-01-24T19:40:50.152935", "exception": false, "start_time": "2022-01-24T19:40:50.074553", "status": "completed"} tags=[]
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">LONGER SEQUENCES</b>
#
# <p style="font-family: Georgia;">With Transformer models, there is a limit to the lengths of the sequences we can pass the models. Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences. There are two solutions to this problem:
#
# <ul style="font-family: Georgia;">
#     <li>Use a model with a longer supported sequence length.</li>
#     <li>Truncate your sequences.</li>
# </ul>
#
# Models have different supported sequence lengths, and some specialize in handling very long sequences. <b><a href="https://huggingface.co/transformers/model_doc/longformer.html">Longformer</a></b> is one example, and another is <b><a href="https://huggingface.co/transformers/model_doc/led.html">LED</a></b>. If you’re working on a task that requires very long sequences, we recommend you take a look at those models.</p>
#
# <p style="font-family: Georgia;">Otherwise, we recommend you truncate your sequences by specifying the <b><code>max_sequence_length</code></b> parameter</p>
#
# <pre style="font-weight: bold; white-space: pre-wrap; background-color: #eee; border: 1px dashed #999; display: block; padding: 16px; line-height:160%">
#     sequence = sequence[:max_sequence_length]
# </pre>

# + [markdown] papermill={"duration": 0.077905, "end_time": "2022-01-24T19:40:50.309298", "exception": false, "start_time": "2022-01-24T19:40:50.231393", "status": "completed"} tags=[]
# <a id="2_6"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.6 PUTTING IT ALL TOGETHER&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# <p style="font-family: Georgia;">In the last few sections, we’ve been trying our best to do most of the work by hand. We’ve explored how tokenizers work and looked at tokenization, conversion to input IDs, padding, truncation, and attention masks.</p>
#
# <p style="font-family: Georgia;">However, as we saw in <b><a href="#2_2">Section 2.2</a></b>, the 🤗 Transformers API can handle all of this for us with a high-level function that we’ll dive into here. When you call your <b><code>tokenizer</code></b> directly on the sentence, you get back inputs that are ready to pass through your model</p>
#
# <p style="font-family: Georgia;">The <b><code>tokenizer</code></b> will return an inputs variable containing everything that’s necessary for a model to operate well. For DistilBERT, that includes the input IDs as well as the attention mask. Other models that accept additional inputs will also have those outputs returned by the tokenizer object.</p><br>
#
# <p style="font-family: Georgia;">As we’ll see in some examples below, this method is very powerful.</p>
#
# <ul style="font-family: Georgia;">
#     <li>First, it can tokenize a single sequence</li>
#     <li>It also handles multiple sequences at a time, with no change in the API</li>
#     <li>It can pad according to several objectives</li>
#     <li>It can also truncate sequences:</li>
# </ul>
#
# <p style="font-family: Georgia;">The <b><code>tokenizer</code></b> object can handle the conversion to specific framework tensors, which can then be directly sent to the model. For example, in the following code sample we are prompting the tokenizer to return tensors from the different frameworks — "pt" returns <b style="color: #EE4C2C;">PyTorch</b> tensors, "tf" returns <b style="color: #ff6f00;">TensorFlow</b> tensors, and "np" returns <b style="color: navy;">NumPy</b> arrays:</p>

# + papermill={"duration": 1.589704, "end_time": "2022-01-24T19:40:51.978961", "exception": false, "start_time": "2022-01-24T19:40:50.389257", "status": "completed"} tags=[]
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."
sequences_same_length = ["Been holding out for a HuggingFace course my entire life.", "I've waited for a HuggingFace course so long!"]
sequences_diff_length = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]


print("\n... WHAT CAN THE TOKENIZER OBJECT DO? ...\n")

############################################################################
#                        TOKENIZE A SINGLE SEQUENCE                        #
############################################################################
print("\n\n\tTOKENIZE A SINGLE SEQUENCE (RETURN TF COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequence, return_tensors='tf')}\n")
print("\n\tTOKENIZE A SINGLE SEQUENCE (RETURN PT COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequence, return_tensors='pt')}\n")
############################################################################

############################################################################
#                TOKENIZE MULTIPLE SEQUENCES - SAME LENGTH                 #
############################################################################
print("\n\n\tTOKENIZE MULTIPLE SEQUENCES (SAME LENGTH)(RETURN TF COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_same_length, return_tensors='tf')}\n")
print("\n\tTOKENIZE MULTIPLE SEQUENCE (SAME LENGTH)(RETURN PT COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_same_length, return_tensors='pt')}\n")
############################################################################

############################################################################
#              TOKENIZE MULTIPLE SEQUENCES - DIFFERENT LENGTH              #
############################################################################
print("\n\n\tTOKENIZE MULTIPLE SEQUENCES (DIFF LENGTH W/ PADDING)(RETURN TF COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='tf', padding=True)}\n")
print("\n\tTOKENIZE MULTIPLE SEQUENCE (DIFF LENGTH W/ PADDING)(RETURN PT COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='pt', padding=True)}\n")
############################################################################

############################################################################
#                     PADDING UP TO SEQUENCE MAX LENGTH                    #
############################################################################
# Will pad the sequences up to the maximum sequence length
print("\n\n\tTOKENIZE MULTIPLE SEQUENCES W/ PADDING UP TO SEQUENCE MAX LENGTH (RETURN TF COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='tf', padding='longest')}\n")
print("\n\tTOKENIZE MULTIPLE SEQUENCE W/ PADDING UP TO SEQUENCE MAX LENGTH (RETURN PT COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='pt', padding='longest')}\n")
############################################################################

############################################################################
#                      PADDING UP TO MODEL MAX LENGTH                      #
############################################################################
# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
print("\n\n\tTOKENIZE MULTIPLE SEQUENCES W/ PADDING UP TO MODEL MAX LENGTH (RETURN TF COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='tf', padding='max_length')}\n")
print("\n\tTOKENIZE MULTIPLE SEQUENCE W/ PADDING UP TO MODEL MAX LENGTH (RETURN PT COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='pt', padding='max_length')}\n")
############################################################################

############################################################################
#                      PADDING UP TO SPECIFIED LENGTH                      #
############################################################################
# Will pad the sequences up to the specified max length
print("\n\n\tTOKENIZE MULTIPLE SEQUENCES W/ PADDING UP TO SPECIFIED LENGTH (RETURN TF COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='tf', padding='max_length', max_length=8, truncation=True)}\n")
print("\n\tTOKENIZE MULTIPLE SEQUENCE W/ PADDING UP TO SPECIFIED LENGTH (RETURN PT COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='pt', padding='max_length', max_length=8, truncation=True)}\n")
############################################################################

############################################################################
#                       TRUNCATE TO MODEL MAX LENGTH                       #
############################################################################
# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
print("\n\n\tTOKENIZE MULTIPLE SEQUENCES W/ TRUNCATION TO MODEL MAX LENGTH (RETURN TF COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='tf', truncation=True, padding=True)}\n")
print("\n\tTOKENIZE MULTIPLE SEQUENCES W/ TRUNCATION TO MODEL MAX LENGTH (RETURN PT COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='pt', truncation=True, padding=True)}\n")
############################################################################

############################################################################
#                      TRUNCATE TO A SPECIFIED LENGTH                      #
############################################################################
# Will truncate the sequences that are longer than the specified max length
print("\n\n\tTOKENIZE MULTIPLE SEQUENCES W/ PADDING UP TO SPECIFIED LENGTH (RETURN TF COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='tf', padding=True, truncation=True, max_length=8)}\n")
print("\n\tTOKENIZE MULTIPLE SEQUENCE W/ PADDING UP TO SPECIFIED LENGTH (RETURN PT COMPATIBLE TENSORS)...\n")
print(f"\t\t--> {tokenizer(sequences_diff_length, return_tensors='pt', padding=True, truncation=True, max_length=8)}\n")
############################################################################

# + [markdown] papermill={"duration": 0.078584, "end_time": "2022-01-24T19:40:52.137116", "exception": false, "start_time": "2022-01-24T19:40:52.058532", "status": "completed"} tags=[]
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">SPECIAL TOKENS</b>
#
# <p style="font-family: Georgia;">If we take a look at the input IDs returned by the tokenizer, we will see they are a tiny bit different from what we had earlier. One token ID was added at the beginning, and one at the end. The tokenizer added the special word <b><code>[CLS]</code></b> at the beginning and the special word <b><code>[SEP]</code></b> at the end. This is because the model was pretrained with those, so to get the same results for inference we need to add them as well. Note that some models don’t add special words, or add different ones; models may also add these special words only at the beginning, or only at the end. In any case, the tokenizer knows which ones are expected and will deal with this for you.</p>
#

# + papermill={"duration": 0.094108, "end_time": "2022-01-24T19:40:52.310547", "exception": false, "start_time": "2022-01-24T19:40:52.216439", "status": "completed"} tags=[]
sequence = "I've been waiting for a HuggingFace course my whole life."
print("\n\n\n... SEQUENCE ...")
print(f"\t--> '{sequence}'")

model_inputs = tokenizer(sequence)
print("\n\n... MODEL INPUT IDS (STYLE 1 - FOR MODEL CONSUMPTION) ...")
print("\t-->", model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print("\n\n... MODEL INPUT IDS (STYLE 2 - DIRECT CONVERSION NOT READY FOR MODEL) ...")
print("\t-->", ids)

decoded_style_1 = tokenizer.decode(model_inputs["input_ids"])
print("\n\n... MODEL INPUT IDS DECODED (STYLE 1 - FOR MODEL CONSUMPTION) ...")
print(f"\t--> '{decoded_style_1}'")

decoded_style_2 = tokenizer.decode(ids)
print("\n\n... MODEL INPUT IDS DECODED (STYLE 2 - DIRECT CONVERSION NOT READY FOR MODEL) ...")
print(f"\t--> '{decoded_style_2}'")

# + [markdown] papermill={"duration": 0.122048, "end_time": "2022-01-24T19:40:52.555602", "exception": false, "start_time": "2022-01-24T19:40:52.433554", "status": "completed"} tags=[]
# <br><b style="font-family: Georgia; text-decoration: underline; text-transform: uppercase; font-size: 15px;">Wrapping up: From tokenizer to model</b>
#
# <p style="font-family: Georgia;">Now that we’ve seen all the individual steps the tokenizer object uses when applied on texts, let’s see one final time how it can handle multiple sequences (padding!), very long sequences (truncation!), and multiple types of tensors with its main API</p>
#

# + [markdown] papermill={"duration": 0.121079, "end_time": "2022-01-24T19:40:52.797523", "exception": false, "start_time": "2022-01-24T19:40:52.676444", "status": "completed"} tags=[]
# <br><b style="color: #ff6f00; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">TENSORFLOW</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 3.311177, "end_time": "2022-01-24T19:40:56.231204", "exception": false, "start_time": "2022-01-24T19:40:52.920027", "status": "completed"} tags=[]
# Step 1 - Imports
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Step 2 - Initialize Tokenizer and Model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

# Step 3 - Create a list of sentences
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Step 4 - Tokenize the sentences and pass them to the model
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
output = model(**tokens)

# Step 5 - Print the output
print(f"\n\n\n... TENSORFLOW MODEL OUTPUT\n\n{output}")

# + [markdown] papermill={"duration": 0.081067, "end_time": "2022-01-24T19:40:56.393037", "exception": false, "start_time": "2022-01-24T19:40:56.311970", "status": "completed"} tags=[]
# <br><b style="color: #EE4C2C; font-family: Verdana; font-size: 18px; letter-spacing: 0.4em;">PYTORCH</b> <b style="color: black; font-family: Verdana; font-size: 18px; letter-spacing: 0.2em;">&nbsp;&nbsp;&nbsp;CODE</b>

# + papermill={"duration": 3.391027, "end_time": "2022-01-24T19:40:59.865262", "exception": false, "start_time": "2022-01-24T19:40:56.474235", "status": "completed"} tags=[]
# Step 1 - Imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Step 2 - Initialize Tokenizer and Model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Step 3 - Create a list of sentences
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Step 4 - Tokenize the sentences and pass them to the model
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)

# Step 5 - Print the output
print(f"\n\n\n... PYTORCH MODEL OUTPUT\n\n{output}")

# + [markdown] papermill={"duration": 0.081564, "end_time": "2022-01-24T19:41:00.027939", "exception": false, "start_time": "2022-01-24T19:40:59.946375", "status": "completed"} tags=[]
# <a id="2_7"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.7 BASIC USAGE COMPLETED!&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# <p style="font-family: Georgia;">Great job following the course up to here! To recap, in this chapter you:</p>
#
# <ul style="font-family: Georgia;">
#     <li>Learned the basic building blocks of a Transformer model.</li>
#     <li>Learned what makes up a tokenization pipeline.</li>
#     <li>Saw how to use a Transformer model in practice.</li>
#     <li>Learned how to leverage a tokenizer to convert text to tensors that are understandable by the model.</li>
#     <li>Set up a tokenizer and a model together to get from text to predictions.</li>
#     <li>Learned the limitations of input IDs, and learned about attention masks.</li>
#     <li>Played around with versatile and configurable tokenizer methods.</li>
# </ul>
#
# <p style="font-family: Georgia;">From now on, you should be able to freely navigate the 🤗 Transformers docs: the vocabulary will sound familiar, and you’ve already seen the methods that you’ll use the majority of the time.</p>

# + [markdown] papermill={"duration": 0.080434, "end_time": "2022-01-24T19:41:00.189616", "exception": false, "start_time": "2022-01-24T19:41:00.109182", "status": "completed"} tags=[]
# <a id="2_8"></a>
#
# <br><h3 style="font-family: Georgia; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2.8 CHAPTER QUIZ RECAP&nbsp;&nbsp;&nbsp;&nbsp;<a href="#chapter_2">&#10514;</a></h3>
#
# ---
#
# <center><div class="alert alert-block alert-info" style="margin: 2em; line-height: 1.5em; font-family: Georgia; font-size: 12px;">
#     <br><b>⚠️&nbsp;&nbsp;I am taking the end-of-chapter quiz that is included in the course and simply retrieving the relevant information.</b><br><br>
# </div></center><br>
#
#     
# <ol style="font-family: Georgia;">
#     <li style="font-family: Georgia; font-weight: bold; text-decoration: none; text-transform: uppercase; font-size: 15px;">The order of the language modeling pipeline</li>
#     <ul style="font-family: Georgia;">
#         <li>The tokenizer handles text and returns IDs</li>
#         <li>Then the model handles these IDs and outputs a prediction</li>
#         <li>Finally, the tokenizer can then be used again to convert these predictions back to some text</li>
#     </ul>
#     <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 0.8em; font-family: Georgia; font-size: 12px;">
#     <br><b>📚📚📚&nbsp;&nbsp;&nbsp;&nbsp;The tokenizer can be used for both tokenizing and de-tokenizing&nbsp;&nbsp;&nbsp;&nbsp;📚📚📚</b><br><br>
# </div></center><br>
#     <li style="font-family: Georgia; font-weight: bold; text-decoration: none; text-transform: uppercase; font-size: 15px;">The dimensions of the tensor outputed by the base Transformer model
# </li>
#     <ul style="font-family: Georgia;">
#         <li>The sequence length</li>
#         <li>The batch size</li>
#         <li>The hidden size</li>
#     </ul><br><br>
#     <li style="font-family: Georgia; font-weight: bold; text-decoration: none; text-transform: uppercase; font-size: 15px;">an example of subword tokenization
# </li>
#     <ul style="font-family: Georgia;">
#         <li><b><a href="https://arxiv.org/abs/1508.07909">BPE</a></b></li>
#         <li><b><a href="https://arxiv.org/abs/2106.02289">Unigram</a></b></li>
#         <li><b><a href="https://arxiv.org/pdf/1609.08144.pdf">WordPiece</a></b></li>
#     </ul><br><br>
#     <li style="font-family: Georgia; font-weight: bold; text-decoration: none; text-transform: uppercase; font-size: 15px;">The definition of a 'model head'
# </li>
#     <ul style="font-family: Georgia;">
#         <li>An additional component, usually made up of one or a few layers, to convert the transformer predictions to a task-specific output</li>
#     </ul>
#     <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 0.8em; font-family: Georgia; font-size: 12px;">
#     <br><b>📚📚📚&nbsp;&nbsp;&nbsp;&nbsp;Adaptation heads, also known simply as heads, come up in different forms: language modeling heads, question answering heads, sequence classification heads...&nbsp;&nbsp;&nbsp;&nbsp;📚📚📚</b><br><br>
# </div></center><br>
#     <li style="font-family: Georgia; font-weight: bold; text-decoration: none; text-transform: uppercase; font-size: 15px;">The definition of a <b>[<b style="color: #ff6f00;">TFAutoModel</b>|<b style="color: #EE4C2C;">AutoModel</b>]</b>
# </li>
#     <ul style="font-family: Georgia;">
#         <li>An object that returns the correct architecture based on the checkpoint</li>
#     </ul>
#     <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 0.8em; font-family: Georgia; font-size: 12px;">
#     <br><b>📚📚📚&nbsp;&nbsp;&nbsp;&nbsp;The <b>[<b style="color: #ff6f00;">TFAutoModel</b>|<b style="color: #EE4C2C;">AutoModel</b>]</b> only needs to know the checkpoint from which to initialize to return the correct architecture.&nbsp;&nbsp;&nbsp;&nbsp;📚📚📚</b><br><br>
# </div></center><br>
#     <li style="font-family: Georgia; font-weight: bold; text-decoration: none; text-transform: uppercase; font-size: 15px;">The different techniques to be aware of when batching sequences of different lengths together
# </li>
#     <ul style="font-family: Georgia;">
#         <li>Truncating</li>
#         <li>Padding</li>
#         <li>Attention Masking</li>
#     </ul><br>
#     <li style="font-family: Georgia; font-weight: bold; text-decoration: none; text-transform: uppercase; font-size: 15px;">The point of applying a SoftMax function to the logits output by a sequence classification model
# </li>
#     <ul style="font-family: Georgia;">
#         <li>It applies a lower and upper bound so that they're understandable.</li>
#         <li>The total sum of the output is then 1, resulting in a possible probabilistic interpretation.</li>
#     </ul><br>
#     <li style="font-family: Georgia; font-weight: bold; text-decoration: none; text-transform: uppercase; font-size: 15px;">The method that most of the tokenizer API is centered around
# </li>
#     <ul style="font-family: Georgia;">
#         <li>Calling the tokenizer object directly</li>
#     </ul>
#     <center><div class="alert alert-block alert-success" style="margin: 2em; line-height: 0.8em; font-family: Georgia; font-size: 12px;">
#         <br><b>📚📚📚&nbsp;&nbsp;&nbsp;&nbsp;The <code style="color: darkgreen;">__call__</code> method of the tokenizer is a powerful method which can handle pretty much anything. It is also the method used to retrieve predictions from a model.&nbsp;&nbsp;&nbsp;&nbsp;📚📚📚</b><br><br>
# </div></center><br>
#
# </ol>
#
#
