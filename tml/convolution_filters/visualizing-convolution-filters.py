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

# + [markdown] papermill={"duration": 0.021317, "end_time": "2021-12-04T08:54:07.680244", "exception": false, "start_time": "2021-12-04T08:54:07.658927", "status": "completed"} tags=[]
# # EDA Visualizations for Image Recognition (Conv Filter Edition)

# + [markdown] papermill={"duration": 0.019824, "end_time": "2021-12-04T08:54:07.720448", "exception": false, "start_time": "2021-12-04T08:54:07.700624", "status": "completed"} tags=[]
# ## Dependencies and Imports

# + papermill={"duration": 124.183508, "end_time": "2021-12-04T08:56:11.923821", "exception": false, "start_time": "2021-12-04T08:54:07.740313", "status": "completed"} tags=[]
# !pip install -q timm
# !pip install -q torch==1.10.0 torchvision==0.11.1 torchaudio===0.10.0

# + papermill={"duration": 1.985461, "end_time": "2021-12-04T08:56:13.929702", "exception": false, "start_time": "2021-12-04T08:56:11.944241", "status": "completed"} tags=[]
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import timm
import torch
import torchvision
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)

# %matplotlib inline
import glob
import os
from math import ceil
import random

import cv2
import PIL
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from typing import *

# + [markdown] papermill={"duration": 0.019763, "end_time": "2021-12-04T08:56:13.969857", "exception": false, "start_time": "2021-12-04T08:56:13.950094", "status": "completed"} tags=[]
# ## Config and Logging

# + papermill={"duration": 0.028948, "end_time": "2021-12-04T08:56:14.020507", "exception": false, "start_time": "2021-12-04T08:56:13.991559", "status": "completed"} tags=[]
import logging
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

def init_logger(log_file: str = "info.log") -> logging.Logger:
    """Initialize logger and save to file.

    Consider having more log_file paths to save, eg: debug.log, error.log, etc.

    Args:
        log_file (str, optional): [description]. Defaults to Path(LOGS_DIR, "info.log").

    Returns:
        logging.Logger: [description]
    """
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(
        Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    file_handler = FileHandler(filename=log_file)
    file_handler.setFormatter(
        Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

logger = init_logger()


# + [markdown] papermill={"duration": 0.020539, "end_time": "2021-12-04T08:56:14.062467", "exception": false, "start_time": "2021-12-04T08:56:14.041928", "status": "completed"} tags=[]
# ### Utils

# + papermill={"duration": 0.029894, "end_time": "2021-12-04T08:56:14.112867", "exception": false, "start_time": "2021-12-04T08:56:14.082973", "status": "completed"} tags=[]
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(
        figsize=(20, 15),
        nrows=ceil(len(img_matrix_list) / ncols),
        ncols=ncols,
        squeeze=False,
    )
    fig.suptitle(main_title, fontsize=30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()


# + [markdown] papermill={"duration": 0.020247, "end_time": "2021-12-04T08:56:14.153223", "exception": false, "start_time": "2021-12-04T08:56:14.132976", "status": "completed"} tags=[]
# ## Seeding

# + papermill={"duration": 0.032784, "end_time": "2021-12-04T08:56:14.206470", "exception": false, "start_time": "2021-12-04T08:56:14.173686", "status": "completed"} tags=[]
def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def seed_worker(_worker_id) -> None:
    """Seed a worker with the given ID."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
seed_all()

# + [markdown] papermill={"duration": 0.02011, "end_time": "2021-12-04T08:56:14.247331", "exception": false, "start_time": "2021-12-04T08:56:14.227221", "status": "completed"} tags=[]
# ### Transforms Params

# + papermill={"duration": 0.029496, "end_time": "2021-12-04T08:56:14.297303", "exception": false, "start_time": "2021-12-04T08:56:14.267807", "status": "completed"} tags=[]
mean: List[float] = [0.485, 0.456, 0.406]
std: List[float] = [0.229, 0.224, 0.225]
image_size: int = 224

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ]
)

pre_normalize_transform =  torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
    ]
)

# + [markdown] papermill={"duration": 0.020216, "end_time": "2021-12-04T08:56:14.337924", "exception": false, "start_time": "2021-12-04T08:56:14.317708", "status": "completed"} tags=[]
# ## Visualizations

# + papermill={"duration": 0.027093, "end_time": "2021-12-04T08:56:14.385398", "exception": false, "start_time": "2021-12-04T08:56:14.358305", "status": "completed"} tags=[]
cat_p = "../input/petfinder-pawpularity-score/train/0042bc5bada6d1cf8951f8f9f0d399fa.jpg"
dog_p = "../input/petfinder-pawpularity-score/train/86a71a412f662212fe8dcd40fdaee8e6.jpg"

# + papermill={"duration": 0.660279, "end_time": "2021-12-04T08:56:15.066262", "exception": false, "start_time": "2021-12-04T08:56:14.405983", "status": "completed"} tags=[]
# plot cat and dog with title using PIL
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
cat = PIL.Image.open(cat_p)
plt.imshow(cat)
plt.title("Cat")
plt.subplot(1, 2, 2)
dog = PIL.Image.open(dog_p)
plt.imshow(dog)
plt.title("Dog")
plt.show();


# + [markdown] papermill={"duration": 0.027989, "end_time": "2021-12-04T08:56:15.122368", "exception": false, "start_time": "2021-12-04T08:56:15.094379", "status": "completed"} tags=[]
# ## Convolution Layers <a id="2.3"></a>
#
# Courtesy of [https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models](https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models).
#
# ---
#
# Convolution is a rather simple algorithm which involves a kernel (a 2D matrix) which moves over the entire image, calculating dot products with each window along the way. The GIF below demonstrates convolution in action.
#
# <center><img src="https://i.imgur.com/wYUaqR3.gif" width="450px"></center>
#
# The above process can be summarized with an equation, where *f* is the image and *h* is the kernel. The dimensions of *f* are *(m, n)* and the kernel is a square matrix with dimensions smaller than *f*:
#
# <center><img src="https://i.imgur.com/9scTOGv.png" width="350px"></center>
# <br>
#
# In the above equation, the kernel *h* is moving across the length and breadth of the image. The dot product of *h* with a sub-matrix or window of matrix *f* is taken at each step, hence the double summation (rows and columns). 

# + [markdown] papermill={"duration": 0.027904, "end_time": "2021-12-04T08:56:15.179144", "exception": false, "start_time": "2021-12-04T08:56:15.151240", "status": "completed"} tags=[]
# I have always remembered from the revered Andrew Ng about how he taught us about what convolutional layers do.
#
# > In the beginning, the conv layers are of low level abstraction, detailing a image's features such as shapes and sizes. In particular, he described to us the horizontal and vertical conv filters. As the conv layers go later, it will pick up on many abstract features, which is not really easily distinguished by human eyes.
#
# Below, we see an example of horizontal and vertical filters.

# + papermill={"duration": 0.041715, "end_time": "2021-12-04T08:56:15.248993", "exception": false, "start_time": "2021-12-04T08:56:15.207278", "status": "completed"} tags=[]
def conv_horizontal(image: np.ndarray) -> None:
    """Plot the horizontal convolution of the image.

    Args:
        image (torch.Tensor): [description]
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((3, 3), np.float32)
    kernel[1] = np.array([0, 0, 0], np.float32)
    kernel[2] = np.array([-1, -1, -1], np.float32)
    conv = cv2.filter2D(image, -1, kernel)
    ax[0].imshow(image)
    ax[0].set_title("Original Image", fontsize=24)
    ax[1].imshow(conv)
    ax[1].set_title("Convolved Image with horizontal edges", fontsize=24)
    plt.show()


def conv_vertical(image: np.ndarray) -> None:
    """Plot the vertical convolution of the image.

    Args:
        image (torch.Tensor): [description]
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((3, 3), np.float32)
    kernel[0] = np.array([1, 0, -1])
    kernel[1] = np.array([1, 0, -1])
    kernel[2] = np.array([1, 0, -1])
    conv = cv2.filter2D(image, -1, kernel)
    ax[0].imshow(image)
    ax[0].set_title("Original Image", fontsize=24)
    ax[1].imshow(conv)
    ax[1].set_title("Convolved Image with vertical edges", fontsize=24)
    plt.show()


# + [markdown] papermill={"duration": 0.0279, "end_time": "2021-12-04T08:56:15.304880", "exception": false, "start_time": "2021-12-04T08:56:15.276980", "status": "completed"} tags=[]
# Well, I can easily make out the horizontal and vertical edges from the cat image! 

# + papermill={"duration": 2.070837, "end_time": "2021-12-04T08:56:17.403882", "exception": false, "start_time": "2021-12-04T08:56:15.333045", "status": "completed"} tags=[]
conv_horizontal(np.asarray(cat))
conv_vertical(np.asarray(cat))


# + [markdown] papermill={"duration": 0.104905, "end_time": "2021-12-04T08:56:17.617937", "exception": false, "start_time": "2021-12-04T08:56:17.513032", "status": "completed"} tags=[]
# The issue is, I want to visualize what our models' conv layers are seeing, like for example, the first conv layer usually has 64 filters, that is a whooping 64 different combinations of filters, each doing a slightly different thing. A mental model that I have for the first conv layer looks something like the following.
#
# ```python
# conv_1_filters = ["vertical edge detector", "horizontal edge detector",
#                   "slanted 45 degrees detector", "slanted 180 degrees detector",
#                   ...]
# ```

# + [markdown] papermill={"duration": 0.109568, "end_time": "2021-12-04T08:56:17.832025", "exception": false, "start_time": "2021-12-04T08:56:17.722457", "status": "completed"} tags=[]
# ### Feature Extractor using PyTorch's native Feature Extraction Module
#
# In order to visualize properly, I made use of **PyTorch's** newest `feature_extraction` module to do so. Note that the new feature is still in development, but it does make my life easier and reduces overhead. I no longer need use `hooks` or what not to plot layer information!
#
# We just need to import
# ```python
# from torchvision.models.feature_extraction import (create_feature_extractor,
#                                                    get_graph_node_names)
# ```

# + papermill={"duration": 0.120603, "end_time": "2021-12-04T08:56:18.057131", "exception": false, "start_time": "2021-12-04T08:56:17.936528", "status": "completed"} tags=[]
def get_conv_layers(model: torchvision.models) -> Dict[str, str]:
    """Create a function that give me the conv layers of PyTorch model.

    Args:
        model (Union[torchvision.models, timm.models]): A PyTorch model.

    Returns:
        conv_layers (Dict[str, str]): {"layer1.0.conv1": layer1.0.conv1, ...}
    """
    conv_layers = {}
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers[name] = name
    return conv_layers


def get_feature_maps(
    model_name: str, image: torch.Tensor, reduction: str = "mean",
    pretrained: bool = True
) -> Union[Dict[str, torch.Tensor], List[torch.Tensor], List[str]]:
    """Function to plot feature maps from PyTorch models.

    Args:
        model_name (str): Name of the model to use.
        image (torch.Tensor): image should be a tensor of shape (1, 3, H, W)
        reduction (str, optional): Defaults to "mean". One of ["mean", "max", "sum"]
        pretrained (bool): whether the model is pretrained or not

    Raises:
        ValueError: Must use Torchvision models.

    Returns:
        model_feature_maps (Dict[str, torch.Tensor]): {"conv_1": conv_1_feature_map, ...}
        processed_feature_maps (List[torch.Tensor]): [conv_1_feature_map, ...] processed using a reduction method.
        feature_map_names (List[str]): [conv_1, ...]

    Example:
        >>> from torchvision.models.vgg import vgg16
        >>> model = vgg16(pretrained=True)
        >>> image = torch.rand(1, 3, 224, 224)
        >>> feature_maps = get_feature_maps(model, image, reduction="mean")

    Reduction:
        If a feature map has 4 filters, in the shape of (4, H, W) = (4, 32, 32), then the reduction can be done as follows:
        >>> reduction = "mean": There are 4 filters in this feature map, you can imagine it as 4 32x32 images.
                                We sum up all 4 filters element-wise and get a single 32x32 image.
                                Then we take the mean of all 32x32 images by dividing by num of kernels to get a single 32x32 image, which is reduction="mean".
    """

    try:
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
    except AttributeError:
        raise ValueError(f"Model {model_name} not found.")

    train_nodes, eval_nodes = get_graph_node_names(model)
    logger.info(f"The train nodes of the model graph is:\n\n{train_nodes}")

    return_conv_nodes = get_conv_layers(model)
    feature_extractor = create_feature_extractor(model, return_nodes=return_conv_nodes)

    # `model_feature_maps` will be a dict of Tensors, each representing a feature map
    model_feature_maps = feature_extractor(image)

    processed_feature_maps = []
    feature_map_names = []

    for conv_name, conv_feature_map in model_feature_maps.items():

        conv_feature_map = conv_feature_map.squeeze(dim=0)
        num_filters = conv_feature_map.shape[0]

        if reduction == "mean":
            gray_scale = torch.sum(conv_feature_map, dim=0) / num_filters
        elif reduction == "max":
            gray_scale = torch.max(conv_feature_map, dim=0)
        elif reduction == "sum":
            gray_scale = torch.sum(conv_feature_map, dim=0)

        processed_feature_maps.append(gray_scale.data.cpu().numpy())
        feature_map_names.append(conv_name)

    return model_feature_maps, processed_feature_maps, feature_map_names


# + [markdown] papermill={"duration": 0.10428, "end_time": "2021-12-04T08:56:18.264901", "exception": false, "start_time": "2021-12-04T08:56:18.160621", "status": "completed"} tags=[]
# ### Visualizing VGG16 and ResNet18

# + [markdown] papermill={"duration": 0.10428, "end_time": "2021-12-04T08:56:18.474457", "exception": false, "start_time": "2021-12-04T08:56:18.370177", "status": "completed"} tags=[]
# #### Step 1: Initialize the models.
#
# As of now, I recommend using `torchvision`'s models. Ideally, I will want to use `timm` library for a more detailed list, but there are some bugs that is not easily integrated with the module.

# + papermill={"duration": 39.29018, "end_time": "2021-12-04T08:56:57.869225", "exception": false, "start_time": "2021-12-04T08:56:18.579045", "status": "completed"} tags=[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.models as models

vgg16_pretrained_true = models.vgg16(pretrained=True)
vgg16_pretrained_true = vgg16_pretrained_true.to(device)

resnet18_pretrained_true = models.resnet18(pretrained=True)
resnet18_pretrained_true = resnet18_pretrained_true.to(device)

# + papermill={"duration": 0.16738, "end_time": "2021-12-04T08:56:58.142823", "exception": false, "start_time": "2021-12-04T08:56:57.975443", "status": "completed"} tags=[]
# Get node names
train_nodes, eval_nodes = get_graph_node_names(vgg16_pretrained_true)
logger.info(f"Train nodes of VGG16:\n\n{train_nodes}")

train_nodes, eval_nodes = get_graph_node_names(resnet18_pretrained_true)
logger.info(f"Train nodes of ResNet18:\n\n{train_nodes}")

# + [markdown] papermill={"duration": 0.104241, "end_time": "2021-12-04T08:56:58.352015", "exception": false, "start_time": "2021-12-04T08:56:58.247774", "status": "completed"} tags=[]
# **Good God!** When I saw the layer names from `vgg16`, I nearly fainted, I see no easy way to know which layer belongs to a Conv layer. I understand that `get_graph_node_names` will get all the nodes on the model's graph, but it is difficult to map the node names to a layer if it is named as such, seeing `resnet18`'s node names is much easier for one to identify which is conv layer or not.
#
# ```python
# train_nodes, eval_nodes = get_graph_node_names(model)
# logger.info(f"The train nodes of the model graph is:\n\n{train_nodes}")
# ```
#
# Thus I wrote a small function `get_conv_layers` to get the conv layer names. It is not perfect, as downsample layers (1x1 conv layers) are tagged under `Conv2d` but we may not really need to use them to visualize our feature maps. One can tweak a bit if need be, but for now, I will get all layers that use the `Conv2d` blocks.
#
# If the feature names in vgg16 are named with conv, then we can simply use a small loop below to find the conv layer names.
#
# ```python
# conv_layers = []
# for node in nodes:
#     if "conv" in node:
#         conv_layers.append(node)
# ```

# + [markdown] papermill={"duration": 0.103812, "end_time": "2021-12-04T08:56:58.559827", "exception": false, "start_time": "2021-12-04T08:56:58.456015", "status": "completed"} tags=[]
# I actually thought ResNet18 has 18 conv layers, but even minusing to 3 downsample layers, it's 17 conv layers, wonder why?

# + [markdown] papermill={"duration": 0.103984, "end_time": "2021-12-04T08:56:58.768317", "exception": false, "start_time": "2021-12-04T08:56:58.664333", "status": "completed"} tags=[]
# #### Step 2: Transform the Tensors
#
# The PyTorch `feature_extraction` expects the image input to be of shape `[B,C,H,W]`. 
#
# ```python
# # We use torchvision's transform to transform the cat image to channels first.
# cat_tensor = transform(cat)
#
# # Now feature_extractor expects batch_size x C x H x W, so we expand one dimension in the 0th dim
# cat_tensor = cat_tensor.unsqueeze(dim=0).to(device)
# ```

# + papermill={"duration": 0.128394, "end_time": "2021-12-04T08:56:59.002398", "exception": false, "start_time": "2021-12-04T08:56:58.874004", "status": "completed"} tags=[]
# We use torchvision's transform to transform the cat image with resize and normalization.
# Conveniently, also making it channel first!
cat_tensor = transform(cat)
dog_tensor = transform(dog)
assert cat_tensor.shape[0] == dog_tensor.shape[0] == 3, "PyTorch expects Channel First!"

# Now feature_extractor expects batch_size x C x H x W, so we expand one dimension in the 0th dim
cat_tensor = cat_tensor.unsqueeze(dim=0).to(device)
dog_tensor = dog_tensor.unsqueeze(dim=0).to(device)

logger.info(f"\n\ncat_tensor's shape:\n{cat_tensor.shape}\n\ndog_tensor's shape:\n{dog_tensor.shape}")

# + [markdown] papermill={"duration": 0.104494, "end_time": "2021-12-04T08:56:59.211659", "exception": false, "start_time": "2021-12-04T08:56:59.107165", "status": "completed"} tags=[]
# #### Step 3: Plotting the Feature Maps
#
# We first walk through `get_feature_maps` and see what my function is doing.

# + [markdown] papermill={"duration": 0.10438, "end_time": "2021-12-04T08:56:59.421265", "exception": false, "start_time": "2021-12-04T08:56:59.316885", "status": "completed"} tags=[]
# ```python
# # Get node names
# train_nodes, eval_nodes = get_graph_node_names(model)
#
# # Since get node names do not indicate properly which is a conv layer or not,
# # we use get_conv_layer instead to do the job, which returns a dict {"conv_layer_name": "conv_layer_name"}
# return_conv_nodes = get_conv_layers(model)
#
# # call create_feature_extractor on the model and its corresponding conv layer names.
# feature_extractor = create_feature_extractor(model, return_nodes=return_conv_nodes)
#
# # `model_feature_maps` will be a dict of Tensors, each representing a feature map
# # {"conv_layer_1": output filter map,...}
# model_feature_maps = feature_extractor(image)
#
# # we need to further process the feature maps
# processed_feature_maps, feature_map_names = [], []
#
#
# for conv_name, conv_feature_map in model_feature_maps.items():
#     # Squeeze the dimension from [1, 64, 32, 32] to [64, 32, 32]
#     # This means we have 64 filters of 32x32 "images" or kernels
#     conv_feature_map = conv_feature_map.squeeze(dim=0)
#     # get number of feature/kernels in this layer
#     num_filters = conv_feature_map.shape[0]
#     
#
#     # If a feature map has 4 filters, in the shape of (4, H, W) = (4, 32, 32), then the reduction mean can be done as follows: There are 4 filters in this feature map, you can imagine it as 4 32x32 images.
#     # Step 1: We sum up all 4 filters element-wise and get a single 32x32 image.
#     # Step 2: Then we take the mean of all 32x32 images to get a single 32x32 image, which is reduction="mean".
#     if reduction == "mean":
#         gray_scale = torch.sum(conv_feature_map, dim=0) / num_filters
#     elif reduction == "max":
#         gray_scale = torch.max(conv_feature_map, dim=0)
#     elif reduction == "sum":
#         gray_scale = torch.sum(conv_feature_map, dim=0)
#
#     processed_feature_maps.append(gray_scale.data.cpu().numpy())
#     feature_map_names.append(conv_name)
# ```

# + papermill={"duration": 2.428962, "end_time": "2021-12-04T08:57:01.955153", "exception": false, "start_time": "2021-12-04T08:56:59.526191", "status": "completed"} tags=[]
_, vgg16_processed_feature_maps, vgg16_feature_map_names = get_feature_maps(
    model_name="vgg16", image=cat_tensor, reduction="mean", pretrained=True
)
_, resnet18_processed_feature_maps, resnet18_feature_map_names = get_feature_maps(
    model_name="resnet18", image=cat_tensor, reduction="mean", pretrained=True
)


# + [markdown] papermill={"duration": 0.10515, "end_time": "2021-12-04T08:57:02.166691", "exception": false, "start_time": "2021-12-04T08:57:02.061541", "status": "completed"} tags=[]
# Then we create a simple `plot_feature_maps` that take in the `processed_feature_maps` and `feature_map_names` to plot them.

# + papermill={"duration": 0.115373, "end_time": "2021-12-04T08:57:02.387618", "exception": false, "start_time": "2021-12-04T08:57:02.272245", "status": "completed"} tags=[]
def plot_feature_maps(
    processed_feature_maps: List[torch.Tensor], feature_map_names: List[str], nrows: int,
    title: str = None
) -> None:
    """Plot the feature maps.

    Args:
        processed_feature_maps (List[torch.Tensor]): [description]
        feature_map_names (List[str]): [description]
        nrows (int): [description]
    """
    fig = plt.figure(figsize=(30, 50))
    ncols = len(processed_feature_maps) // nrows + 1
    for i in range(len(processed_feature_maps)):
        a = fig.add_subplot(nrows, ncols, i + 1)
        imgplot = plt.imshow(processed_feature_maps[i])
        a.axis("off")
        a.set_title(feature_map_names[i].split("(")[0], fontsize=30)

    fig.suptitle(title, fontsize=50)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig(title, bbox_inches='tight')
    plt.show();


# + papermill={"duration": 6.320362, "end_time": "2021-12-04T08:57:08.813189", "exception": false, "start_time": "2021-12-04T08:57:02.492827", "status": "completed"} tags=[]
plot_feature_maps(
    vgg16_processed_feature_maps,
    vgg16_feature_map_names,
    nrows=5,
    title="VGG16 Pretrained Feature Maps",
)

plot_feature_maps(
    resnet18_processed_feature_maps,
    resnet18_feature_map_names,
    nrows=5,
    title="ResNet18 Pretrained Feature Maps",
)

# + [markdown] papermill={"duration": 0.14375, "end_time": "2021-12-04T08:57:09.100618", "exception": false, "start_time": "2021-12-04T08:57:08.956868", "status": "completed"} tags=[]
# ## Comparison with Randomly Initialized Weights
#
# We know that if the model is not pretrained, it will initialize with random weights using weight initialization methods such as Kaimin or Xavier. I expect the edges to be not so "smooth" as the ones that are pretrained! This is logical, as the filters in the conv layers are mostly random, and we have not trained any epochs yet, so let's see what it gives us.

# + papermill={"duration": 2.962974, "end_time": "2021-12-04T08:57:12.207007", "exception": false, "start_time": "2021-12-04T08:57:09.244033", "status": "completed"} tags=[]
_, vgg16_processed_feature_maps, vgg16_feature_map_names = get_feature_maps(
    model_name="vgg16", image=cat_tensor, reduction="mean", pretrained=False
)
_, resnet18_processed_feature_maps, resnet18_feature_map_names = get_feature_maps(
    model_name="resnet18", image=cat_tensor, reduction="mean", pretrained=False
)

# + papermill={"duration": 6.388335, "end_time": "2021-12-04T08:57:18.738677", "exception": false, "start_time": "2021-12-04T08:57:12.350342", "status": "completed"} tags=[]
plot_feature_maps(
    vgg16_processed_feature_maps,
    vgg16_feature_map_names,
    nrows=5,
    title="VGG16 NOT Pretrained Feature Maps",
)
plot_feature_maps(
    resnet18_processed_feature_maps,
    resnet18_feature_map_names,
    nrows=5,
    title="ResNet18 NOT Pretrained Feature Maps",
)

# + [markdown] papermill={"duration": 0.178593, "end_time": "2021-12-04T08:57:19.099442", "exception": false, "start_time": "2021-12-04T08:57:18.920849", "status": "completed"} tags=[]
# References:
#
# - [https://pytorch.org/vision/stable/feature_extraction.html](https://pytorch.org/vision/stable/feature_extraction.html)
# - [https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch](https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573)
# - [https://pytorch.org/blog/FX-feature-extraction-torchvision/](https://pytorch.org/blog/FX-feature-extraction-torchvision/)
# - [https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models](https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models)
