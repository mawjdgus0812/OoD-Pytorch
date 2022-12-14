import numpy as np
import torch
import clip
from tqdm import tqdm

from torchvision.datasets import  MNIST, CIFAR10, SVHN
from loader import AmbiguousCIFAR,  notMNIST
import torchvision.transforms as transforms
import torchvision

from PIL import Image
from torch.autograd import Variable

from figure import make_figure

print("Torch version: ", torch.__version__)

print(clip.available_models())

model, preprocess = clip.load('ViT-B/32')

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

def Prompt_classes():
    cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck','unknown']
    cifar10_templates = [
    'Let\'s think step by step, a photo of a {}.',
    'Let\'s think step by step, a blurry photo of a {}.',
    'Let\'s think step by step, a black and white photo of a {}.',
    'Let\'s think step by step, a low contrast photo of a {}.',
    'Let\'s think step by step, a high contrast photo of a {}.',
    'Let\'s think step by step, a bad photo of a {}.',
    'Let\'s think step by step, a good photo of a {}.',
    'Let\'s think step by step, a photo of a small {}.',
    'Let\'s think step by step, a photo of a big {}.',
    'Let\'s think step by step, a photo of the {}.',
    'Let\'s think step by step, a blurry photo of the {}.',
    'Let\'s think step by step, a black and white photo of the {}.',
    'Let\'s think step by step, a low contrast photo of the {}.',
    'Let\'s think step by step, a high contrast photo of the {}.',
    'Let\'s think step by step, a bad photo of the {}.',
    'Let\'s think step by step, a good photo of the {}.',
    'Let\'s think step by step, a photo of the small {}.',
    'Let\'s think step by step, a photo of the big {}.',]

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("CIFAR10: ",f"{len(cifar10_classes)} classes, {len(cifar10_templates)} templates")

class_map = {'CIFAR10': cifar10_classes}
template_map = {'CIFAR10': cifar10_templates}

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk) # ?????? ?????? topk??? maxk??? ??????
    batch_size = target.size(0) # ?????????????????? ????????? ?????????

    _, pred = output.topk(maxk, 1, True, True) # ????????? ???????????? topk??? ?????????, 
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred)) # ????????? ????????????

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

@torch.no_grad()
def extract_text_features(dataset_name):
    # code borrowed from: https://github.com/openai/CLIP/blob/fcab8b6eb92af684e7ff0a904464be7b99b49b88/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    class_names = class_map[dataset_name] # dataset????????? ????????????. (MNIST or CIFAR)
    templates = template_map[dataset_name] # template??? dataset ?????? ????????????.
    model.to(device) # ????????? GPU??? ?????????.
    model.eval() # ????????? ??????????????? ?????????.

    zeroshot_weights = [] # zeroshot weight??? ????????? ?????????
    for classname in class_names: # ????????? ??????(???????????? ????????? ????????? ?????????)
        texts = [template.format(classname) for template in templates] # ??? ???????????? ???????????? template??? ???????????? ???????????? ?????????
        texts = clip.tokenize(texts).to(device) # ????????? ????????????, ????????? ?????????????????? ?????? ????????? ??????
        class_embeddings = model.encode_text(texts) # ????????? ???????????? ???????????? encoding??? ?????? embedding??????
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # ???????????? ???????????? ???????????? 
        class_embedding = class_embeddings.mean(dim=0) # ????????? ?????? ???????????? embedding??? ????????????
        class_embedding /= class_embedding.norm() # ?
        zeroshot_weights.append(class_embedding) # ????????? ?????? ????????? ???????????? ??????
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device) # ????????????
    return zeroshot_weights

dataset_name='CIFAR10'
class_names = class_map[dataset_name] # dataset????????? ????????????. (MNIST or CIFAR)
templates = template_map[dataset_name]

cifar10 = CIFAR10(root="/data/cifar", download=False, train=False)
ood_dataset = SVHN(root='/data/svhn', split='test', download=False)

clean_logits = 0
ood_logits = 0


for dataset in [cifar10,ood_dataset]:
    # extract image feature, code borrowed from: https://github.com/openai/CLIP#zero-shot-prediction
    image_features = []
    image_labels = []

    for i,(image, class_id) in enumerate(tqdm(dataset)):
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
        image_feature /= image_feature.norm()
        image_features.append(image_feature)
        image_labels.append(class_id)
        if i == 9999:
            break
    image_features = torch.stack(image_features, dim=1).to(device)
    image_features = image_features.squeeze()

    dataset_name = 'CIFAR10'
    text_features = extract_text_features(dataset_name)
    
    # compute top-1 accuracy
    logits = (100. * image_features @ text_features).softmax(dim=-1)
    
    if dataset == cifar10:
        clean_logits = logits
    elif dataset == amb_dataset:
        amb_logits = logits
    else:
        ood_logits = logits

    image_labels = torch.tensor(image_labels).unsqueeze(dim=1).to(device)
    top1_acc = accuracy(logits, image_labels, (1,))
    print(f'top-1 accuracy for {dataset_name} dataset: {top1_acc[0]:.3f}')

clean_logits = clean_logits.cpu().numpy()
ood_logits = ood_logits.cpu().numpy()

# np.save('clean_RN50_CIFAR',clean_logits)
# np.save('ood_RN50_CIFAR',ood_logits)
# clean_logits_or = np.load('clean_logits_CIFAR_lets.npy')
# ood_logits_or = np.load('ood_logits_CIFAR_lets.npy')

make_figure('abc.png',clean_logits, ood_logits)


