import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.utils.data as data
import numpy as np
from arch import *
import copy

# ======================== Dataset ========================

def load_data(dataset, data_path=None, batch_size=1):

    if dataset == 'stl10'and data_path==None:
        data_path = '../../datasets/stl10' 
    elif dataset == 'cifar10'and data_path==None:
        data_path = '../../datasets/cifar10'
    elif dataset == 'mnist'and data_path==None:
        data_path = '../../datasets/MNIST' 

    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset == 'stl10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.STL10(root=data_path, split ='test', download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    elif dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.MNIST(root=data_path, train=False, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return testloader, testset 

##========================================================================================
class BayesWrap(nn.Module):
    def __init__(self,net, num_particles):
        super().__init__()

        self.num_particles = num_particles
        self.h_kernel = 0
        self.particles = []

        for i in range(self.num_particles):
            self.particles.append(copy.deepcopy(net))
        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)

    def sample_particle(self):
        return self.particles[np.random.randint(0, len(self.particles))]

    def sample_particle_x(self, i):
        return self.particles[i]

    def get_particle(self, index):
        return self.particles[index]

    # ======== single forward ==========

    def sing_forward(self, x, **kwargs):
        idx =kwargs['idx']
        logits = self.particles[idx](x)

        return logits
    
    # ======== random forward  ==========

    def rnd_forward(self, x, **kwargs):
        logits = []
        n_particle =kwargs['n_selected']
        for i in range(n_particle):
            particle = self.sample_particle()
            l = particle(x)
            logits.append(l)
        logits = torch.stack(logits).mean(0)

        return logits
        
    # ============= full forward ===============
    def full_forward(self, x, **kwargs):
        logits, entropies = [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        for particle in self.particles:
            l = particle(x)
            logits.append(l)
            if return_entropy:
                l = torch.softmax(l, 1)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
        logits = torch.stack(logits).mean(0)

        return logits

    def forward(self,x,**kwargs):    
        if kwargs['pred_mode']=='single':
            output=self.sing_forward(x,**kwargs)
        elif kwargs['pred_mode']=='random':
            output=self.rnd_forward(x,**kwargs)
        elif kwargs['pred_mode'] =='full':
            output=self.full_forward(x,**kwargs)
        return output

class PretrainedModel():
    def __init__(self,model,
                 dataset='cifar10',
                 **kwargs):
        self.model = model
        self.dataset = dataset
        self.kwargs = kwargs
        #----------------------
        self.bounds =  [0,1]
        self.num_queries = 0        
        # ======= CIFAR10 ==========
        if self.dataset == 'cifar10': # 'ensemble': normal or dropout 
            self.mu = torch.Tensor([0., 0., 0.]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([1., 1., 1.]).float().view(1, 3, 1, 1).cuda()
            self.num_classes = 10
            
        # ======= SVHN ==========
        elif self.dataset == 'svhn': # 'ensemble': normal or dropout 
            self.mu = torch.Tensor([0., 0., 0.]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([1., 1., 1.]).float().view(1, 3, 1, 1).cuda()
            self.num_classes = 10
            
        # ======= STL10 ==========
        elif self.dataset == 'stl10':
            self.mu = torch.Tensor([0., 0., 0.]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([1., 1., 1.]).float().view(1, 3, 1, 1).cuda()
            self.num_classes = 10

        # ======= MNIST =========
        elif self.dataset == 'mnist':
            self.mu = torch.Tensor([0.]).float().view(1, 1, 1, 1).cuda()
            self.sigma = torch.Tensor([1.]).float().view(1, 1, 1, 1).cuda()          
            self.num_classes = 10

    def predict(self, x):
        
        img = (x - self.mu) / self.sigma
        with torch.no_grad():
            out = self.model(img,**self.kwargs)
            self.num_queries += x.size(0)
        return  out

    def predict_label(self, x):
        out = self.predict(x)
        out = torch.max(out,1)[1]
        return out

    def __call__(self, x):
        if self.kwargs['threat_model']=='decision-based':                
            out = self.predict_label(x)
        else:
            out = self.predict(x)

        return out

def load_model(dataset, num_particles):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dataset == "cifar10":
        NET = VGG("VGG16")
        NET = torch.nn.DataParallel(NET)
        
    elif dataset == 'mnist':
        NET = MNIST()
        NET = torch.nn.DataParallel(NET)
        
    elif dataset == 'stl10':
        NET = models.resnet18(pretrained=False)
        NET.fc = torch.nn.Linear(in_features=512, out_features=10)
        NET = torch.nn.DataParallel(NET)
        
    net = BayesWrap(NET,num_particles)
    net = net.to(device)
    if dataset == 'mnist':
        model_path = './models/shared_final_trained_model/svgd-sample_loss-mnist-40particles-ckpt_best.pth' 

    elif dataset == 'cifar10':
        model_path = './models/shared_final_trained_model/svgd-sample_loss-cifar10-10particles-ckpt_best.pth'
        
    elif dataset == 'stl10':
        model_path = './models/shared_final_trained_model/svgd-sample_loss-stl10-10particles-ckpt_best.pth'  
        
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    
    return net