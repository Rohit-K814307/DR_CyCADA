import os, sys
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class CyCADAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_sem_A', type=float, default=1.0, help='weight for semantic loss (A -> B -> A)')
            parser.add_argument('--lambda_sem_B', type=float, default=1.0, help='weight for semantic loss (B -> A -> B)')
            parser.add_argument('--c_lr', type=float, default=0.0001, help='learning rate of classifier')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.score_names = ['acc_real_A', 'acc_fake_B', 'acc_rec_A', 'acc_real_B', 'acc_fake_A', 'acc_rec_B', 'acc_depth_A','acc_depth_B']
        # self.score_acc_real_A = 1.0
        # self.score_acc_real_B = 1.0
        # self.score_acc_fake_A = 1.0
        # self.score_acc_fake_B = 1.0
        # self.score_acc_rec_A = 1.0
        # self.score_acc_rec_B = 1.0
        
        
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B', 'depth_A', 'depth_B']
        if self.isTrain:
            # specify the training losses you want to print out. The program will call base_model.get_current_losses
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B','depth_A', 'depth_B']
            # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            if self.opt.monitor_gnorm:
                self.gnorm_names = self.model_names
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCLS = torch.nn.modules.CrossEntropyLoss()
            self.Batch_MSE = lambda a, b: torch.nn.MSELoss(reduction='none')(a.view(a.size()[0], -1),b.view(b.size()[0], -1)).mean(dim=1)
            self.Batch_L1 = lambda a, b: torch.nn.L1Loss(reduction='none')(a.view(a.size()[0], -1),b.view(b.size()[0], -1)).mean(dim=1)
            self.depth_loss = torch.nn.L1Loss()
            self.midas = torch.hub.load("intel-isl/MiDaS","MiDaA_small").to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr/2., betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_pretrain_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.target_A = input['A_target' if AtoB else 'B_target'].to(self.device)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.target_A = input['A_target' if AtoB else 'B_target'].to(self.device)
        self.target_B = input['B_target' if AtoB else 'A_target'].to(self.device)
        #use for feature discriminator
        self.source_label = torch.ones(self.real_A.size()[0]).long().to(self.device)
        self.target_label = torch.zeros(self.real_B.size()[0]).long().to(self.device)
    
    def midas_pred(self, imr):
        img = imr.clone().detach()
        for i in range(img.size()[0]):
            img[i] = img[i] / torch.max(img[i])
        
        return self.midas(img).detach()
    

    def midas_depth(self, imr):
        img = imr.clone()
        prediction = self.midas_pred(img).cpu()
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=[128,128],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return prediction

    def forward(self):
        self.fake_A = self.netG_B(self.real_B) # G_B(B)
        self.fake_B = self.netG_A(self.real_A) # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        self.depth_A = self.midas_depth(self.fake_A.detach())
        self.depth_B = self.midas_depth(self.fake_B.detach())

        self.score_acc_real_A = 1.0
        self.score_acc_real_B = 1.0
        self.score_acc_fake_A = (self.fake_A.squeeze().cpu() == self.real_A.squeeze().cpu()).sum().item()/4
        self.score_acc_fake_B = (self.fake_B.squeeze().cpu() == self.real_B.squeeze().cpu()).sum().item()/4
        self.score_acc_rec_A = (self.rec_A.squeeze().cpu() == self.real_A.squeeze().cpu()).sum().item()/4
        self.score_acc_rec_B = (self.rec_B.squeeze().cpu() == self.real_B.squeeze().cpu()).sum().item()/4
        
        
        self.loss_depth_A = self.depth_loss(self.midas_pred(self.fake_A.detach()), self.midas_pred(self.real_A.detach())).sum().item()/4
        self.loss_depth_B = self.depth_loss(self.midas_pred(self.fake_B.detach()), self.midas_pred(self.real_B.detach())).sum().item()/4
        self.score_acc_depth_A = self.loss_depth_A
        self.score_acc_depth_B = self.loss_depth_B
        
        

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        if self.opt.gan_mode == "wgangp":
            gradient_penalty, gradients = networks.cal_gradient_penalty(netD,real,fake,self.device)
            gradient_penalty.backward(retain_graph=True)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        else:
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # Calculate discriminator accuracy
        # true_labels = torch.ones(real.size()[0]).long()
        # fake_labels = torch.zeros(fake.detach().size()[0]).long()
        # _, true_acc = networks.prediction(pred_real.squeeze().cpu(), true_labels, onehot=False)
        # _, fake_acc = networks.prediction(pred_fake.squeeze().cpu(), fake_labels, onehot=False)
        acc = 0.5
        return loss_D, acc

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_basic_A, self.score_acc_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        self.loss_D_A = self.loss_D_basic_A
        self.loss_D_A.backward()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_basic_B, self.score_acc_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
        self.loss_D_B = self.loss_D_basic_B
        self.loss_D_B.backward()

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        # combined loss standard cyclegan
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        # clip gradient of G, D, C
        if self.opt.monitor_gnorm:
            self.gnorm_G_A = torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), self.opt.max_gnorm)
            self.gnorm_G_B = torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), self.opt.max_gnorm)
            self.gnorm_D_A = torch.nn.utils.clip_grad_norm_(self.netD_A.parameters(), self.opt.max_gnorm)
            self.gnorm_D_B = torch.nn.utils.clip_grad_norm_(self.netD_B.parameters(), self.opt.max_gnorm)