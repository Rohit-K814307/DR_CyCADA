import numpy as np
import torch
import torch.nn as nn
from data import create_dataset
from models import networks
from util import util
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval(G_A_path="checkpoints/drcycada_2_0/models/50_net_G_A.pth", 
         G_B_path="checkpoints/drcycada_2_0/models/50_net_G_B.pth",
         dataroot="./data/habitat_dataset_dr_cycada",
         model_name="drcycada",
         show_images=False):

    # load in models
    opt = lambda: None
    opt.model = "drcycada"
    opt.input_nc = 3
    opt.output_nc = 3
    opt.ngf = 64
    opt.netG = "resnet_2blocks"
    opt.norm = "instance"
    opt.no_dropout = 1
    opt.init_type = "normal"
    opt.init_gain = 0.02
    opt.gpu_ids = "0"

    netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    netG_A.load_state_dict(torch.load(G_A_path))
    netG_B.load_state_dict(torch.load(G_B_path))
    netG_A.to(device).eval()
    netG_B.to(device).eval()
    midas = torch.hub.load("intel-isl/MiDaS","DPT_Large").to(device)
    midas.eval()


    # load in data
    opt = lambda: None
    opt.dataroot = dataroot
    opt.no_flip = 1
    opt.dataset_mode = "class_unaligned"
    opt.phase = "test"
    opt.serial_batches = 1
    opt.direction = "BtoA"
    opt.input_nc = 3
    opt.output_nc = 3
    opt.batch_size = 1
    opt.load_size = 32
    opt.crop_size = 32
    opt.preprocess = "resize"
    opt.num_threads = 4
    opt.gpu_ids = "0"
    opt.max_dataset_size = 100000
    dataset = create_dataset(opt)


    loss_fn = nn.L1Loss()
    losses = {"iteration":[],"depth_loss_pred":[], "depth_loss_cycle":[]}


    print("\n\n\n\nNew Testing Session\n-------------------\n")

    # iter through data
    for i, data in enumerate(dataset):
        losses["iteration"].append(i)

        A = data["A"].detach().to(device)
        B = data["B"].detach().to(device)

        true_depth = midas(A).detach()

        A_hat = netG_A(B).detach()

        pred_depth = midas(A_hat).detach()

        B_hat = netG_A(A_hat).detach()

        B_depth = midas(B).detach()

        B_hat_depth = midas(B_hat).detach()

        #calculate depth loss
        depth_loss = loss_fn(true_depth, pred_depth)
        cycle_loss = loss_fn(B_depth, B_hat_depth)
        losses["depth_loss_pred"].append(depth_loss.item())
        losses["depth_loss_cycle"].append(cycle_loss.item())

        fig = plt.figure(figsize=(10, 10))
        rows = 1
        columns = 3


        #save actual A - ground truth source domain
        res = util.tensor2im(A)
        res = Image.fromarray(res)
        res = res.resize((512,128),Image.ANTIALIAS)
        ground_image_A = res
        ground_image_A.save("./results/" + model_name +"/images/ground_truth_source/{}.png".format(i))
        
        
        #save actual B - target domain
        res = util.tensor2im(B)
        res = Image.fromarray(res)
        res = res.resize((512,128),Image.ANTIALIAS)
        ground_image_B = res
        ground_image_B.save("./results/" + model_name +"/images/ground_truth_target/{}.png".format(i))
        
        

        #save predicted A - predicted source domain adaptation
        res = util.tensor2im(A_hat)
        res = Image.fromarray(res)
        res = res.resize((512,128),Image.ANTIALIAS)
        ground_image_A_hat = res
        ground_image_A_hat.save("./results/" + model_name +"/images/predicted_source_adapt/{}.png".format(i))

        if show_images:
            fig.add_subplot(rows, columns, 1)
            plt.imshow(ground_image_A)
            plt.axis('off')
            plt.title("Ground Truth A")

            fig.add_subplot(rows, columns, 2)
            plt.imshow(ground_image_B)
            plt.axis('off')
            plt.title("Ground Truth B")

            fig.add_subplot(rows, columns, 3)
            plt.imshow(ground_image_A_hat)
            plt.axis('off')
            plt.title("Predicted A")

            plt.show(block=False)
            plt.pause(1)

            plt.close()

        print(f"Iteration: {i}, Depth Loss: {depth_loss.item()}, Depth Cycle Loss: {cycle_loss.item()}")

    df = pd.DataFrame(losses)
    df.to_csv("./results/" + model_name + "/test_depth_retention_losses.csv", index=False)

def compare_losses():
    res_cycada = pd.read_csv("./results/cycada/test_depth_retention_losses.csv")
    res_drcycada = pd.read_csv("./results/drcycada/test_depth_retention_losses.csv")
    avg_depth_cycada = sum(res_cycada["depth_loss_pred"]) / len(res_cycada["depth_loss_pred"])
    avg_depth_drcycada = sum(res_drcycada["depth_loss_pred"]) / len(res_drcycada["depth_loss_pred"])

    print(f"CyCADA avg depth loss: {avg_depth_cycada}, DR-CyCADA avg depth loss: {avg_depth_drcycada}")


def evaluate_al():

    # eval cycada
    eval(G_A_path="../CyCADA/checkpoints/cycada_2_0/models/50_net_G_A.pth",
        G_B_path="../CyCADA/checkpoints/cycada_2_0/models/50_net_G_B.pth",
        model_name="cycada")

    # eval drcycada
    eval()

    #compare raw average depth retention losses (not fully representative but should be close to each other)
    compare_losses()