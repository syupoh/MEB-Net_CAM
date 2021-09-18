from __future__ import print_function, absolute_import

import pdb
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter
from torch.autograd import Variable
import numpy as np


import cv2
from torchvision import transforms as T

# import os

# class Trainer_Unet(object):
#     def __init__(self, model, args):
#         super(Trainer_Unet, self).__init__()
#         self.model = model
#         self.criterion_mse = nn.MSELoss().cuda()
#         self.criterion_bce = nn.BCELoss().cuda()
#         self.batch_size = args.batch_size
#         self.type = args.type
#
#     def train(self, epoch, data_loader_source, data_loader_target,
#               optimizer, train_iters=200, print_freq=1, alpha=0.001, beta=1):
#
#         self.model.train()
#
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         losses_mse = AverageMeter()
#         losses_g = AverageMeter()
#         losses_r = AverageMeter()
#         losses_f = AverageMeter()
#         # precisions = AverageMeter()
#
#         end = time.time()
#         print_freq = int(train_iters/5)
#
#         Tensor = torch.cuda.FloatTensor
#
#         valid = Variable(Tensor(self.batch_size, 1).fill_(1.0), requires_grad=False)
#         fake = Variable(Tensor(self.batch_size, 1).fill_(0.0), requires_grad=False)
#
#         optimizerG = optimizer[0]
#         optimizerD = optimizer[1]
#
#         for i in range(train_iters):
#             data_time.update(time.time() - end)
#
#             source_inputs = data_loader_source.next()
#             s_inputs, targets = self._parse_data(source_inputs)
#
#             target_inputs = data_loader_target.next()
#             t_inputs, _ = self._parse_data(target_inputs)
#
#             # -----------------
#             #  Train Generator
#             # -----------------
#             # print('Train Generator')
#
#             optimizerG.zero_grad()
#
#             t_outputs, t_inputs_val = self.model(t_inputs)
#             _, t_outputs_val = self.model(t_outputs)
#             _, s_inputs_val = self.model(s_inputs)
#
#             loss_mse = self.criterion_mse(t_outputs, t_inputs)
#             losses_mse.update(loss_mse.item())
#
#             if self.type == 'D':
#                 loss = loss_mse
#             elif self.type == 'E':
#                 pass
#             elif self.type == 'F':
#                 loss_g = self.criterion_bce(self.model.d(s_inputs_val), valid)
#                 loss_g += self.criterion_bce(self.model.d(t_inputs_val), fake)
#                 loss_g *= 1/2
#                 losses_g.update(loss_g.item())
#             else:
#                 # loss_f = self.criterion_bce(self.model.d(t_inputs_val.detach()), fake)
#                 loss_g = self.criterion_bce(self.model.d(t_outputs_val), valid)
#
#                 loss = (1 - alpha) * loss_mse + (alpha) * loss_g
#                 losses_g.update(loss_g.item())
#
#             ##########
#             ##########
#             # print(self.model.conv_block4.layers[4].weight)
#             # print(self.model.conv_block4.layers[4].weight.grad)
#             # tempweight = self.model.conv_block4.layers[4].weight.clone()
#             ##########
#             ##########
#
#             if self.type == 'E':
#                 loss = loss_mse
#
#                 loss.backward()
#                 optimizerG.step()
#             elif self.type == 'F':
#                 loss = loss_g
#
#                 loss.backward()
#                 optimizerG.step()
#             else:
#                 loss.backward()
#                 # loss.backward(retain_graph=True)
#
#                 optimizerG.step()
#
#             ##########
#             ##########
#             # print(self.model.conv_block4.layers[4].weight)
#             # print(self.model.conv_block4.layers[4].weight.grad)
#             # print(tempweight - self.model.conv_block4.layers[4].weight)
#             ##########
#             ##########
#
#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#             # print('Train Discriminator')
#
#             optimizerD.zero_grad()
#
#             loss_r = self.criterion_bce(self.model.d(s_inputs_val.detach()), valid)
#
#             if self.type == 'A':
#                 loss_f = self.criterion_bce(self.model.d(t_inputs_val.detach()), fake)
#             elif self.type == 'B':
#                 loss_f = self.criterion_bce(self.model.d(t_outputs_val.detach()), fake)
#             elif self.type == 'C':
#                 loss_f = self.criterion_bce(self.model.d(t_outputs_val.detach()), fake)
#                 loss_f = loss_f + self.criterion_bce(self.model.d(t_inputs_val.detach()), fake)
#                 loss_f = loss_f * 0.5
#             elif self.type == 'D':
#                 loss_f = self.criterion_bce(self.model.d(t_inputs_val.detach()), fake)
#             elif self.type == 'E':
#                 loss_f = self.criterion_bce(self.model.d(t_inputs_val.detach()), fake)
#             elif self.type == 'F':
#                 loss_f = self.criterion_bce(self.model.d(t_inputs_val.detach()), fake)
#
#
#             ##########
#             # t_outputs, t_inputs_val = self.model(t_inputs)
#             #
#             # loss_r = self.criterion_bce(t_inputs_val, valid)
#             # loss_f = self.criterion_bce(s_outputs_val.detach(), fake)
#             ##########
#
#             loss_d = beta * (loss_r + loss_f)
#
#             losses_r.update(loss_r.item())
#             losses_f.update(loss_f.item())
#
#             tempweight = self.model.d.model[4].weight.clone()
#             # print(self.model.d.model[4].weight)
#             # print(self.model.d.model[4].weight.grad)
#             loss_d.backward()
#             optimizerD.step()
#             # print(self.model.d.model[4].weight)
#             # print(self.model.d.model[4].weight.grad)
#
#             # pdb.set_trace()
#             # self.model.d.model[4].weight - tempweight
#
#             batch_time.update(time.time() - end)
#             end = time.time()
#             # print('epoch-1')
#             if (i % print_freq == 0):
#               print('Epoch: [{}][{}/{}]\t'
#                     # 'Time {:.3f} ({:.3f})\t'
#                     # 'Data {:.3f} ({:.3f})\t'
#                     'Loss_mse {:.3f} ({:.3f})\t'
#                     'Loss_r {:.3f} ({:.3f})\t'
#                     'Loss_f {:.3f} ({:.3f})\t'
#                     'Loss_g {:.3f} ({:.3f})\t'
#                     # 'Loss_mse {:.3f} \t'
#                     # 'Loss_r {:.3f} \t'
#                     # 'Loss_f {:.3f} \t'
#                     # 'Loss_g {:.3f} \t'
#                     # 'Prec {:.2%} ({:.2%})\t'
#                     # 'Time {:.2f}'
#                     .format(epoch, i + 1, train_iters,
#                             # batch_time.val, batch_time.avg,
#                             # data_time.val, data_time.avg,
#                             losses_mse.val, losses_mse.avg,
#                             losses_r.val, losses_r.avg,
#                             losses_f.val, losses_f.avg,
#                             losses_g.val, losses_g.avg,
#                             # loss_mse,
#                             # loss_r,
#                             # loss_f,
#                             # loss_g,
#                             # precisions.val, precisions.avg
#                             # time.time() - start
#                             ))
#
#     def _parse_data(self, inputs):
#         imgs, _, pids, _ = inputs
#         inputs = imgs.cuda()
#         targets = pids.cuda()
#         return inputs, targets


def preprocess_image(img):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = T.Compose([
        # T.ToPILImage(),
        # T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalize,
    ])

    return preprocessing(img.copy()).unsqueeze(0),


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# class FeatureExtractor():
#     """ Class for extracting activations and
#     registering gradients from targetted intermediate layers """
#
#     def __init__(self, model, target_layers, printly):
#         self.model = model
#         self.target_layers = target_layers
#         self.gradients = []
#         self.print = printly
#
#     def save_gradient(self, grad):
#         self.gradients.append(grad)
#
#     def __call__(self, x):
#         outputs = []
#         self.gradients = []
#
#         for name, module in self.model._modules.items():
#             if self.print:
#                 print('  ' + name)
#             x = module(x)
#             if name in self.target_layers:
#                 x.register_hook(self.save_gradient)
#                 outputs += [x]
#
#         return outputs, x
#
#
# class ModelOutputs():
#     """ Class for making a forward pass, and getting:
#     1. The network output.
#     2. Activations from intermeddiate targetted layers.
#     3. Gradients from intermeddiate targetted layers. """
#
#     def __init__(self, model, feature_module, target_layers, name=None, printly=False):
#         self.model = model
#         self.feature_module = feature_module
#         self.feature_extractor = FeatureExtractor(self.feature_module, target_layers, printly)
#         self.name = name
#
#         self.target_layers = target_layers
#         self.print = printly
#         if self.print:
#             print(target_layers)
#
#     def get_gradients(self):
#         return self.feature_extractor.gradients
#
#     def __call__(self, x):
#         target_activations = []
#         for name, module in self.model._modules.items():
#
#             if self.print:
#                 print(name)
#             # print(module)
#
#             if name == 'base':
#                 if self.name == 'incep':
#                     target_activations, x = self.feature_extractor(x)
#                 else:
#                     for name2, module2 in module._modules.items():
#                         if self.print:
#                             print(' ' + name2)
#
#                         if module2 == self.feature_module:
#                             target_activations, x = self.feature_extractor(x)
#                         elif "avgpool" in name2.lower():
#                             x = module2(x)
#                             x = x.view(x.size(0), -1)
#                         elif "gap" in name2.lower():
#                             x = module2(x)
#                             x = x.view(x.size(0), -1)
#                         else:
#                             x = module2(x)
#             else:
#                 if module == self.feature_module:
#                     target_activations, x = self.feature_extractor(x)
#                 elif "avgpool" in name.lower():
#                     x = module(x)
#                     x = x.view(x.size(0), -1)
#                 elif "gap" in name.lower():
#                     x = module(x)
#                     x = x.view(x.size(0), -1)
#                 elif "feat_bn" in name.lower() and self.name=='dense':
#                     x = torch.cat([x, x], dim=1)
#                     x = module(x)
#                     x = x.view(x.size(0), -1)
#                 elif "feat_bn" in name.lower():
#                     x = module(x)
#                     x = x.view(x.size(0), -1)
#                 else:
#                     x = module(x)
#
#         return target_activations, x
#
#
# class GradCam:
#     def __init__(self, model, feature_module, target_layer_names, use_cuda, name=None, printly=False):
#         self.model = model
#         self.feature_module = feature_module
#         self.model.eval()
#         self.cuda = use_cuda
#         if self.cuda:
#             self.model = model.cuda()
#
#         self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names, name=name, printly=printly)
#
#     def forward(self, input_img):
#         return self.model(input_img)
#
#     def __call__(self, input_img, target_category=None):
#         if self.cuda:
#             input_img = input_img.cuda()
#
#         features, output = self.extractor(input_img)
#
#         if target_category == None:
#             target_category = np.argmax(output.cpu().data.numpy())
#
#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][target_category] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         if self.cuda:
#             one_hot = one_hot.cuda()
#
#         one_hot = torch.sum(one_hot * output)
#
#         self.feature_module.zero_grad()
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)
#
#         grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
#
#         target = features[-1]
#         target = target.cpu().data.numpy()[0, :]
#
#         weights = np.mean(grads_val, axis=(2, 3))[0, :]
#         cam = np.zeros(target.shape[1:], dtype=np.float32)
#
#         for i, w in enumerate(weights):
#             cam += w * target[i, :, :]
#
#         cam = np.maximum(cam, 0)
#         cam = cv2.resize(cam, input_img.shape[2:])
#         cam = cam - np.min(cam)
#         cam = cam / np.max(cam)
#         return cam

def compute_gradient_penalty(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class Trainer_Unet(object):
    def __init__(self, model, args, grayscale_cam_id=None):
        super(Trainer_Unet, self).__init__()
        self.model_unet = model[0]
        self.model_d = model[1]
        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_bce = nn.BCELoss().cuda()
        self.criterion_bce2 = nn.BCEWithLogitsLoss().cuda() # This loss combines a Sigmoid layer and BCELoss in one single class
        self.batch_size = args.batch_size
        self.type = args.type

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.delta = args.delta

        self.height = args.height
        self.width = args.width

        Tensor = torch.cuda.FloatTensor
        valid = Variable(Tensor(args.batch_size, 3, args.height, args.width).fill_(1.0), requires_grad=False)
        tempout = self.model_d(valid)
        self.valid_size = tempout.shape
        self.valid_size2 = self.model_d.output_shape
        self.gradcam = grayscale_cam_id

        # print(self.valid_size)
        # print(tempout[1].shape)
        # print(self.valid_size2)

        # img = cv2.imread(img_name, 1)
        # img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
        # img_clone = img.copy()
        # img = np.float32(img) / 255
        #
        # # Opencv loads as BGR:
        # img = img[:, :, ::-1]
        # input_img = preprocess_image(img)
        #
        # # If None, returns the map for the highest scoring category.
        # # Otherwise, targets the requested category.
        # target_category = None
        #
        # grayscale_cam_id = grad_cam_res(input_img[0], target_category)
        # grayscale_cam_id = cv2.resize(grayscale_cam_id, (img.shape[1], img.shape[0]))
        # # cam_res = show_cam_on_image(img, grayscale_cam_id)

    def train(self, epoch, data_loader_source, data_loader_target,
              optimizer, train_iters=200, print_freq=1, alpha=0.001, beta=1):
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        delta = self.delta

        self.model_unet.train()
        self.model_d.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_mse = AverageMeter()
        losses_g = AverageMeter()
        losses_d = AverageMeter()

        losses_r = AverageMeter()
        losses_f = AverageMeter()
        # precisions = AverageMeter()

        end = time.time()
        print_freq = int(train_iters/5)

        Tensor = torch.cuda.FloatTensor

        ### Cycle GAN
        valid = Variable(Tensor(self.valid_size).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(self.valid_size).fill_(0.0), requires_grad=False)

        optimizerG = optimizer[0]
        optimizerD = optimizer[1]

        for i in range(train_iters):
            data_time.update(time.time() - end)

            source_inputs = data_loader_source.next()
            s_inputs, targets = self._parse_data(source_inputs)

            target_inputs = data_loader_target.next()
            t_inputs, t_targets = self._parse_data(target_inputs)

            # -----------------
            #  Train Generator
            # -----------------
            ########## StarGAN Generator ##########
            # gen_imgs = generator(imgs, sampled_c)
            # recov_imgs = generator(gen_imgs, labels)
            # # Discriminator evaluates translated image
            # fake_validity, pred_cls = discriminator(gen_imgs)
            # # Adversarial loss
            # loss_G_adv = -torch.mean(fake_validity)
            # # Classification loss
            # loss_G_cls = criterion_cls(pred_cls, sampled_c)
            # # Reconstruction loss
            # loss_G_rec = criterion_cycle(recov_imgs, imgs)
            # # Total loss
            # loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec
            #
            # loss_G.backward()
            # optimizer_G.step()
            #########################################

            if self.gradcam is not None:
                for jk in range(len(t_inputs)):
                    mask = self.gradcam(t_inputs[jk].unsqueeze(0), t_targets[jk].cpu().numpy())
                    mask = torch.from_numpy(mask).reshape(t_inputs[jk].shape[1], t_inputs[jk].shape[2]).cuda()

                    # sum(sum((mask >= 0.0001).to(torch.int)))
                    # sum(sum((mask > 0).to(torch.int)))
                    mask = (mask * (mask <= delta).to(torch.float32)) + (mask > delta).to(torch.float32)

                    t_inputs[jk] = t_inputs[jk] * (mask > delta).to(torch.float32)

            optimizerG.zero_grad()

            t_outputs = self.model_unet(t_inputs)
            t_outputs_val = self.model_d(t_outputs)

            loss_mse = self.criterion_mse(t_outputs, t_inputs)
            losses_mse.update(loss_mse.item())

            if self.type == 'A':
                loss_g = self.criterion_bce2(t_outputs_val, valid)
                losses_g.update(loss_g.item())

                loss = (1 - alpha) * loss_mse + (alpha) * loss_g

            elif self.type == 'B':
                pass

            elif self.type == 'C':
                loss = loss_mse

            elif self.type == 'D':
                # StarGAN
                # t_outputs_val, _ = self.model_d(t_outputs)

                # t_outputs_val = torch.mean(t_outputs_val, dim=-1)
                # t_outputs_val = torch.mean(t_outputs_val, dim=-1)
                # t_outputs_val = torch.mean(t_outputs_val, dim=-1, keepdim=True)
                loss_g = -torch.mean(t_outputs_val)
                losses_g.update(loss_g.item())

                loss = (1 - alpha) * loss_mse + (alpha) * loss_g

            elif self.type == 'F':
                loss_g = self.criterion_bce2(t_outputs_val, valid)
                losses_g.update(loss_g.item())

                loss = (1 - alpha) * loss_mse + (alpha) * loss_g

            else:
                loss_g = self.criterion_bce2(t_outputs_val, valid)
                losses_g.update(loss_g.item())

                loss = (1 - alpha) * loss_mse + (alpha) * loss_g

            ##########
            ##########
            # print(self.model.conv_block4.layers[4].weight)
            # print(self.model.conv_block4.layers[4].weight.grad)
            # tempweight = self.model.conv_block4.layers[4].weight.clone()
            ##########
            ##########

            if self.type == 'B':
                pass

            else:
                loss.backward(retain_graph=True)
                optimizerG.step()

            ##########
            ##########
            # print(self.model.conv_block4.layers[4].weight)
            # print(self.model.conv_block4.layers[4].weight.grad)
            # print(tempweight - self.model.conv_block4.layers[4].weight)
            ##########
            ##########

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizerD.zero_grad()

            t_inputs_val = self.model_d(t_inputs)
            s_inputs_val = self.model_d(s_inputs)
            t_outputs_val = self.model_d(t_outputs.detach())

            if self.type == 'A':
                loss_r = self.criterion_bce2(s_inputs_val, valid)
                loss_f = self.criterion_bce2(t_inputs_val, fake)
            elif self.type == 'B':
                loss_r = self.criterion_bce(s_inputs_val, valid)
                loss_f = self.criterion_bce(t_inputs_val, fake)
            elif self.type == 'C':
                loss_r = self.criterion_bce2(s_inputs_val, valid)
                loss_f = self.criterion_bce2(t_inputs_val, fake)

            #StarGAN
            # t_inputs_val, _ = self.model_d(t_inputs)
            # s_inputs_val, _ = self.model_d(s_inputs)
            # t_outputs_val, _ = self.model_d(t_outputs.detach())

            # t_outputs_val = torch.mean(t_outputs_val, dim=-1)
            # t_outputs_val = torch.mean(t_outputs_val, dim=-1)
            # t_outputs_val = torch.mean(t_outputs_val, dim=-1, keepdim=True)
            #
            # t_inputs_val = torch.mean(t_inputs_val, dim=-1)
            # t_inputs_val = torch.mean(t_inputs_val, dim=-1)
            # t_inputs_val = torch.mean(t_inputs_val, dim=-1, keepdim=True)
            #
            # s_inputs_val = torch.mean(s_inputs_val, dim=-1)
            # s_inputs_val = torch.mean(s_inputs_val, dim=-1)
            # s_inputs_val = torch.mean(s_inputs_val, dim=-1, keepdim=True)

            # elif self.type == 'D':
            #     lambda_gp = 10

            #     # Gradient penalty
            #     gradient_penalty = compute_gradient_penalty(self.model_d, s_inputs, t_outputs)
            #     # Adversarial loss
            #     loss_d = -torch.mean(s_inputs_val) + torch.mean(t_outputs_val) + \
            #              lambda_gp * gradient_penalty
            #     losses_d.update(loss_d.item())

            elif self.type == 'E':
                loss_f = self.criterion_bce2(t_outputs_val, fake)
            elif self.type == 'F':
                loss_r = self.criterion_bce2(s_inputs_val, valid)
                loss_f = self.criterion_bce2(t_outputs_val, fake)
                loss_f = loss_f + self.criterion_bce2(t_inputs_val, fake)
                loss_f = loss_f * gamma

            ##############

            if self.type == 'C':
                pass
            elif self.type == 'D':

                # tempweight = self.model_d.model[4].weight.clone()
                # print(self.model_d.model[4].weight)
                # print(self.model_d.model[4].weight.grad)
                loss_d.backward()
                optimizerD.step()

            else:
                loss_d = beta * (loss_r + loss_f)

                losses_r.update(loss_r.item())
                losses_f.update(loss_f.item())

                # tempweight = self.model_d.model[4].weight.clone()
                # print(self.model_d.model[4].weight)
                # print(self.model_d.model[4].weight.grad)
                loss_d.backward()
                optimizerD.step()
                # print(self.model_d.model[4].weight)
                # print(self.model_d.model[4].weight.grad)

                # self.model_d.model[4].weight - tempweight

            batch_time.update(time.time() - end)
            end = time.time()
            if (i % print_freq == 0):
              print('Epoch: [{}][{}/{}]\t'
                    # 'Time {:.3f} ({:.3f})\t'
                    # 'Data {:.3f} ({:.3f})\t'
                    'Loss_mse {:.3f} ({:.3f})\t'
                    'Loss_r {:.3f} ({:.3f})\t'
                    'Loss_f {:.3f} ({:.3f})\t'
                    'Loss_g {:.3f} ({:.3f})\t'
                    'Loss_d {:.3f} ({:.3f})\t'
                    # 'Prec {:.2%} ({:.2%})\t'
                    # 'Time {:.2f}'
                    .format(epoch, i + 1, train_iters,
                            # batch_time.val, batch_time.avg,
                            # data_time.val, data_time.avg,
                            losses_mse.val, losses_mse.avg,
                            losses_r.val, losses_r.avg,
                            losses_f.val, losses_f.avg,
                            losses_g.val, losses_g.avg,
                            losses_d.val, losses_d.avg,
                            # precisions.val, precisions.avg
                            # time.time() - start
                            ))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0, AEtransfer=None):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.AEtransfer = AEtransfer

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        start = time.time()
        print_freq = int(train_iters/5)
        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)

            if self.AEtransfer:
                s_inputs = self.AEtransfer(s_inputs)

            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)

            ## target samples: only forward
            # t_features, _ = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      # 'Time {:.3f} ({:.3f})\t'
                      # 'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      # 'Prec {:.2%} ({:.2%})\t'
                      # 'Time {:.2f}'
                      .format(epoch, i + 1, train_iters,
                              # batch_time.val, batch_time.avg,
                              # data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg
                              # time.time() - start
                              ))
                start = time.time()

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class SingelmeanTrainer(object):
    def __init__(self, model_1, 
                       model_1_ema, num_cluster=500, alpha=0.999):
        super(SingelmeanTrainer, self).__init__()
        self.model_1 = model_1
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_1_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, targets = self._parse_data(target_inputs)

            # forward
            f_out_t1, p_out_t1 = self.model_1(inputs_1)

            f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)

            loss_ce_1 = self.criterion_ce(p_out_t1, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t1_ema)

            loss_tri_soft = self.criterion_tri_soft(f_out_t1, p_out_t1_ema, targets)

            loss = loss_ce_1*(1-ce_soft_weight) + \
                     loss_tri_1*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      # 'Time {:.3f} ({:.3f})\t'
                      # 'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              # batch_time.val, batch_time.avg,
                              # data_time.val, data_time.avg,
                              losses_ce[0].avg,
                              losses_tri[0].avg, 
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        targets = pids.cuda()
        return inputs_1, targets


# EBS + preference scatter
class MEBTrainer(object):
    def __init__(self, model_list, model_ema_list, num_cluster=500, alpha=0.999, scatter=[1,1,1], unet=None):
        super(MEBTrainer, self).__init__()
        self.models = model_list
        self.num_cluster = num_cluster
        self.model_num = len(self.models)
        self.model_emas = model_ema_list
        self.alpha = alpha
        self.scatter = F.normalize(torch.FloatTensor(scatter).cuda() ,p=1, dim=0)

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.unet = unet

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        for model in self.models:
            model.train()
        for model_ema in self.model_emas:
            model_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter() for i in range(self.model_num)]

        end = time.time()
        for iter_idx in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            if self.unet:
                for i in range(self.model_num):
                    inputs[i] = self.unet(inputs[i])

            # forward
            f_out_t = []
            p_out_t = []
            for i in range(self.model_num):
                f_out_t_i, p_out_t_i = self.models[i](inputs[i])
                f_out_t.append(f_out_t_i)
                p_out_t.append(p_out_t_i)

            f_out_t_ema = []
            p_out_t_ema = []
            for i in range(self.model_num):
                f_out_t_ema_i, p_out_t_ema_i = self.model_emas[i](inputs[i])
                f_out_t_ema.append(f_out_t_ema_i)
                p_out_t_ema.append(p_out_t_ema_i)

            
            authority_ce = []
            authority_tri = []

            loss_ce = loss_tri = 0
            for i in range(self.model_num):
                loss_ce += self.criterion_ce(p_out_t[i], targets)
                loss_tri += self.criterion_tri(f_out_t[i], f_out_t[i], targets)

                authority_ce.append(self.criterion_ce(p_out_t[i], targets))
                authority_tri.append(self.criterion_tri(f_out_t[i], f_out_t[i], targets))
            
            beta = 2
            authority = 3*self.scatter

            
            loss_ce_soft = loss_tri_soft = 0
            for i in range(self.model_num): #speaker
                for j in range(self.model_num): #listener
                    if i != j:
                        loss_ce_soft += 0.5*authority[i]*self.criterion_ce_soft(p_out_t[j], p_out_t_ema[i])
                        loss_tri_soft += 0.5*authority[i]*self.criterion_tri_soft(f_out_t[j], f_out_t_ema[i], targets)

            loss = loss_ce*(1-ce_soft_weight) + \
                     loss_tri*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for i in range(self.model_num):
                self._update_ema_variables(self.models[i], self.model_emas[i], self.alpha, epoch*len(data_loader_target)+iter_idx)

            prec_1, = accuracy(p_out_t[0].data, targets.data)
            prec_2, = accuracy(p_out_t[1].data, targets.data)
            prec_3, = accuracy(p_out_t[2].data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])
            precisions[2].update(prec_3[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (iter_idx + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      # 'Time {:.3f} ({:.3f})\t'
                      # 'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%} / {:.2%}\t'
                      .format(epoch, iter_idx + 1, len(data_loader_target),
                              # batch_time.val, batch_time.avg,
                              # data_time.val, data_time.avg,
                              losses_ce.avg, losses_tri.avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg, precisions[1].avg, precisions[2].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, imgs_3, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        inputs_3 = imgs_3.cuda()
        targets = pids.cuda()
        inputs_list = [inputs_1,inputs_2,inputs_3]
        return inputs_list, targets
