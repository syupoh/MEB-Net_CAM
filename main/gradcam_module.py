import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from torchvision import transforms as T

import pdb
import os
import os.path as osp
import sys
from mebnet import models as meb_models
from mebnet.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from mebnet.utils.logging import Logger


def preprocess_image(img):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    preprocessing = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    return preprocessing(img.copy()).unsqueeze(0),


def preprocess_image2(img):
    Tresize = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128), interpolation=3),
    ])

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    preprocessing = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    resized_img = Tresize(img)

    return preprocessing(resized_img).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            # print(name)
            # print(module)

            if name == 'base':
                for name2, module2 in module._modules.items():
                    if module2 == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    elif "avgpool" in name2.lower():
                        x = module2(x)
                        x = x.view(x.size(0), -1)
                    elif "gap" in name2.lower():
                        x = module2(x)
                        x = x.view(x.size(0), -1)
                    else:
                        x = module2(x)
            elif name.find('conv_block') > -1 or name == 'd' or name == 'output_block':
                # if module == self.feature_module:
                #     target_activations, x = self.feature_extractor(x)
                if name == 'conv_block1':
                    x, c1 = module(x)
                elif name == 'conv_block2':
                    x, c2 = module(x)
                elif name == 'conv_block3':
                    x, c3 = module(x)
                elif name == 'conv_block4':
                    validity, c4 = module(x)
                    target_activations, x = self.feature_extractor(x)
                    validity = validity.view([validity.shape[0], -1])

                elif name == 'd':
                    pass
                elif name == 'deconv_block1':
                    d1 = module(x)
                elif name == 'deconv_block2':
                    d2 = module(torch.cat((c4, d1), dim=1))
                elif name == 'deconv_block3':
                    d3 = module(torch.cat((c3, d2), dim=1))
                elif name == 'deconv_block4':
                    d4 = module(torch.cat((c2, d3), dim=1))
                elif name == 'output_block':
                    out = module(torch.cat((c1, d4), dim=1))
                else:
                    x = module(x)
            else:
                if module == self.feature_module:
                    target_activations, x = self.feature_extractor(x)
                elif "avgpool" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0), -1)
                elif "gap" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0), -1)
                else:
                    x = module(x)

        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img[0].cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./gradcam/imgs/person1.jpg',
                        help='Input image path')
    parser.add_argument('--model', type=str, default='./gradcam/resnet_market.pth.tar')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    args.use_cuda = True
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def main():
    """ python grad_cam.py <path_to_image>
       1. Loads an image with opencv.
       2. Preprocesses it for VGG19 and converts to a pytorch variable.
       3. Makes a forward pass to find the category index with the highest score,
       and computes intermediate activations.
       Makes the visualization. """

    chk_incep = './gradcam/inceptionv3_market.pth.tar'
    chk_den = './gradcam/densenet_market.pth.tar'
    chk_unet = './logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_B_2021-04-08T03:54/checkpoint_79.pth.tar'

    sys.stdout = Logger(osp.join('./gradcam', 'gradcam_log.txt'))
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)
    args.use_cuda = True

    # name = args.image_path
    # name = name.split('/')
    # name = name[-1]
    # name = name[:-4]
    # name = './gradcam/results/{0}'.format(name)

    if args.model == None:
        chk_res = './gradcam/resnet_market.pth.tar'
    else:
        chk_res = args.model

    if chk_res.find('market') > -1:
        dataset = 'market'
        num_classes = 751
    else:
        dataset = 'duke'
        num_classes = 702

    model_path = chk_res

    name = args.image_path
    name = name.split('/')

    img_type = name[-2]
    name = name[-1]
    name = name[:-4]

    model_path = model_path.split('/')
    model_path = model_path[-1]
    model_path = model_path.split('.')
    model_path = model_path[0]

    dir_path = './gradcam/results_id/{img_type}_{model_path}'.format(
        img_type=img_type, model_path=model_path)

    os.makedirs(dir_path, exist_ok=True)
    name = '{dir_path}/{name}'.format(
        name=name, dir_path=dir_path)

    # print(args.image_path)
    img = cv2.imread(args.image_path, 1)

    img_clone = img.copy()
    img = np.float32(img) / 255

    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)
    resized_img = preprocess_image2(img_clone)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # Create model

    # Duke
    model_res = meb_models.create("resnet50", num_features=0, dropout=0, num_classes=num_classes)

    checkpoint = load_checkpoint(chk_res)
    copy_state_dict(checkpoint['state_dict'], model_res)

    grad_cam_res = GradCam(model=model_res, feature_module=model_res.base[6],
                           target_layer_names=["2"], use_cuda=args.use_cuda)
    model_res.cuda()

    grayscale_cam_res = grad_cam_res(input_img[0], target_category)
    #########################
    #########################

    grayscale_cam_res = cv2.resize(grayscale_cam_res, (img.shape[1], img.shape[0]))

    pdb.set_trace()
    cam = show_cam_on_image(img, grayscale_cam_res)

    gb_model_res = GuidedBackpropReLUModel(model=model_res, use_cuda=args.use_cuda)
    gb = gb_model_res(input_img, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam_res, grayscale_cam_res, grayscale_cam_res])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    # addh = cv2.createMat
    # cv2.hconcat([input_img, cam, cam_gb, gb], addh)
    # pdb.set_trace()

    addh = cv2.hconcat([img_clone, cam, cam_gb, gb])

    cv2.imwrite("{0}.jpg".format(name), addh)
    print(' save {0}.jpg'.format(name))

    # cv2.imwrite("{0}_cam.jpg".format(name), cam)
    # cv2.imwrite('{0}_gb.jpg'.format(name), gb)
    # cv2.imwrite('{0}_cam_gb.jpg'.format(name), cam_gb)


if __name__ == '__main__':
    main()
