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
        # T.ToPILImage(),
        # T.Resize((256, 128), interpolation=3),
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

    def __init__(self, model, target_layers, printly):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.print = printly

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        for name, module in self.model._modules.items():
            if self.print:
                print('  ' + name)
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

    def __init__(self, model, feature_module, target_layers, name=None, printly=False):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers, printly)
        self.name = name

        self.target_layers = target_layers
        self.print = printly
        if self.print:
            print(target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():

            if self.print:
                print(name)
            # print(module)

            if name == 'base':
                if self.name == 'incep':
                    target_activations, x = self.feature_extractor(x)
                else:
                    for name2, module2 in module._modules.items():
                        if self.print:
                            print(' ' + name2)

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
            else:
                if module == self.feature_module:
                    target_activations, x = self.feature_extractor(x)
                elif "avgpool" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0), -1)
                elif "gap" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0), -1)
                elif "feat_bn" in name.lower() and self.name=='dense':
                    x = torch.cat([x, x], dim=1)
                    x = module(x)
                    x = x.view(x.size(0), -1)
                elif "feat_bn" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0), -1)
                else:
                    x = module(x)

        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, name=None, printly=False):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names, name=name, printly=printly)

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
    parser.add_argument('--image-path', type=str, default='./gradcam/imgs/market/0001_c1s1_001051_00.jpg',
                        help='Input image path')
    # parser.add_argument('--image-path', type=str, default='./gradcam/imgs/imgs/person1.jpg',
    #                     help='Input image path')
    parser.add_argument('--res', type=str, default='./gradcam/resnet_market.pth.tar')
    parser.add_argument('--den', type=str, default='./gradcam/densenet_market.pth.tar')
    parser.add_argument('--incep', type=str, default='./gradcam/inceptionv3_market.pth.tar')
    parser.add_argument('--printly', action='store_true')
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
    sys.stdout = Logger(osp.join('./gradcam', 'gradcam_log.txt'))
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)
    args.use_cuda = True

    if args.res == None:
        chk_res = './gradcam/resnet_market.pth.tar'
    else:
        chk_res = args.res

    if args.den == None:
        chk_den = './gradcam/densenet_market.pth.tar'
    else:
        chk_den = args.den

    if args.incep == None:
        chk_incep = './gradcam/inceptionv3_market.pth.tar'
    else:
        chk_incep = args.incep

    if chk_res.find('market') > -1:
        dataset = 'market'
        num_classes = 751
    else:
        dataset = 'duke'
        num_classes = 702

    name_res = chk_res.split('/')[-1].split('.')[0]
    name_den = chk_den.split('/')[-1].split('.')[0]
    name_incep = chk_incep.split('/')[-1].split('.')[0]

    ### Create model
    #### ResNet
    model_res = meb_models.create("resnet50", num_features=0, dropout=0, num_classes=num_classes)
    checkpoint = load_checkpoint(chk_res)
    copy_state_dict(checkpoint['state_dict'], model_res)
    model_res.cuda()
    grad_cam_res = GradCam(model=model_res, feature_module=model_res.base[6],
                           target_layer_names=["2"], use_cuda=args.use_cuda, printly=args.printly)

    #### DenseNet3
    model_den = meb_models.create("densenet", num_features=0, dropout=0, num_classes=num_classes)
    checkpoint = load_checkpoint(chk_den)
    copy_state_dict(checkpoint['state_dict'], model_den)
    model_den.cuda()
    grad_cam_den = GradCam(model=model_den, feature_module=model_den.base[0],
                           target_layer_names=["denseblock3"], name='dense', use_cuda=args.use_cuda, printly=args.printly)

    grad_cam_den4 = GradCam(model=model_den, feature_module=model_den.base[0],
                           target_layer_names=["denseblock4"], name='dense', use_cuda=args.use_cuda, printly=args.printly)

    #### InceptionV3
    model_incep = meb_models.create("inceptionv3", num_features=0, dropout=0, num_classes=num_classes)
    checkpoint = load_checkpoint(chk_incep)
    copy_state_dict(checkpoint['state_dict'], model_incep)
    model_incep.cuda()

    grad_cam_incep = GradCam(model=model_incep, feature_module=model_incep.base,
                           target_layer_names=["16"], name='incep', use_cuda=args.use_cuda, printly=args.printly)

    grad_cam_incep2 = GradCam(model=model_incep, feature_module=model_incep.base,
                           target_layer_names=["17"], name='incep', use_cuda=args.use_cuda, printly=args.printly)

    grad_cam_incep3 = GradCam(model=model_incep, feature_module=model_incep.base,
                           target_layer_names=["15"], name='incep', use_cuda=args.use_cuda, printly=args.printly)

    grad_cam_incep4 = GradCam(model=model_incep, feature_module=model_incep.base,
                           target_layer_names=["14"], name='incep', use_cuda=args.use_cuda, printly=args.printly)

    if args.image_path.split('.')[-1].lower() == 'jpg':
        img_names = args.image_path.split('/')[-1]
    else:
        img_names = os.listdir(args.image_path)

    # pdb.set_trace()
    for img_name in img_names:
        img_name = '{0}{1}'.format(args.image_path, img_name)
        name = img_name.split('/')

        img_type = name[-2]
        name = name[-1]
        name = name[:-4]

        dir_path = './gradcam/results_id/{img_type}_{name_res}_{name_den}_{name_incep}'.format(
            img_type=img_type, name_res=name_res, name_den=name_den, name_incep=name_incep)

        os.makedirs(dir_path, exist_ok=True)
        name = '{dir_path}/{name}'.format(
            name=name, dir_path=dir_path)

        img = cv2.imread(img_name, 1)
        # img = cv2.resize(img, dsize=(0, 0), fx=2, fy=2,
        #                  interpolation=cv2.INTER_LINEAR)

        img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
        img_clone = img.copy()
        img = np.float32(img) / 255

        # Opencv loads as BGR:
        img = img[:, :, ::-1]
        input_img = preprocess_image(img)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        target_category = None

        grayscale_cam_incep = grad_cam_incep(input_img[0], target_category)
        grayscale_cam_incep = cv2.resize(grayscale_cam_incep, (img.shape[1], img.shape[0]))
        cam_incep = show_cam_on_image(img, grayscale_cam_incep)

        grayscale_cam_incep2 = grad_cam_incep2(input_img[0], target_category)
        grayscale_cam_incep2 = cv2.resize(grayscale_cam_incep2, (img.shape[1], img.shape[0]))
        cam_incep2 = show_cam_on_image(img, grayscale_cam_incep2)

        grayscale_cam_incep3 = grad_cam_incep3(input_img[0], target_category)
        grayscale_cam_incep3 = cv2.resize(grayscale_cam_incep3, (img.shape[1], img.shape[0]))
        cam_incep3 = show_cam_on_image(img, grayscale_cam_incep3)

        grayscale_cam_incep4 = grad_cam_incep4(input_img[0], target_category)
        grayscale_cam_incep4 = cv2.resize(grayscale_cam_incep4, (img.shape[1], img.shape[0]))
        cam_incep4 = show_cam_on_image(img, grayscale_cam_incep4)

        grayscale_cam_res = grad_cam_res(input_img[0], target_category)
        grayscale_cam_res = cv2.resize(grayscale_cam_res, (img.shape[1], img.shape[0]))
        cam_res = show_cam_on_image(img, grayscale_cam_res)

        grayscale_cam_den = grad_cam_den(input_img[0], target_category)
        grayscale_cam_den = cv2.resize(grayscale_cam_den, (img.shape[1], img.shape[0]))
        cam_den = show_cam_on_image(img, grayscale_cam_den)

        grayscale_cam_den4 = grad_cam_den4(input_img[0], target_category)
        grayscale_cam_den4 = cv2.resize(grayscale_cam_den4, (img.shape[1], img.shape[0]))
        cam_den4 = show_cam_on_image(img, grayscale_cam_den4)

        addh = cv2.hconcat([img_clone, cam_res, cam_den,
                            cam_den4, cam_incep, cam_incep2,
                            cam_incep3, cam_incep4
                            ])

        cv2.imwrite("{0}.jpg".format(name), addh)
        print(' save {0}.jpg'.format(name))
        #########################
        #########################

        # pdb.set_trace()
        #
        # out_res = model_res(input_img[0].cuda())
        # out_den = model_den(input_img[0].cuda())
        # out_incep = model_incep(input_img[0].cuda())

        # gb_model_res = GuidedBackpropReLUModel(model=model_res, use_cuda=args.use_cuda)
        # gb = gb_model_res(input_img, target_category=target_category)
        # gb = gb.transpose((1, 2, 0))
        #
        # cam_mask = cv2.merge([grayscale_cam_res, grayscale_cam_res, grayscale_cam_res])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)
        #
        # addh = cv2.hconcat([img_clone, cam_res, cam_gb, gb])

        # cv2.imwrite("{0}.jpg".format(name), addh)
        # print(' save {0}.jpg'.format(name))

        # cv2.imwrite("{0}_cam.jpg".format(name), cam)
        # cv2.imwrite('{0}_gb.jpg'.format(name), gb)
        # cv2.imwrite('{0}_cam_gb.jpg'.format(name), cam_gb)
        #########################
        #########################


if __name__ == '__main__':
    main()
