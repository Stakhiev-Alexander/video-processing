import argparse
import os
import numpy as np
from glob import glob
from DeepLab.modeling.deeplab import *
from DeepLab.dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from DeepLab.dataloaders.utils import *
from torchvision.utils import make_grid, save_image


IMG_EXTENTION = 'png'

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

parser.add_argument('--in-path', type=str, required=True, help='image to test')
parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
parser.add_argument('--dl-ckpt', type=str, default='./DeepLab/DLv3+torch.pth.tar', help='saved model')
args = parser.parse_args()

backbone = 'resnet'
class_number = 2
out_stride = 16
dataset = 'pascal'
crop_size = 513

model = DeepLab(num_classes=class_number, backbone=backbone, output_stride=out_stride, sync_bn=False)
ckpt = torch.load(args.dl_ckpt, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model.eval()

composed_transforms = transforms.Compose([
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor()])


# get input images from input folder
imgs_paths = sorted(glob(args.in_path + '*.' + IMG_EXTENTION))

# get first 3
imgs = [Image.open(path).convert('RGB') for path in imgs_paths[:3]]
imgs[0].save(args.out_path + '/' + '1'.zfill(10) + '.' + IMG_EXTENTION)

target = imgs[0]
sample = {'image': imgs, 'label': target}
tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
imgs = imgs.cuda()

with torch.no_grad():
    output = model(tensor_in)  # np.set_printoptions(threshold=np.inf)

print(output.shape)
grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3,
                        normalize=False, range=(0, 255))
print("type(grid) is: ", type(grid_image))
print("grid_image.shape is: ", grid_image.shape)
pred = output[0][1].numpy()
print(pred)
img_prob = []
print(pred.max())
pred -= pred.min()
pred *= 255.0 / pred.max()
pred = np.where(pred < 220, 0, 255)
print(pred.shape)
im = Image.fromarray(pred.astype('uint8'), 'L')
im.save(args.out_path)