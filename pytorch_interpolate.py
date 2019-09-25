from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
im = Image.open('epoch003_sparse_real_A.png')

im_t = transforms.ToTensor()(im)
im_t = im_t[0]
im_t=im_t.unsqueeze(0)
im_t=im_t.unsqueeze(0)
im_small_no_align = F.interpolate(im_t,scale_factor=0.25,mode='bilinear',align_corners=False)
#im_small_no_align =

#im_small_no_align = im_small_no_align.squeeze()
im_small_no_align = im_small_no_align.resize_(1,32,32) #im_small_no_align.squeeze()
im_small_no_align.expand(3,32,32)
im_small_no_align_img = transforms.ToPILImage()(im_small_no_align).convert("RGB")
im_small_no_align_img.save('small_no_align.png')


im_small = F.interpolate(im_t,scale_factor=0.25,mode='bilinear',align_corners=True)

im_small = im_small.resize_(1,32,32) #im_small_no_align.squeeze()
im_small.expand(3,32,32)
im_small_img = transforms.ToPILImage()(im_small) .convert("RGB")
im_small_img.save('small.png')

im_small_nearest = F.interpolate(im_t,scale_factor=0.25,mode='nearest')

im_small_nearest = im_small_nearest.resize_(1,32,32) #im_small_no_align.squeeze()
im_small_nearest.expand(3,32,32)
im_small_nearest_img = transforms.ToPILImage()(im_small_nearest) .convert("RGB")
im_small_nearest_img.save('small_nearest.png')
