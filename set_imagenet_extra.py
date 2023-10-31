from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/datasets/ImageNet2012nonpub", extra="../imagenet_extras")
    dataset.dump_extra()
