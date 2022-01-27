from torchvision.datasets import ImageFolder
import torch


def main():
    dataset = ImageFolder(root='/fs/cml-datasets/ImageNet/ILSVRC2012/train', transform=None)
    boxes = torch.load('boxes.pt')
    merged = {}
    for k, v in boxes.items():
        merged.update(v)

    indices = [i for i, (name, _) in enumerate(dataset.samples) if name.split('/')[-1] in merged]
    torch.save(indices, 'indices.pt')


if __name__ == '__main__':
    main()
