import torch
import torchvision.transforms as transforms


class PrefetchLoader:
    """A data loader wrapper for prefetching data along with
    `ToTensor` and `Normalize` transformations.

    Borrowed from https://github.com/open-mmlab/OpenSelfSup.
    """

    def __init__(self, loader, mean, std):
        self.loader = loader
        self._mean = mean
        self._std = std

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        self.mean = torch.tensor([x * 255 for x in self._mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in self._std]).cuda().view(1, 3, 1, 1)

        for next_item in self.loader:
            with torch.cuda.stream(stream):
                if "img" in next_item:
                    img = next_item["img"].cuda(non_blocking=True)
                    next_item["img"] = img.float().sub_(self.mean).div_(self.std)
                else:
                    # Semi-supervised loader
                    img1 = next_item["img1"].cuda(non_blocking=True)
                    img2 = next_item["img2"].cuda(non_blocking=True)
                    next_item["img1"] = img1.float().sub_(self.mean).div_(self.std)
                    next_item["img2"] = img2.float().sub_(self.mean).div_(self.std)

            if not first:
                yield item
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            item = next_item

        yield item

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def prefetch_transform(transform):
    """Remove `ToTensor` and `Normalize` in `transform`.
    """
    transform_list = []
    normalize = False
    for t in transform.transforms:
        if "Normalize" in str(type(t)):
            normalize = True
    if not normalize:
        raise KeyError("No Normalize in transform: {}".format(transform))
    for t in transform.transforms:
        if not ("ToTensor" or "Normalize" in str(type(t))):
            transform_list.append(t)
        if "Normalize" in str(type(t)):
            mean, std = t.mean, t.std
    transform = transforms.Compose(transform_list)

    return transform, mean, std
