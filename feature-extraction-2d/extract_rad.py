# %%
import os
import re
import json
from typing import Optional, Sequence
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import PIL
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import h5py
from sklearn.preprocessing import LabelBinarizer



__all__ = ['extract_features_']


class SlideTileDataset(Dataset):
    def __init__(self, slide_dir: Path, transform=None, *, repetitions: int = 1) -> None:
        self.tiles = list(slide_dir.glob('*.jpg'))
        assert self.tiles, f'no tiles found in {slide_dir}'
        self.tiles *= repetitions
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.open(self.tiles[i])
        if self.transform:
            image = self.transform(image)

        return image

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):


        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def _get_location(filename) -> Optional[np.ndarray]:
    # this function will work for files created with med2images with format filename-slice00X.jpg
    if matches := re.match(r'.*-slice(\d\d\d).jpg', str(filename)): 
        coords = matches.group(1)
        return coords
    else:
        print('Wrong filename', filename)
        return None


def extract_features_(
        *,
        model, model_name, slide_tile_paths: Sequence[Path], outdir: Path, 
        augmented_repetitions: int = 0, 
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """

    if model_name.endswith('-imagenet'):
        t_mean = [0.485, 0.456, 0.406]
        t_sd = [0.229, 0.224, 0.225]
    if model_name.endswith('-RADimagenet'):
        t_mean = [0.223, 0.223, 0.223]
        t_sd = [0.203, 0.203, 0.203]

    normal_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(t_mean,t_sd)])
    augmenting_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomRotation(90),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
    transforms.ToTensor(),
    transforms.Normalize(t_mean,t_sd),
    AddGaussianNoise(0., 1.)])


    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    extractor_string = model_name
    with open(outdir/'info.json', 'w') as f:
        json.dump({'extractor': extractor_string,
                  'augmented_repetitions': augmented_repetitions}, f)

    for slide_tile_path in tqdm(slide_tile_paths):
        slide_tile_path = Path(slide_tile_path)
        # check if h5 for slide already exists / slide_tile_path path contains tiles
        if (h5outpath := outdir/f'{slide_tile_path.name}.h5').exists():
            print(f'{h5outpath} already exists.  Skipping...')
            continue
        if not next(slide_tile_path.glob('*.jpg'), False):
            print(f'No tiles in {slide_tile_path}.  Skipping...')
            continue

        unaugmented_ds = SlideTileDataset(slide_tile_path, normal_transform)
        augmented_ds = SlideTileDataset(slide_tile_path, augmenting_transform,
                                        repetitions=augmented_repetitions)
    
        ds = ConcatDataset([unaugmented_ds, augmented_ds])
        dl = torch.utils.data.DataLoader(
            ds, batch_size=64, shuffle=False, num_workers=os.cpu_count(), drop_last=False)

        feats = []
        for batch in tqdm(dl, leave=False):
            feats.append(
                model(batch.type_as(next(model.parameters()))).half().cpu().detach())
        with h5py.File(h5outpath, 'w') as f:
            try:    
                f['location'] = [_get_location(fn) for fn in unaugmented_ds.tiles] + [_get_location(fn) for fn in augmented_ds.tiles]
                f['feats'] = torch.concat(feats).cpu().numpy()
                f['augmented'] = np.repeat(
                [False, True], [len(unaugmented_ds), len(augmented_ds)])
                f.attrs['extractor'] = extractor_string
            except:
                print('Error with file naming, no location given')
            


if __name__ == '__main__':
    import fire
    fire.Fire(extract_features_)

