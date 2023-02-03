

import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
# from datasets import Dataset
import pdb
import json
import random
import PIL.Image
import hashlib

class CustomDatasetForImages(Dataset):
  def __init__(self, directory, pp_fn=None, is_train=True):
    # directory containing the images
    self.directory = directory

    self.canonical_manifest = pd.read_csv(open(os.path.join(directory, 'canonical_manifest.csv')))
    self.generated_manifest = pd.read_csv(os.path.join(directory, 'generated_manifest.csv'))

    # Split based on person
    self.generated_manifest['is_train'] = self.generated_manifest.person_id.apply(lambda h: (int(hashlib.md5(h.encode('utf-8')).hexdigest(), 16) % 100) > 30)
    self.generated_manifest = self.generated_manifest[self.generated_manifest['is_train'] == is_train]
    split = 'TRAIN' if is_train else 'TEST'
    print(split, len(self.generated_manifest))
    # print(split, list(self.generated_manifest.person_id.values))
    self.pp_fn = pp_fn

    grouped = self.canonical_manifest.groupby('person_id').agg(list)['image_name']
    self.generated_manifest = pd.merge(self.generated_manifest, grouped, how='left', left_on='person_id', right_index=True)


  def __len__(self):
    return self.generated_manifest.shape[0]

  def __getitem__(self, idx, debug=False):

    # Fetch a generated image
    row = self.generated_manifest.iloc[idx]
    gen_path = os.path.join(self.directory, 'generated', row.person_id, f'{row.filename}.jpeg')
    gen_img = PIL.Image.open(gen_path)
    if debug:
      gen_img.save(f'debug/{row.filename}_gen.jpeg')

    # Fetch a corressponding canonical image.
    canon_name = random.choice(row.image_name)
    canon_path = os.path.join(self.directory, 'canonical', row.person_id, canon_name)
    canon_img = PIL.Image.open(canon_path)
    if debug:
      canon_img.save(f'debug/{row.filename}_can.jpeg')

    # TODO: variations on prompt?

    out = {
      'pixel_values': [gen_img],
      'cond_pixel_values': [canon_img],
      'input_ids': [row.no_person_prompt]
    }
    if self.pp_fn:
      return self.pp_fn(out)
    return out


if __name__ == '__main__':
  ds = CustomDatasetForImages('/home/james/data_v1.export', is_train=False)
  ds[2]

  os.makedirs('debug', exist_ok=True)
  for i in range(10):
    idx = random.choice(range(len(ds)))
    ds.__getitem__(idx, True)