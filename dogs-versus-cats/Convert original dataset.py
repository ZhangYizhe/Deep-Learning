import os, shutil
import Config as config

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(config.original_dataset_dir, fname)
    dst = os.path.join(config.train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(config.original_dataset_dir, fname)
    dst = os.path.join(config.validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(config.original_dataset_dir, fname)
    dst = os.path.join(config.test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(config.original_dataset_dir, fname)
    dst = os.path.join(config.train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(config.original_dataset_dir, fname)
    dst = os.path.join(config.validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(config.original_dataset_dir, fname)
    dst = os.path.join(config.test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(config.train_cats_dir)))
print('total training dog images:', len(os.listdir(config.train_dogs_dir)))
print('total validation cat images:', len(os.listdir(config.validation_cats_dir)))
print('total validation dog images:', len(os.listdir(config.validation_dogs_dir)))
print('total test cat images:', len(os.listdir(config.test_cats_dir)))
print('total test dog images:', len(os.listdir(config.test_dogs_dir)))