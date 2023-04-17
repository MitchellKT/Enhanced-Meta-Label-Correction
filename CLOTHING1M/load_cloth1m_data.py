import argparse
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Clothing1M data preprocessing')
parser.add_argument('--root', default='data/clothing1m', type=str, help='Clothing1M dataset root directory path.')
args = parser.parse_args()

def create_img_folder(img_list, label_list, folder_name, root):
    # load label dict
    label_dict = {}
    with open(os.path.join(root, label_list), 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            label_dict[parts[0][7:]] = int(parts[1])

    # following previous works on cloth1m
    preprocess_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
    ])

    label_list = []

    # create folder if not exist
    if not os.path.isdir(os.path.join(root, folder_name)):
        os.mkdir(os.path.join(root, folder_name))
    
    cnt = 0
    name_set = set()
    with open(os.path.join(root, img_list), 'r') as f:
        for line in f:
            img_name = line.strip()[7:]
            label = label_dict[img_name]

            out_dir = os.path.join(root, folder_name, str(label))
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            trailing_name = img_name.split('/')[-1]
            assert trailing_name not in name_set, 'Image duplicates!'
            name_set.add(trailing_name)
            
            with Image.open(os.path.join(root, img_name)) as img:
                processed_img = preprocess_transform(img)

            processed_img.save(os.path.join(out_dir, trailing_name))
            cnt += 1

            if (cnt % 10000 == 0):
                print ('%d images processed' % cnt)

    print ('In total: %d images processed.' % cnt)

if __name__ == '__main__':
    create_img_folder('noisy_train_key_list.txt', 'noisy_label_kv.txt', 'noisy_train', root=args.root)
    create_img_folder('clean_train_key_list.txt', 'clean_label_kv.txt', 'clean_train', root=args.root)
    create_img_folder('clean_val_key_list.txt',   'clean_label_kv.txt', 'clean_val', root=args.root)
    create_img_folder('clean_test_key_list.txt',  'clean_label_kv.txt', 'clean_test', root=args.root)
