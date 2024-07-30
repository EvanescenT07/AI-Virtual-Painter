import urllib.request

def download_all_classes():
    # List of all class names in the Quick, Draw! dataset
    all_classes = [
        'car', 'fish', 'flower', 'moon', 'mountain', 'pencil', 'star', 'sun', 'cloud', 'lightning'
    ]

    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

    for c in all_classes:
        cls_url = c.replace('_', '%20')
        path = base + cls_url + '.npy'
        print(f'Downloading {c}...')
        urllib.request.urlretrieve(path, 'Dataset/' + c + '.npy')
        print(f'{c} downloaded.')

download_all_classes()