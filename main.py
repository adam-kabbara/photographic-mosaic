import cv2
import random
import os
import numpy as np
import json
import tqdm
import time


def select_random_img(imgs_dir):
    walk = os.walk(imgs_dir)
    img_name = random.choice(list(walk)[0][2])
    img = cv2.imread(os.path.join(imgs_dir, img_name))
    return img


def get_pixel_median(img):
    # cv2 reads BGR and NOT RGB
    median = np.array([0, 0, 0])
    height, width = img.shape[:2]
    pixel_count = height * width
    for row in img:
        for pixel in row:
            median += pixel
    
    median = median // pixel_count
    return median


def resize_img(pixel_img, main_img):
    if type(pixel_img) == str:
        pixel_img = cv2.imread(pixel_img)

    w, h = main_img.shape[:2]
    pixel_img = cv2.resize(pixel_img, (w//6, h//8))
    return pixel_img


def map_imgs_to_pixels(imgs_dir, main_img):
    img_pixel_dict = {}
    walk = os.walk(imgs_dir)
    print("PROCESSING IMAGES")
    for img_name in tqdm.tqdm(list(walk)[0][2]):
        img = cv2.imread(os.path.join(imgs_dir, img_name))
        img = resize_img(img, main_img)
        img_color = get_pixel_median(img)
        img_color_str = ",".join(img_color.astype('str')) # since np array not hashable
        try:
            img_pixel_dict[img_color_str].append(img_name)
        except KeyError:
            img_pixel_dict[img_color_str] = [img_name]
    return img_pixel_dict


def aspect_ratio_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def find_closest_color(color, colors):
    b, g, r = color
    color_diffs = []
    for color in colors:
        cb, cg, cr = color
        color_diff = abs(b - cb) + abs(r - cr) + abs(g - cg)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def read_data(file):
    print("READING DATA")
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def write_data(data, file):
    print("SAVING DATA")
    with open(file, 'w') as f:
        json.dump(data, f)
    return 0


dataset_path = r"C:\Users\kabba\PythonProjects\python project\photographic mosaic\archive\raw-img\cane"
json_file = "data.json"
base_img = select_random_img(dataset_path)
base_img = aspect_ratio_resize(base_img, height=200)


if os.path.isfile(json_file):
    data = read_data(json_file)
else:
    data = map_imgs_to_pixels(dataset_path, base_img)
    write_data(data, json_file)

colors = [tuple(map(int, k.split(','))) for k in data.keys()]
pixel_imgs_matrix = []

start_time = time.time()
for i, row in enumerate(base_img):
    print(f'{i} out of {base_img.shape[0]-1}')
    lst = [] # use cv2 ig concatinate function
    for pixel in tqdm.tqdm(row):
        color = find_closest_color(pixel, colors)
        key = ",".join([str(i) for i in color])
        pixel_img = cv2.imread(os.path.join(dataset_path, random.choice(data[key])))
        lst.append(pixel_img)
    pixel_imgs_matrix.append(lst)

pixel_img_rows = []
for row in pixel_imgs_matrix:
    concat_row = hconcat_resize_min(row)
    pixel_img_rows.append(concat_row)

output = vconcat_resize_min(pixel_img_rows)

cv2.imwrite('output.jpg', output)
cv2.imwrite('input.jpg', base_img)

print(f"TIME ELAPSED: {time.time() - start_time}")
print("IMAGE COMPUTED")
