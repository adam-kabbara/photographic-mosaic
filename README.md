# Photographic Mosaic

Photographic Mosaic is, as the name suggests, is a python program that takes in a dataset of images and concatenates them to create a photographic mosaic (an image out of images)
## Logic Summary
In order to create a photographic mosaic, we need to map each pixel of the base image (the image we are trying to make out of many images) to an image from the dataset. To do this, we calculate the average RGB values of all images in the dataset and then map each pixel to the closes image RGB average

```python
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


def find_closest_color(color, colors):
    b, g, r = color
    color_diffs = []
    for color in colors:
        cb, cg, cr = color
        color_diff = abs(b - cb) + abs(r - cr) + abs(g - cg)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]
```

Combining these two functions can help us map each pixel value to an image from the dataset

## Efficiency
This program will be slower and slower as the images become larger and larger. So, the ```tqdm``` module has been used to show the progress of the program.

In addition, after calculating the average color of each image in the dataset, the data is saved in a JSON file so that the program won't have to calculate the same values over and over again each time it runs.

However, a drawback of this program is that the output image will be huge (>100 MB) if the input image was a bit large.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Image Examples
This example uses the [Animals-10](https://www.kaggle.com/alessiocorrado99/animals10) dataset obtained from [kaggle.com](kaggle.com)

<img width="430" alt="image" src="https://user-images.githubusercontent.com/62020687/157481223-91ce8f33-af5e-4222-a283-8597ef0e08cc.png">
<img width="960" alt="image" src="https://user-images.githubusercontent.com/62020687/157481457-67ff4bb3-8e94-4bf5-8485-bf0b40782a57.png">
