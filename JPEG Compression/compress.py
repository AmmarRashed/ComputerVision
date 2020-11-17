# Ammar Raşid S017288 Graduate School of Engineering and Science
import time

import numpy as np
import copy
import os, ntpath
import cv2 as cv
import matplotlib.pyplot as plt

from argparse import ArgumentParser

raw_Q50 = \
"""
6 11 10 16 24 40 51 61
12 12 14 19 26 58 60 55
14 13 16 24 40 57 69 56
14 17 22 29 51 87 80 62
18 22 37 56 68 109 103 77
24 35 55 64 81 104 113 92
49 64 78 87 103 121 120 101
72 92 95 98 112 100 103 99
"""

raw_Q90 = \
"""
3 2 2 3 5 8 10 12
2 2 3 4 5 12 12 11
3 3 3 5 8 11 14 11
3 3 4 6 10 17 16 12
4 4 7 11 14 22 21 15
5 7 11 13 16 12 23 18
10 13 16 17 21 24 24 21
14 18 19 20 22 20 20 20
"""

raw_Q10 = \
"""
80 60 50 80 120 200 255 255
55 60 70 95 130 255 255 255
70 65 80 120 200 255 255 255
70 85 110 145 255 255 255 255
90 110 185 255 255 255 255 255
120 175 255 255 255 255 255 255
245 255 255 255 255 255 255 255
255 255 255 255 255 255 255 255
"""


def raw2array(table):
    return np.array(
    [row.split() for row in table.split("\n")[1:-1]], dtype=np.int8)


def quantize(blocks, quantization_matrix):
    return {idx: np.round(block/quantization_matrix) for idx, block in blocks.items()}


def pad(img, block_size):
    """
    :param img: numpy nd_array image tensor
    :param block_size: int
    :return: padded image (using BORDER REPLICATE)
    """
    h, w = img.shape
    hpad = int(np.ceil(h%block_size / 2.))
    wpad = int(np.ceil(w%block_size / 2.))
    return cv.copyMakeBorder(img, hpad, hpad, wpad, wpad, cv.BORDER_REPLICATE)


def img2blocks(img, block_size):
    img = pad(img, block_size)
    h, w = img.shape
    blocks = dict()
    for r, i in enumerate(range(0, h, block_size)):
        for c, j in enumerate(range(0, w, block_size)):
            blocks[(r, c)] = np.array(img[i:i + block_size, j:j + block_size], dtype=np.float32)

    try:
        assert len(blocks) == h*w/(block_size**2)
    except AssertionError:
        print(h, w)
        raise Exception("Number of blocks does not match the expected value")

    return blocks


def blocks2img(blocks, dtype=np.int16):
    block_size = len(blocks[(0, 0)])
    h = block_size * (max(blocks, key=lambda x: x[0])[0] + 1)
    w = block_size * (max(blocks, key=lambda x: x[1])[1] + 1)

    img = np.zeros(shape=(h, w), dtype=dtype)
    for (x, y), data in blocks.items():
        img[x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = data
    return img


def abbreviate(text):
    return "".join([c[0] for c in text.split() if c[0].isupper()])


def get_sizes(imgs_dict):

    """
    :param dict of quantization type and the corresponding (de)compression images
            i.e. {Quantization type: [dct image(compressed), idct image (decompressed)]
    :return: dict of quantization type and the corresponding (de)compression output sizes
    """
    return {q: [get_img_size(img) for img in imgs]
    for q, imgs in imgs_dict.items()}


def plot_sizes(imgsize, sizes_dict):
    """
    :param imgsize: image size (before compression)
    :param sizes_dict: dict {Quantization type: [dct file (compressed) size, idct file (decompressed) size]
    """
    x = np.arange(2)
    xticks = ["Decompressed", "Compressed"]
    imgname = ntpath.basename(imgpath)
    title = "File Sizes"
    figname = "{0}: {1}".format(abbreviate(title), imgname)
    plt.figure(figsize=(12, 8), num=figname)
    plt.suptitle(title, fontsize=22)
    pad = -0.2
    plt.plot([imgsize]*2, label='Pre-Compressino size')
    plt.text(.5, imgsize*1.01, "{0:,} KB".format(int(imgsize)))
    for q, sizes in sizes_dict.items():
        plt.bar(x + pad, sizes, width=0.2, label=q)
        for i, s in enumerate(sizes):
            if s != 0:
                plt.text(i + pad - 0.05, s * 1.01, "{0:,} KB".format(int(s)))

        plt.xticks(x, xticks)
        pad += 0.2

    plt.ylabel("Size in (KB)")
    plt.legend()
    plt.savefig(os.path.join("output", figname), format='png')


def get_img_size(img_array):
    img_array.dtype = np.int8
    np.savez_compressed('temp', img_array)
    img_size = os.path.getsize('temp.npz')//1e3
    os.remove('temp.npz')

    return img_size

def analysis_pipeline(imgpath):
    """
    :param imgpath: path to input image
    :param block_size: block size
    :return: img_size: the image size in kilobytes
    imgs_dict: dict of quantization type and the corresponding (de)compression images
            i.e. {Quantization type: [dct image(compressed), idct image (decompressed)]

        size_dict: dict {Quantization type: [dct file (compressed) size, idct file (decompressed) size]
    """
    block_size = 8

    if not os.path.isdir("output"):
        os.makedirs("output")

    img_array = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
    img_size = get_img_size(img_array)
    if img_array is None:
        raise Exception("Image not found")
    # Dividing the image into blocks
    blocks = img2blocks(img_array, block_size)

    # Applying DCT
    dct_blocks = {block_idx: cv.dct(block_data) for block_idx, block_data in blocks.items()}
    dct_img = blocks2img(dct_blocks)

    # Decompression (no quantization)
    idct_img = blocks2img({idx:cv.idct(block) for idx, block in dct_blocks.items()})

    # Quantization
    imgs_dict = {"None": [idct_img, dct_img]}

    for q in ["Q10", "Q50", "Q90"]:
        q_blocks = quantize(dct_blocks, globals()[q])
        q_dct_img = blocks2img(q_blocks)

        # Applying Inverse DCT (Decompression)
        q_idct_blocks = {idx:cv.idct(block) for idx, block in q_blocks.items()}
        q_idct_img = blocks2img(q_idct_blocks)

        imgs_dict[q] = (q_idct_img, q_dct_img)

    sizes_dict = get_sizes(copy.deepcopy(imgs_dict))
    return img_size, imgs_dict, sizes_dict


def plot_img_list(imgs, nrows, title="", imgname='', cmap='gray', abbreviate_title=True):
    """
    imgs: dict {title: image tensor}
    nrows: number of rows of subplots
    """
    ncols = int(np.ceil(len(imgs)/nrows))

    figname = "{0}: {1}".format(abbreviate(title) if abbreviate_title else title, imgname)

    plt.figure(figsize=(20, 7), num=figname)

    for i, (title, tensor) in enumerate(imgs.items(), 1):
        if i % 2 == 0:
            idx = len(imgs)//2 + i//2
        else:
            idx = (i+1)//2
        ax = plt.subplot(nrows, ncols, idx)
        ax.set_title(title)
        ax.imshow(tensor, cmap, aspect='equal')

    plt.savefig(os.path.join("output", figname), format='png')



def plot_fft_magnitude_spectrums(imgs_dict, imgname="Image"):
    """
    :param imgs_dict: dict of quantization type and the corresponding (de)compression images
        i.e. {Quantization type: [dct image(compressed), idct image (decompressed)]
    :param imgname: imagename
    """
    img2fft = lambda img: np.fft.fft2(img)
    shift_fft = lambda fft_vals: np.fft.fftshift(fft_vals)
    fft2MS = lambda fshift: 20 * np.log(np.abs(fshift))  # MS: Magnitued Spectrum

    img2MS = lambda img: fft2MS(
        shift_fft(
            img2fft(img)))  # MS: Magnitued Spectrum

    magnitude_spectrums = dict()

    for q, (idct_img, dct_img) in imgs_dict.items():
        if idct_img is None:
            raise Exception("Image not found")

        idct_title = "IDCT "+q
        dct_title = "DCT "+q
        if q == 'None':
            idct_title = "Original"
            dct_title = "DCT no Quantization"
        magnitude_spectrums[idct_title] = img2MS(idct_img)
        magnitude_spectrums[dct_title] = img2MS(dct_img)


    plot_img_list(magnitude_spectrums, nrows=2,
              title="Magnitude Spectrums in Frequency Domain", imgname=imgname)



Q50 = raw2array(raw_Q50)
Q90 = raw2array(raw_Q90)
Q10 = raw2array(raw_Q10)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("p", help="Images paths", nargs="+")
    parser.add_argument("-s", help="Show plots", default=True, type=int)
    args = parser.parse_args()
    start = time.time()
    for imgpath in args.p:
        imgname = ntpath.basename(imgpath)
        img_size, imgs_dict, sizes_dict = analysis_pipeline(imgpath)
        plot_fft_magnitude_spectrums(imgs_dict, imgname=imgname)
        plot_sizes(img_size, sizes_dict)

    if args.s:
        plt.show()
    print("Took: {0}".format(time.time()-start))
