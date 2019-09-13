from numpy import float64, stack
from numpy import roll, max
import skimage

def border_cut(x, y):
    left_cut = max([-y, 0])
    right_cut = max([y, 0])
    top_cut = max([-x, 0])
    bottom_cut = max([x, 0])
    return (left_cut, right_cut, top_cut, bottom_cut)

def mse(I1, I2):
    from numpy import mean
    return mean((I1 - I2) ** 2)


def norm(I1, I2):
    from numpy import sum, sqrt
    return sum(I1 * I2) / sqrt(sum(I1 * I1) * sum(I2 * I2))

def intersect(img1, img2, x, y):
    left_cut, right_cut, top_cut, bottom_cut = border_cut(x, y)
    height, width = img1.shape
    out_img1 = img1[top_cut:height-bottom_cut, left_cut:width-right_cut]
    out_img2 = img2[bottom_cut:height-top_cut, right_cut:width-left_cut]
    return (out_img1, out_img2)
    
def best_shifting(channel1, channel2, x_range, y_range):
    best_eq = None
    pos = [0, 0]
    for x in range(x_range[0], x_range[1]):
        for y in range(y_range[0], y_range[1]):
            shifted_channel1, cuted_channel2 = intersect(channel1, channel2)
            metrik = mse(cuted_channel2, shifted_channel1)
            if best_eq == None or metrik < best_eq:
                best_eq = metrik
                pos = (x, y)
    return pos


def pyramid(channel1, channel2):
    if max(channel1.shape) < 400:
        return best_shifting(channel1, channel2, (-15, 16), (-15, 16))

    resize_channel1 = channel1[::2, ::2]
    resize_channel2 = channel2[::2, ::2]
    search_range = pyramid(resize_channel1, resize_channel2)

    return best_shifting(channel1, channel2, 2 * (search_range[0], search_range[0]) + (-2, 3),
                       2 * (search_range[1], search_range[1]) + (-2, 3))


def align(img, g_coord):
    row_g, col_g = g_coord

    img = skimage.img_as_float(img)
    full_h, full_w = img.shape
    full_h -= full_h % 3

    # Разделение на 3 канала
    cur_h = full_h // 3
    cur_w = full_w
    full_b = img[:cur_h]
    full_g = img[cur_h:2 * cur_h]
    full_r = img[2 * cur_h: full_h]

    # Обрезка краев
    k_cut = 0.10
    channel_row_cut = int(cur_h * k_cut)
    channel_col_cut = int(cur_w * k_cut)

    b = full_b[channel_row_cut: cur_h - channel_row_cut,
               channel_col_cut: cur_w - channel_col_cut]
    g = full_g[channel_row_cut: cur_h - channel_row_cut,
               channel_col_cut: cur_w - channel_col_cut]
    r = full_r[channel_row_cut: cur_h - channel_row_cut,
               channel_col_cut: cur_w - channel_col_cut]

    # совмещаем
    x, y = pyramid(r, g)
    shifted_r = roll(full_r, (x, y), (0, 1))
    row_r = row_g - x + img.shape[0] // 3
    col_r = col_g - y

    x, y = pyramid(b, g)
    shifted_b = roll(full_b, (x, y), (0, 1))
    row_b = row_g - x - img.shape[0] // 3
    col_b = col_g - y

    return stack((shifted_r, full_g, shifted_b), axis=-1), (row_b, col_b), (row_r, col_r)