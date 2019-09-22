import scipy
import numpy as np


def brightness(img):
    """
    Подсчет яркости изображения.
    Возращаемое значение -- матрица (yij), где yij - яркость пикселя img[i, j]
    """
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float64)


def energy_function(img):
    '''
    Функция считает энергию всех пикселей изображения img
    Возращаемое значение -- матрица (Eij), где Eij - энергия пикселя img[i, j]
    '''
    Y = brightness(img)
    ker_x = [[0., 0., 0.],
             [1., 0., -1.],
             [0., 0., 0.]]
    ker_y = [[0., 1., 0.],
             [0., 0., 0.],
             [0., -1., 0.]]
    I_x = scipy.signal.convolve2d(Y, ker_x, boundary='symm', mode='same')
    I_y = scipy.signal.convolve2d(Y, ker_y, boundary='symm', mode='same')
    return np.sqrt(I_x * I_x + I_y * I_y).astype(np.float64)


def change_energy(energy, mask):
    '''
    Изменение energy отдельных областей, обозначенных '1' в mask
    Возращает измененную матрицу energy
    '''
    delta = np.float64(energy.shape[0] * energy.shape[1] * 256.0)
    energy += delta * mask

def find_vertical_seam(img, mask):
    '''
    Нахождение вертикального шва с минимальной энергией из img.
    Возвращает маску шва.
    '''
    dp = np.full((img.shape[0], img.shape[1]), np.inf, dtype='float64')
    prev = np.zeros((img.shape[0], img.shape[1]), dtype='int8')

    energy = energy_function(img)
    change_energy(energy, mask)

    dp[0, :] = energy[0, :]

    height, width = energy.shape
    for y in range(1, height):
        for x in range(0, width):
            shift = dp[y - 1, max(0, x - 1):min(width, x + 2)].argmin() - (0 if x - 1 < 0 else 1)
            dp[y][x] = dp[y - 1][x + shift] + energy[y][x]
            prev[y][x] = shift

    cur_position = [height - 1, 0]
    lowest_energy = dp[height - 1, 0]
    for x in range(1, width):
        if lowest_energy > dp[height - 1, x]:
            cur_position[1] = x
            lowest_energy = dp[height - 1, x]

    seam_mask = np.zeros((height, width), dtype='int8')
    y, x = cur_position
    seam_mask[y, x] = 1
    while cur_position[0] > 0:
        cur_position[0] += -1
        cur_position[1] += prev[y, x]
        y, x = cur_position
        seam_mask[y, x] = 1
        
    return seam_mask


def horizontal_shrink(img, mask):
    '''
    Удаление вертикального шва с минимальной энергией из img.
    '''
    seam_mask = find_vertical_seam(img, mask)
    height, width, _ = img.shape

    new_mask = np.zeros((height, width - 1), dtype='int8')
    out = np.zeros(((height, width - 1, 3)), dtype='uint8')
    for i in range(0, height):
        for j in range(0, width):
            if seam_mask[i][j]:
                out[i, :j] = img[i, :j]
                out[i, j:] = img[i, j + 1:]
                new_mask[i, :j] = mask[i, :j]
                new_mask[i, j:] = mask[i, j + 1:]

    return [out, new_mask, seam_mask]


def vertical_shrink(img, mask):
    '''
    Сжатие на 1 pxl по вертикали
    '''
    transp = (img[:, :, 0].transpose(),
              img[:, :, 1].transpose(), img[:, :, 2].transpose())
    mask = mask.transpose()

    out, new_mask, seam_mask = horizontal_shrink(np.stack(transp, axis=-1), mask)
    out_channels = (out[:, :, 0].transpose(),
                    out[:, :, 1].transpose(), out[:, :, 2].transpose())
    return [np.stack(out_channels, axis=-1), new_mask.transpose(), seam_mask.transpose()]


def horizontal_expand(img, mask):
    '''
    Расширить избражения по вертикали на 1pxl
    '''
    seam_mask = find_vertical_seam(img, mask)
    height, width, _ = img.shape
    new_mask = np.zeros((height, width + 1), dtype='int8')
    out = np.zeros((height, width + 1, 3), dtype='uint8')
    new_seam_mask = np.zeros((height, width + 1), dtype='int8')

    for i in range(0, height):
        k = 0
        for j in range(0, width):
            if seam_mask[i][j]:
                delta = img[i][j + 1] if j + 1 != width else img[i][j]
                out[i, :j + 1] = img[i, :j + 1]
                out[i, j + 2:] = img[i, j + 1:]
                out[i][j + 1] = img[i][j] // 2 + delta // 2
                new_mask[i, :j + 1] = mask[i, :j + 1]
                new_mask[i, j + 2:] = mask[i, j + 1:]
                new_mask[i][j + 1] = mask[i][j]
                

    return [out, new_mask, seam_mask]


def vertical_expand(img, mask):
    '''
    Расширить избражения по вертикали на 1 pxl
    '''
    transp = (img[:, :, 0].transpose(),
              img[:, :, 1].transpose(), img[:, :, 2].transpose())
    mask = mask.transpose()

    out, new_mask, seam_mask = horizontal_expand(np.stack(transp, axis=-1), mask)
    out_channels = (out[:, :, 0].transpose(),
                    out[:, :, 1].transpose(), out[:, :, 2].transpose())
    return [np.stack(out_channels, axis=-1), new_mask.transpose(), seam_mask.transpose()]


def seam_carve(img, op, mask=None):
    '''
    Изменяет img в 4-х режимах op используя mask
    '''
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype='int8')

    do = {'horizontal shrink': horizontal_shrink,
          'vertical shrink': vertical_shrink,
          'horizontal expand': horizontal_expand,
          'vertical expand': vertical_expand}
    return do[op](img, mask)