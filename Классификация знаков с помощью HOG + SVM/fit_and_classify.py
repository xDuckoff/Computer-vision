import numpy as np
import scipy
import math
import sklearn.svm as svm
from sklearn.utils import shuffle 
from sklearn.model_selection import cross_val_score

def rgb_to_grayscale(img):
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] +
            0.114 * img[:, :, 2]).astype(np.float64)


def gradient_calculation(img):
    Y = rgb_to_grayscale(img)
    ker_x = [[1., 0., -1.]]
    ker_y = [[1.],
             [0.],
             [-1.]]
    #ПРОВЕРИТЬ ДЕЙСТВИТЕЛЬНО ЛИ arctan ВОЗВРАЩАЕТ -pi до pi, прибавлять pi такая себе идея
    Ix = scipy.signal.convolve2d(Y, ker_x, boundary='symm', mode='same')
    Iy = scipy.signal.convolve2d(Y, ker_y, boundary='symm', mode='same')
    return (np.sqrt(Ix * Ix + Iy * Iy).astype(np.float64), np.pi + np.arctan2(Iy, Ix))

def histogram_calculation(img, cell_rows, cell_cols, bin_count):
    modulus, direction = gradient_calculation(img)
    heigth, width, _ = img.shape
    row_count = heigth // cell_rows
    col_count = width // cell_cols
    
    magic_matrix = np.zeros((heigth, width, bin_count), dtype='float64')
    alpha = 2 * np.pi / bin_count
    
    for i in range(bin_count):
        magic_matrix[:,:, i] = np.logical_and(direction > (i * alpha), 
                               direction < (i + 1) * alpha) * modulus
    
    histogram_matrix = np.zeros((row_count, col_count, bin_count), dtype='float64')
    for i in range(row_count):
        for j in range(col_count):
            for k in range(bin_count):
                histogram_matrix[i, j, k] = (magic_matrix[i * cell_rows:(i + 1) * cell_rows, 
                                       j * cell_cols:(j + 1) * cell_cols, k]).sum()
            
    return histogram_matrix

def norm(vector):
    return vector / math.sqrt((vector ** 2).sum() + 1e-10)

def hog(img, block_row_cells, block_col_cells, cell_rows, cell_cols, bin_count):
    histogram_matrix = histogram_calculation(img, cell_rows, cell_cols, bin_count)
    
    height, width, _ = histogram_matrix.shape
    descriptor = np.array([], dtype='float64')
    
    for i in range(0, height - block_row_cells + 1):
        for j in range(0, width - block_col_cells + 1):
            block_hist = np.ravel(histogram_matrix[i:i + block_row_cells, 
                                                   j:j + block_col_cells])
            descriptor = np.append(descriptor, norm(block_hist))
            
    return np.ravel(descriptor)

def extract_hog(img):
    import skimage.transform
    
    k_cut = 0.05
    cur_h, cur_w, _ = img.shape
    img_row_cut = int(cur_h * k_cut)
    img_col_cut = int(cur_w * k_cut)
    
    img1 = img[img_row_cut: cur_h - img_row_cut,
               img_col_cut: cur_w - img_col_cut]
    
    return hog(skimage.transform.resize(img1, (64, 64)), 2, 2, 8, 8, 8)

def fit_and_classify(X_train, y_train, X_test):
    X, y = shuffle(X_train, y_train, random_state=0)
    svm_clf = svm.LinearSVC(C=0.5)

    print(cross_val_score(svm_clf, X, y, cv=5))

    svm_clf.fit(X, y)
    return svm_clf.predict(X_test)
