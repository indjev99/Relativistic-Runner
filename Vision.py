import numpy as np
import scipy.stats as st

COLOR_MATRIX = np.array([[0.67, 0.21, 0.14],
                         [0.33, 0.60, 0.06],
                         [0.00, 0.09, 0.78]])

INV_COLOR_MATRIX = np.linalg.inv(COLOR_MATRIX)

def wavelength_to_rgb(wavelength):
    col_x = receptor_x(wavelength)
    col_y = receptor_y(wavelength)
    col_z = receptor_z(wavelength)
    tot = col_x + col_y + col_z
    rgb_vec = np.dot(INV_COLOR_MATRIX, np.array([col_x, col_y, col_z]) / tot)
    rgb_vec[rgb_vec < 0] = 0
    rgb_vec = rgb_vec / np.max(rgb_vec) * 255
    return rgb_vec

def receptor_x(wavelength):
    return (st.norm.pdf(wavelength, 600, 30) * 30 + st.norm.pdf(wavelength, 450, 10) * 10 * 0.3)/0.4

def receptor_y(wavelength):
    return st.norm.pdf(wavelength, 550, 50) * 50 * 0.9 / 0.4

def receptor_z(wavelength):
    return st.norm.pdf(wavelength, 450, 20) * 20 * 1.75 / 0.4
