import numpy as np
from scipy.ndimage.filters import minimum_filter1d
import matplotlib.pyplot as plt
from skimage.io import imread


def show_image(image):
    # для отображения изображения в Pycharm
    plt.figure()
    plt.imshow(image)
    plt.show(block=True)


def calculate_gradient(image):
    grad_x = np.empty(image.shape)
    grad_x[0, :] = image[1, :] - image[0, :]
    grad_x[-1, :] = image[-1, :] - image[-2, :]
    grad_x[1:-1, :] = image[2:, :] - image[:-2, :]
    grad_y = np.empty(image.shape)
    grad_y[:, 0] = image[:, 1] - image[:, 0]
    grad_y[:, -1] = image[:, -1] - image[:, -2]
    grad_y[:, 1:-1] = image[:, 2:] - image[:, :-2]

    return np.hypot(grad_x, grad_y)


def get_seam(image_energy):
    # формируем матрицу
    for i in range(1, image_energy.shape[0]):
        image_energy[i] += minimum_filter1d(image_energy[i - 1], 3)

    # выделяем шов с минимальной энергией:
    seam_mask = np.ones_like(image_energy, bool)
    j_pos = np.argmin(image_energy[-1])
    seam_mask[-1, j_pos] = False
    for i in range(image_energy.shape[0] - 2, -1, -1):
        j_pos += np.argmin(image_energy[i: i + 1, max(0, j_pos - 1): min(j_pos + 2, image_energy.shape[1])]) - \
                 (j_pos != 0)
        seam_mask[i, j_pos] = False

    return seam_mask


def remove_seam(image, seam_mask):
    return image[seam_mask].reshape(image.shape[0], image.shape[1] - 1)


def add_seam(image, seam_mask):
    new_image = np.empty((image.shape[0], image.shape[1] + 1))
    for row in range(image.shape[0]):
        col = int(np.argmin(seam_mask[row]))
        if col == image.shape[1]:
            new_image[row, : col] = image[row, :col]
            new_image[row, col] = np.average(image[row, -2:])
        else:
            new_image[row, : col + 1] = image[row, : col + 1]
            new_image[row, col + 1] = np.average(image[row, col: min(col + 2, image.shape[1])])
            new_image[row, col + 2:] = image[row, col + 1:]

    return new_image


def seam_carve(image, mode, mask=None):
    """ Контекстно-зависимое масштабирование изображений
        @:param image входное изображение
        @:param mode:
                    'horizontal shrink' — сжатие по горизонтали,
                    'vertical shrink'   — сжатие по вертикали,
                    'horizontal expand' — расширение по горизонтали,
                    'vertical expand'   — расширение по вертикали.
    """
    new_mask = mask
    if mask is None:
        new_mask = np.zeros(image.shape[:2])

    direction, deformation = mode.split(' ')
    if direction == 'vertical':
        image = np.transpose(image, (1, 0, 2))
        new_mask = np.transpose(new_mask)

    # разбиваем изображение по каналам:
    r_channel, g_channel, b_channel = image[..., 0], image[..., 1], image[..., 2]

    # считаем яркость изображение (компонента Y в цветовой модели YUV):
    y_channel = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    magnifier = new_mask.shape[0] * new_mask.shape[1] * 256
    image_energy = calculate_gradient(y_channel) + new_mask.astype(np.int32) * magnifier
    seam_mask = get_seam(image_energy)  # image_energy изменяется

    if deformation == 'shrink':
        # удаляем шов:
        r_channel = remove_seam(r_channel, seam_mask)
        g_channel = remove_seam(g_channel, seam_mask)
        b_channel = remove_seam(b_channel, seam_mask)
        new_mask = remove_seam(new_mask, seam_mask)
    elif deformation == 'expand':
        # добавляем шов:
        r_channel = add_seam(r_channel, seam_mask)
        g_channel = add_seam(g_channel, seam_mask)
        b_channel = add_seam(b_channel, seam_mask)
        new_mask = add_seam(new_mask, seam_mask)

    # собираем изображение
    new_image = np.dstack((r_channel, g_channel, b_channel))
    if direction == 'vertical':
        image = np.transpose(new_image, (1, 0, 2))
        new_image = np.transpose(new_image, (1, 0, 2))
        new_mask = np.transpose(new_mask)
        seam_mask = np.transpose(seam_mask)

    return (new_image * 255).astype(np.uint8), new_mask, (~seam_mask).astype(np.uint8)


if __name__ == '__main__':
    img = imread('img1.png', plugin='matplotlib')
    seam_carve(img, 'horizontal shrink', None)
