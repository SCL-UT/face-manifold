import numpy as np
from skimage import io
from utils_for_3dmm import mesh_numpy
import warnings


def img_from_params(path, name, bfm, sp, ep, tp, s=0.00088, h=256, w=256, angle=[0, 0, 0], t=[0, 0, 0]):
    vertices = bfm.generate_vertices(sp, ep)
    colors = bfm.generate_colors(tp)
    R = mesh_numpy.transform.angle2matrix(angle)
    vertices = mesh_numpy.transform.similarity_transform(vertices, s, R, t)  # transformed vertices
    # light conditions
    light_intensities = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
    light_positions = np.array([[0, 0, 700], [350, 0, 700], [0, 350, 700]])
    ambient_intensities = [0.1, 0.1, 0.1]
    lit_colors = mesh_numpy.light.add_light_new(vertices, bfm.triangles, colors, light_positions, light_intensities, ambient_intensities)
    image_vertices = mesh_numpy.transform.to_image(vertices, h, w)
    rendering = mesh_numpy.render.render_colors(image_vertices, bfm.triangles, lit_colors, h, w)
    rendering = np.minimum((np.maximum(rendering, 0)), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave('{}/{}.jpg'.format(path, name), rendering)
