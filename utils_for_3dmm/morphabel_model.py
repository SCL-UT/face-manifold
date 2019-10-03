from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from utils_for_3dmm import mesh_numpy
from utils_for_3dmm import load


class MorphabelModel(object):
    """docstring for  MorphabelModel
    model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
            'shapeMU': [3*nver, 1]. *
            'shapePC': [3*nver, n_shape_para]. *
            'shapeEV': [n_shape_para, 1]. ~
            'expMU': [3*nver, 1]. ~ 
            'expPC': [3*nver, n_exp_para]. ~
            'expEV': [n_exp_para, 1]. ~
            'texMU': [3*nver, 1]. ~
            'texPC': [3*nver, n_tex_para]. ~
            'texEV': [n_tex_para, 1]. ~
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
            'kpt_ind': [68,] (start from 1). ~
    """
    def __init__(self, model_path, model_type = 'BFM'):
        super( MorphabelModel, self).__init__()
        if model_type=='BFM':
            self.model = load.load_BFM(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()
            
        # fixed attributes
        self.nver = self.model['shapePC'].shape[0]/3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texPC'].shape[1]
        
        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']
        self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))

    def get_shape_para(self, type='random', mean=0, cov=1):
        if type == 'zero':
            sp = np.zeros((self.n_shape_para, 1))
        elif type == 'random':
            m = mean * np.ones(self.n_shape_para)
            c = cov * np.diag((np.power(self.model['shapeEV'], 2)).reshape(-1)).astype(int)
            sp = np.random.multivariate_normal(m, c)
            sp = sp.reshape(self.n_shape_para, 1)
        return sp

    def get_exp_para(self, type='random', mean=0, cov=1):
        if type == 'zero':
            ep = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            m = mean * np.ones(self.n_exp_para)
            c = cov * np.diag((np.power(self.model['expEV'], 2)).reshape(-1)) / np.amax(np.power(self.model['expEV'], 2))
            ep = np.random.multivariate_normal(m, c)
            ep = ep.reshape(self.n_exp_para, 1)
        return ep

    def get_tex_para(self, type='random', mean=0, cov=1):
        if type == 'zero':
            tp = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            m = mean * np.ones(self.n_tex_para)
            c = cov * np.diag((np.power(self.model['texEV'], 2)).reshape(-1)) / np.amax(np.power(self.model['texEV'], 2))
            tp = np.random.multivariate_normal(m, c)
            tp = tp.reshape(self.n_tex_para, 1)
        return tp

    # uniform distribution

    def get_exp_para_uniform(self, mean=0, range=1):
        ep = np.zeros([self.n_exp_para, 1])
        exp_EV = self.model['expEV']/np.amax(self.model['expEV'])
        for i in np.arange(self.n_exp_para):
            ep[i, :] = mean - range*exp_EV[i][0]/2 + range*exp_EV[i][0]*np.random.random()
        return ep
    
    def get_shape_para_uniform(self, mean=0, range=1):
        sp = np.zeros([self.n_shape_para, 1])
        shape_EV = self.model['shapeEV']
        for i in np.arange(self.n_shape_para):
            sp[i, :] = mean - range*shape_EV[i][0]/2 + range*shape_EV[i][0]*np.random.random()
        return sp

    def add_noise_exp(self, exp_para, indices, sigma):
        mu = 0
        added_value = np.random.normal(mu, sigma, len(indices))
        ep = np.zeros(np.shape(exp_para))
        for i, p in enumerate(exp_para):
            ep[i] = exp_para[i]
        for i, index in enumerate(indices):
            ep[index] = exp_para[index] + added_value[i]
        return ep

    def add_noise_shape(self, shape_para, indices, sigma):
        mu = 0
        if sigma == 'ev':
            added_value = np.zeros((len(indices), 1))
            for i, index in enumerate(indices):
                added_value[i] = np.random.normal(mu, self.model['shapeEV'][i])
        else:
            added_value = np.random.normal(mu, sigma, len(indices))
        sp = np.zeros(np.shape(shape_para))
        for i, p in enumerate(shape_para):
            sp[i] = shape_para[i]
        for i, index in enumerate(indices):
            sp[index] = shape_para[index] + added_value[i]
        return sp

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1) 
        Returns:
            vertices: (nver, 3)
        '''
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape_para) + self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T
        return vertices

    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''
        colors = self.model['texMU'] + self.model['texPC'].dot(tex_para*self.model['texEV'])
        colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F').T/255.
        return colors


    # ------------------------------------------- transformation
    # -------------  transform
    def rotate(self, vertices, angles):
        ''' rotate face
        Args:
            vertices: [nver, 3]
            angles: [3] x, y, z rotation angle(degree)
            x: pitch. positive for looking down 
            y: yaw. positive for looking left
            z: roll. positive for tilting head right
        Returns:
            vertices: rotated vertices
        '''
        return mesh_numpy.transform.rotate(vertices, angles)

    def transform(self, vertices, s, angles, t3d):
        R = mesh_numpy.transform.angle2matrix(angles)
        return mesh_numpy.transform.similarity_transform(vertices, s, R, t3d)


