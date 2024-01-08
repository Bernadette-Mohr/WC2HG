import numpy as np
from openpathsampling.engines.toy import PES


class MuellerBrown(PES):

    def __init__(self, A, alpha, beta, gamma, a, b, max_u, scale):
        super(MuellerBrown, self).__init__()
        self.scale = scale
        self.A = np.array(A)
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.gamma = np.array(gamma)
        self.a = np.array(a)
        self.b = np.array(b)
        self.max_u = max_u
        self._local_dVdx = None

    def to_dict(self):
        dct = super(MuellerBrown, self).to_dict()
        dct['alpha'] = dct['alpha'].tolist()
        dct['beta'] = dct['beta'].tolist()
        dct['gamma'] = dct['gamma'].tolist()
        dct['a'] = dct['a'].tolist()
        dct['b'] = dct['b'].tolist()

        return dct

    def __repr__(self):
        return "MuellerBrown({o.A}, {o.alpha}, {o.beta}, {o.gamma}, {o.a}, {o.b}, {o.max_u}, {o.scale})".format(o=self)

    def V(self, sys):
        x, y = sys.positions
        V = 0.0
        for k in range(0, len(self.a)):
            V += self.A[k] * np.exp(self.alpha[k] * np.power((x - self.a[k]), 2)
                                    + self.beta[k] * (x - self.a[k]) * (y - self.b[k])
                                    + self.gamma[k] * np.power((y - self.b[k]), 2))

        return self.scale * np.where(V > self.max_u, self.max_u, V)

    def dVdx(self, sys):
        x, y = sys.positions
        dVdx, dVdy = 0.0, 0.0
        for k in range(0, len(self.a)):
            dVdx += self.A[k] * np.exp(self.alpha[k] * np.power((x - self.a[k]), 2)
                                       + self.beta[k] * (x - self.a[k]) * (y - self.b[k])
                                       + self.gamma[k] * np.power((y - self.b[k]), 2)) \
                    * (2 * self.alpha[k] * (x - self.a[k]) + self.beta[k] * (y - self.b[k]))
        for k in range(0, len(self.a)):
            dVdy += self.A[k] * np.exp(self.alpha[k] * np.power((x - self.a[k]), 2)
                                       + self.beta[k] * (x - self.a[k]) * (y - self.b[k])
                                       + self.gamma[k] * np.power((y - self.b[k]), 2)) \
                    * (self.beta[k] * (x - self.a[k]) + 2 * self.gamma[k] * (y - self.b[k]))

        self._local_dVdx = np.array([self.scale * dVdx, self.scale * dVdy])
        return self._local_dVdx
