from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class kernel(object):

    def __init__(self, conf, nvar):

        self.nvar = nvar
        self.ivarf = np.loadtxt("%s/ivar_%i.dat" % (conf.etcdir, self.nvar), dtype = int) - 1  # 9, 57, 81, 289, 625
        self.nf = np.shape(self.ivarf)[0]

        print("   Kernel initialized with filter size %i and %i free parameters" % (self.nf, self.nvar))
