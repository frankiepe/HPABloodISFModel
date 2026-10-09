import pints
import numpy as np

class Model6Boundaries(pints.Boundaries):
    """
    Boundary contraints of the parameters
    """
    def __init__(self, lower, upper):
        super(Model6Boundaries, self).__init__()

        # Convert to shape (n,) vectors, copy to ensure they remain unchanged
        self._lower = pints.vector(lower)
        self._upper = pints.vector(upper)

        # Get and check dimension
        self._n_parameters = len(self._lower)
        if len(self._upper) != self._n_parameters:
            raise ValueError('Lower and upper bounds must have same length.')

        # Check dimension is at least 1
        if self._n_parameters < 1:
            raise ValueError('The parameter space must have a dimension > 0')

        # Check if upper > lower
        if not np.all(self._upper > self._lower):
            raise ValueError('Upper bounds must exceed lower bounds.')
    
    def check(self, parameters):
            """ See :meth:`pints.Boundaries.check()`. """
            if np.any(parameters < self._lower):
                return False
            if np.any(parameters >= self._upper):
                return False
            return True
        
    def n_parameters(self):
            """ See :meth:`pints.Boundaries.n_parameters()`. """
            return self._n_parameters

    def lower(self):
            """
            Returns the lower boundaries for all parameters (as a read-only NumPy
            array).
            """
            return self._lower
        
    def range(self):
            """
            Returns the size of the parameter space (i.e. ``upper - lower``).
            """
            return self._upper - self._lower
        
    def sample(self, n=1):
            """ Sample kon/koff on logscale """
            f = lambda x: np.log(x)
            unf = lambda x: np.exp(x)
            p = np.zeros(self._n_parameters)
            for i in range(self._n_parameters):
                if i in [11,12,13,14]:
                    p[i] = unf(np.random.uniform(f(self._lower[i]), f(self._upper[i])))
                else:
                    p[i] = np.random.uniform(self._lower[i], self._upper[i])

            return [p]
        
    def upper(self):
            """
            Returns the upper boundary for all parameters (as a read-only NumPy
            array).
            """
            return self._upper