import pints
import numpy as np

class ModelBlood(pints.ForwardModel):
    def __init__(self, model, parameters, times):
        self._model = model
        self.n_parameters_value = len(parameters)
        self._parameters = parameters
        self._times = times

    def n_outputs(self):
        return 2     

    def n_parameters(self):
        return self.n_parameters_value

    def simulate(self, parameters, times):
        res = self._model.simulate(parameters, times)
        return res[:, 0:2]

class ModelBloodFEInter(pints.ForwardModel):
    def __init__(self, model, parameters, times):
        self._model = model
        self.n_parameters_value = len(parameters)
        self._parameters = parameters
        self._times = times

    def n_outputs(self):
        return 3     

    def n_parameters(self):
        return self.n_parameters_value

    def simulate(self, parameters, times):
        res = self._model.simulate(parameters, times)
        return res[:, 0:3]

class ModelBloodFEInterCBGAlbSimple(pints.ForwardModel):
    def __init__(self, model, parameters, times):
        self._model = model
        self.n_parameters_value = len(parameters)
        self._parameters = parameters
        self._times = times

    def n_outputs(self):
        return 3     

    def n_parameters(self):
        return self.n_parameters_value

    def simulate(self, parameters, times):
        res = self._model.simulate(parameters, times)
        return np.transpose([res[:, 0], res[:, 1]+res[:, 3], res[:, 2]+res[:, 4]])

class ModelBloodFEInterCBGAlb(pints.ForwardModel):
    def __init__(self, model, parameters, times):
        self._model = model
        self.n_parameters_value = len(parameters)
        self._parameters = parameters
        self._times = times

    def n_outputs(self):
        return 3     

    def n_parameters(self):
        return self.n_parameters_value

    def simulate(self, parameters, times):
        res = self._model.simulate(parameters, times)
        return np.transpose([res[:, 0], res[:, 1]+res[:, 3]+res[:, 4], res[:, 2]+res[:, 5]+res[:, 6]])

class ModelISFFEInterCBGAlb(pints.ForwardModel):
    def __init__(self, model, parameters, times):
        self._model = model
        self.n_parameters_value = len(parameters)
        self._parameters = parameters
        self._times = times

    def n_outputs(self):
        return 2     

    def n_parameters(self):
        return self.n_parameters_value

    def simulate(self, parameters, times):
        res = self._model.simulate(parameters, times)
        return np.transpose([res[:, 9], res[:, 10]])