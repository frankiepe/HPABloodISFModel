import math
import pints
from ddeint import ddeint
from . import day_len, PARAMETER_BOUNDARIES

class BaseHPAModel(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 num_days=6,
                 days_to_keep=1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()

    def crh(self, t, t_s=None, lambda_a=None, lambda_s=None, sigma=None, T_c=day_len, symmetric=False):
        if t_s is None: 
            t_s = self.parameters['t_s']
        if lambda_a is None: 
            lambda_a = self.parameters['lambda_a']
        if lambda_s is None: 
            lambda_s = self.parameters['lambda_s']
        if sigma is None: 
            sigma = self.parameters['sigma']
        crh = lambda_a * math.e**(lambda_s*math.cos(2*math.pi*((t-t_s)/T_c)+sigma*math.cos(2*math.pi*((t-t_s)/T_c))))
        if symmetric:
            crh = 70*math.cos(2*math.pi*(t/T_c))+75
        return crh

    def simulate(self, parameters, times):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        k_a = self.parameters['k_a']
        k_c = self.parameters['k_c']
        alpha = self.parameters['alpha']
        delay = self.parameters['delay']
        lambda_s = self.parameters['lambda_s']
        lambda_a = self.parameters['lambda_a']
        t_s = self.parameters['t_s']
        m_a = self.parameters['m_a']
        m_c = self.parameters['m_c']
        sigma = self.parameters['sigma']
        gamma_a = self.parameters['gamma_a']
        gamma_c = self.parameters['gamma_c']

        # Define the DDE model
        def model(Y, t):
            A, C = Y(t)
            C_delay = Y(t - delay)[1]

            dAdt = -gamma_a*A + ((k_c**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(k_c**m_a+C_delay**m_a)
            dCdt = -gamma_c*C + alpha*((A**m_c)/(k_a**m_c + A**m_c))

            return [dAdt, dCdt]

        # Define initial conditions
        def initial_conditions(t):
            return [5, 400]
        
        # Run the simulation     
        result = ddeint(model, initial_conditions, times)

        # Truncate to specified range
        result = result[self.length_model*(self.num_days-self.days_to_keep):]

        return result

    def n_outputs(self):
        return 2

    def n_times(self):
        return self.length_model*self.num_days

    def n_parameters(self):
        return self.n_parameters_value

    def suggested_parameters(self):
        return list(self.suggested_params_dict.values())

    def get_and_create_boundaries(self):
        lowerbounds = []
        upperbounds = []

        for item in self.suggested_params_dict.keys():
            lowerbound, upperbound = self.parameter_boundaries.get(item, (None, None))
            if lowerbound is None or upperbound is None: 
                print(f'{item} has no defined bounds. Setting to default (0, 1000)')
                lowerbound, upperbound = (0, 1000)
            
            lowerbounds.append(lowerbound)
            upperbounds.append(upperbound)
        
        return pints.RectangularBoundaries(lowerbounds, upperbounds)
    
