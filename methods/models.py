import math
import pints
from ddeint import ddeint
from . import day_len, PARAMETER_BOUNDARIES

class BaseHPAModel(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
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
            # Arbitrary single cosine csh drive (for testing)
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
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

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
    
class HPAModelFEInter(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
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
            # Arbitrary single cosine csh drive (for testing)
            crh = 70*math.cos(2*math.pi*(t/T_c))+75
        return crh

    def simulate(self, parameters, times):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        K_a = self.parameters['K_a']
        K_f = self.parameters['K_f']
        k_1 = self.parameters['k_1']
        k_2 = self.parameters['k_2']
        alpha = self.parameters['alpha']
        delay = self.parameters['delay']
        lambda_s = self.parameters['lambda_s']
        lambda_a = self.parameters['lambda_a']
        t_s = self.parameters['t_s']
        m_a = self.parameters['m_a']
        m_f = self.parameters['m_f']
        sigma = self.parameters['sigma']
        gamma_a = self.parameters['gamma_a']
        gamma_f = self.parameters['gamma_f']
        gamma_e = self.parameters['gamma_e']

        # Define the DDE model
        def model(Y, t):
            A, F, E = Y(t)
            F_delay = Y(t - delay)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dFdt = -gamma_f*F + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_1*E - k_2*F
            dEdt = -gamma_e*E - k_1*E + k_2*F

            return [dAdt, dFdt, dEdt]

        # Define initial conditions
        def initial_conditions(t):
            return [5, 250, 50]
        
        # Run the simulation     
        result = ddeint(model, initial_conditions, times)

        # Truncate to specified range
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

        return result

    def n_outputs(self):
        return 3

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
    
class HPAModelFEInterCBGAlb(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
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
            # Arbitrary single cosine csh drive (for testing)
            crh = 70*math.cos(2*math.pi*(t/T_c))+75
        return crh

    def simulate(self, parameters, times):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        K_a = self.parameters['K_a']
        K_f = self.parameters['K_f']
        k_1 = self.parameters['k_1']
        k_2 = self.parameters['k_2']
        k_3 = self.parameters['k_3']
        k_4 = self.parameters['k_4']
        k_5 = self.parameters['k_5']
        k_6 = self.parameters['k_6']
        k_7 = self.parameters['k_7']
        k_8 = self.parameters['k_8']
        k_9 = self.parameters['k_9']
        k_10 = self.parameters['k_10']
        alpha = self.parameters['alpha']
        delay = self.parameters['delay']
        lambda_s = self.parameters['lambda_s']
        lambda_a = self.parameters['lambda_a']
        t_s = self.parameters['t_s']
        m_a = self.parameters['m_a']
        m_f = self.parameters['m_f']
        sigma = self.parameters['sigma']
        gamma_a = self.parameters['gamma_a']
        gamma_f = self.parameters['gamma_f']
        gamma_e = self.parameters['gamma_e']

        # Define the DDE model
        def model(Y, t):
            A, F, E, F_CBG, F_Alb, E_CBG, E_Alb, CBG, Alb = Y(t)
            F_delay = Y(t - delay)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dFdt = -(gamma_f+k_2+k_3*CBG+k_5*Alb)*F + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_1*E + k_4*F_CBG + k_6*F_Alb
            dEdt = -(gamma_e+k_1+k_7*CBG+k_9*Alb)*E + k_2*F + k_8*E_CBG + k_10*E_Alb
            dF_CBGdt = k_3*F*CBG - k_4*F_CBG
            dF_Albdt = k_5*F*Alb - k_6*F_Alb
            dE_CBGdt = k_7*E*CBG - k_8*E_CBG
            dE_Albdt = k_9*E*Alb - k_10*E_Alb
            dCBGdt = k_4*F_CBG - k_3*F*CBG + k_8*E_CBG - k_7*E*CBG
            dAlbdt = k_6*F_Alb - k_5*F*Alb + k_10*E_Alb - k_9*E*Alb

            return [dAdt, dFdt, dEdt, dF_CBGdt, dF_Albdt, dE_CBGdt, dE_Albdt, dCBGdt, dAlbdt]

        # Define initial conditions
        def initial_conditions(t):
            return [5, 200, 40, 0, 0, 0, 0, 200, 500]
        
        # Run the simulation     
        result = ddeint(model, initial_conditions, times)

        # Truncate to specified range
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

        return result

    def n_outputs(self):
        return 9

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