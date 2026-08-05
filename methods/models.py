import math
import pints
from ddeint import ddeint
import numpy as np
from . import day_len, PARAMETER_BOUNDARIES

class BaseHPAModel(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 init_conds,
                 times,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.init_conds = init_conds
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()
        self.set_fix_parameters({})

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

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # HPC parameters
        gamma_a = self.parameters['gamma_a'] # ACTH degradation rate
        gamma_c = self.parameters['gamma_c'] # Cortisol degradation rate
        K_a = self.parameters['K_a'] # ACTH receptor half-saturation constant
        K_c = self.parameters['K_c'] # Cortisol receptor half-saturation constant
        m_a = self.parameters['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_c = self.parameters['m_c'] # Hill coefficient for CORT feedback
        delay = self.parameters['delay'] # Feedback delay from CORT to ACTH
        alpha = self.parameters['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = self.parameters['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = self.parameters['lambda_s'] # Circadian modulation strength
        t_s = self.parameters['t_s'] # Circadian phase shift
        sigma = self.parameters['sigma'] # Asymmetry of circadian drive

        # Initial conditions
        A_0 = self.init_conds['A']
        C_0 = self.init_conds['C']

        # Define the DDE model
        def model(Y, t):
            A, C = Y(t)
            C_delay = Y(t - delay)[1]

            dAdt = -gamma_a*A + ((K_c**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_c**m_a+C_delay**m_a)
            dCdt = -gamma_c*C + alpha*((A**m_c)/(K_a**m_c + A**m_c))

            return [dAdt, dCdt]

        # Define initial conditions
        def initial_conditions(t):
            return [A_0, C_0]
        
        # Run the simulation  
        result = ddeint(model, initial_conditions, self.times)

        # Truncate to specified range
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

        if fitting:
            # Find nearest indices
            indices = np.searchsorted(self.times, times)

            # Pull the filtered values
            filtered_output = result[indices]
            result = filtered_output

        return result

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.parameters[p] = parameters[p]

    def n_outputs(self):
        return 2

    def n_times(self):
        return self.length_model*self.num_days

    def n_parameters(self):
        return self.n_parameters_value

    def get_and_create_boundaries(self):
        lowerbounds = []
        upperbounds = []

        for item in self.parameters.keys():
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
                 init_conds,
                 times,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.init_conds = init_conds
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()
        self.set_fix_parameters({})

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

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # HPC parameters
        gamma_a = self.parameters['gamma_a'] # ACTH degradation rate
        gamma_f = self.parameters['gamma_f'] # Cortisol degradation rate
        gamma_e = self.parameters['gamma_e'] # Cortisone degradation rate (new param)
        K_a = self.parameters['K_a'] # ACTH receptor half-saturation constant
        K_f = self.parameters['K_f'] # Cortisol receptor half-saturation constant
        k_mf = self.parameters['k_mf'] # Cortisol conc. when reaction rate is half V_f (new param)
        k_me = self.parameters['k_me'] # Cortisone conc. when reaction rate is half V_e (new param)
        m_a = self.parameters['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_f = self.parameters['m_f'] # Hill coefficient for CORT feedback
        V_f = self.parameters['V_f'] # Max. cortisol to cortisone rate (new param)
        V_e = self.parameters['V_e'] # Max. cortisone to cortisol rate (new param)
        delay = self.parameters['delay'] # Feedback delay from CORT to ACTH
        alpha = self.parameters['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = self.parameters['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = self.parameters['lambda_s'] # Circadian modulation strength
        t_s = self.parameters['t_s'] # Circadian phase shift
        sigma = self.parameters['sigma'] # Asymmetry of circadian drive

        # Initial conditions
        A_0 = self.init_conds['A']
        F_0 = self.init_conds['F']
        E_0 = self.init_conds['E']

        # Define the DDE model
        def model(Y, t):
            A, F, E = Y(t)
            F_delay = Y(t - delay)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dFdt = -gamma_f*F + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + (V_e*E)/(k_me+E) - (V_f+F)/(k_mf+F)
            dEdt = -gamma_e*E - (V_e*E)/(k_me+E) + (V_f+F)/(k_mf+F)

            return [dAdt, dFdt, dEdt]

        # Define initial conditions
        def initial_conditions(t):
            return [A_0, F_0, E_0]
        
        # Run the simulation     
        result = ddeint(model, initial_conditions, self.times)

        # Truncate to specified range
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

        if fitting:
            # Find nearest indices
            indices = np.searchsorted(self.times, times)

            # Pull the filtered values
            filtered_output = result[indices]
            result = filtered_output

        return result

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.parameters[p] = parameters[p]

    def n_outputs(self):
        return 3

    def n_times(self):
        return self.length_model*self.num_days

    def n_parameters(self):
        return self.n_parameters_value

    def get_and_create_boundaries(self):
        lowerbounds = []
        upperbounds = []

        for item in self.parameters.keys():
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
                 init_conds,
                 times,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.init_conds = init_conds
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()
        self.set_fix_parameters({})

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

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # HPC parameters
        gamma_a = self.parameters['gamma_a'] # ACTH degradation rate
        gamma_f = self.parameters['gamma_f'] # Cortisol degradation rate
        gamma_e = self.parameters['gamma_e'] # Cortisone degradation rate
        K_a = self.parameters['K_a'] # ACTH receptor half-saturation constant
        K_f = self.parameters['K_f'] # Cortisol receptor half-saturation constant
        k_mf = self.parameters['k_mf'] # Cortisol conc. when reaction rate is half V_f
        k_me = self.parameters['k_me'] # Cortisone conc. when reaction rate is half V_e
        k_1 = self.parameters['k_1'] # F:CBG on-binding rate (new param)
        k_2 = self.parameters['k_2'] # F:CBG off-binding rate (new param)
        k_3 = self.parameters['k_3'] # F:Alb on-binding rate (new param)
        k_4 = self.parameters['k_4'] # F:Alb off-binding rate (new param)
        k_5 = self.parameters['k_5'] # E:CBG on-binding rate (new param)
        k_6 = self.parameters['k_6'] # E:CBG off-binding rate (new param)
        k_7 = self.parameters['k_7'] # E:Alb on-binding rate (new param)
        k_8 = self.parameters['k_8'] # E:Alb off-binding rate (new param)
        m_a = self.parameters['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_f = self.parameters['m_f'] # Hill coefficient for CORT feedback
        V_f = self.parameters['V_f'] # Max. cortisol to cortisone rate
        V_e = self.parameters['V_e'] # Max. cortisone to cortisol rate
        delay = self.parameters['delay'] # Feedback delay from CORT to ACTH
        alpha = self.parameters['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = self.parameters['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = self.parameters['lambda_s'] # Circadian modulation strength
        t_s = self.parameters['t_s'] # Circadian phase shift
        sigma = self.parameters['sigma'] # Asymmetry of circadian drive

        # Initial conditions
        A_0 = self.init_conds['A']
        F_0 = self.init_conds['F']
        E_0 = self.init_conds['E']
        F_CBG_0 = self.init_conds['F_CBG']
        F_Alb_0 = self.init_conds['F_Alb'] 
        E_CBG_0 = self.init_conds['E_CBG'] 
        E_Alb_0 = self.init_conds['E_Alb'] 
        CBG_0 = self.init_conds['CBG'] 
        Alb_0 = self.init_conds['Alb']  

        # Define the DDE model
        def model(Y, t):
            A, F, E, F_CBG, F_Alb, E_CBG, E_Alb, CBG, Alb = Y(t)
            F_delay = Y(t - delay)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dFdt = -(gamma_f+k_1*CBG+k_3*Alb)*F + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_2*F_CBG + k_4*F_Alb + (V_e*E)/(k_me+E) - (V_f+F)/(k_mf+F)
            dEdt = -(gamma_e+k_5*CBG+k_7*Alb)*E + k_6*E_CBG + k_8*E_Alb - (V_e*E)/(k_me+E) + (V_f+F)/(k_mf+F)
            dF_CBGdt = k_1*F*CBG - k_2*F_CBG
            dF_Albdt = k_3*F*Alb - k_4*F_Alb
            dE_CBGdt = k_5*E*CBG - k_6*E_CBG
            dE_Albdt = k_7*E*Alb - k_8*E_Alb
            dCBGdt = k_2*F_CBG - k_1*F*CBG + k_6*E_CBG - k_5*E*CBG
            dAlbdt = k_4*F_Alb - k_3*F*Alb + k_8*E_Alb - k_7*E*Alb

            return [dAdt, dFdt, dEdt, dF_CBGdt, dF_Albdt, dE_CBGdt, dE_Albdt, dCBGdt, dAlbdt]

        # Define initial conditions
        def initial_conditions(t):
            return [A_0, F_0, E_0, F_CBG_0, F_Alb_0, E_CBG_0, E_Alb_0, CBG_0, Alb_0]
        
        # Run the simulation     
        result = ddeint(model, initial_conditions, self.times)

        # Truncate to specified range
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

        if fitting:
            # Find nearest indices
            indices = np.searchsorted(self.times, times)

            # Pull the filtered values
            filtered_output = result[indices]
            result = filtered_output

        return result

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.parameters[p] = parameters[p]

    def n_outputs(self):
        return 9

    def n_times(self):
        return self.length_model*self.num_days

    def n_parameters(self):
        return self.n_parameters_value

    def get_and_create_boundaries(self):
        lowerbounds = []
        upperbounds = []

        for item in self.parameters.keys():
            lowerbound, upperbound = self.parameter_boundaries.get(item, (None, None))
            if lowerbound is None or upperbound is None: 
                print(f'{item} has no defined bounds. Setting to default (0, 1000)')
                lowerbound, upperbound = (0, 1000)
            
            lowerbounds.append(lowerbound)
            upperbounds.append(upperbound)
        
        return pints.RectangularBoundaries(lowerbounds, upperbounds)
    
class HPAModelFEInterCBGAlbBloodISF(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 times,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()
        self.set_fix_parameters({})

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

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        K_a = self.parameters['K_a']
        K_f = self.parameters['K_f']
        k_mf = self.parameters['k_mf']
        k_me = self.parameters['k_me']
        V_f = self.parameters['V_f']
        V_e = self.parameters['V_e']
        k_1 = self.parameters['k_1']
        k_2 = self.parameters['k_2']
        k_3 = self.parameters['k_3']
        k_4 = self.parameters['k_4']
        k_5 = self.parameters['k_5']
        k_6 = self.parameters['k_6']
        k_7 = self.parameters['k_7']
        k_8 = self.parameters['k_8']
        k_BI = self.parameters['k_BI'] # new param
        V_B = self.parameters['V_B'] # new param
        V_I = self.parameters['V_I'] # new param
        alpha = self.parameters['alpha']
        delay = self.parameters['delay']
        lambda_s = self.parameters['lambda_s']
        lambda_a = self.parameters['lambda_a']
        t_s = self.parameters['t_s']
        m_a = self.parameters['m_a']
        m_f = self.parameters['m_f']
        sigma = self.parameters['sigma']
        gamma_a = self.parameters['gamma_a']
        gamma_f_b = self.parameters['gamma_f_b']
        gamma_e_b = self.parameters['gamma_e_b']
        gamma_f_i = self.parameters['gamma_f_i'] # new param
        gamma_e_i = self.parameters['gamma_e_i'] # new param

        # Define the DDE model
        def model(Y, t):
            A, F_B, E_B, F_CBG, F_Alb, E_CBG, E_Alb, CBG, Alb, F_I, E_I = Y(t)
            F_delay = Y(t - delay)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dF_Bdt = -(gamma_f_b+k_1*CBG+k_3*Alb)*F_B + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_2*F_CBG + k_4*F_Alb + (V_e*E_B)/(k_me+E_B) - (V_f+F_B)/(k_mf+F_B) - (k_BI/V_B)*(F_B-F_I)
            dE_Bdt = -(gamma_e_b+k_5*CBG+k_7*Alb)*E_B + k_6*E_CBG + k_8*E_Alb - (V_e*E_B)/(k_me+E_B) + (V_f+F_B)/(k_mf+F_B) - (k_BI/V_B)*(E_B-E_I)
            dF_CBGdt = k_1*F_B*CBG - k_2*F_CBG
            dF_Albdt = k_3*F_B*Alb - k_4*F_Alb
            dE_CBGdt = k_5*E_B*CBG - k_6*E_CBG
            dE_Albdt = k_7*E_B*Alb - k_8*E_Alb
            dCBGdt = k_2*F_CBG - k_1*F_B*CBG + k_6*E_CBG - k_5*E_B*CBG
            dAlbdt = k_4*F_Alb - k_3*F_B*Alb + k_8*E_Alb - k_7*E_B*Alb
            dF_Idt = (k_BI/V_I)*(F_B-F_I) - gamma_f_i*F_I
            dE_Idt = (k_BI/V_I)*(E_B-E_I) - gamma_e_i*E_I

            return [dAdt, dF_Bdt, dE_Bdt, dF_CBGdt, dF_Albdt, dE_CBGdt, dE_Albdt, dCBGdt, dAlbdt, dF_Idt, dE_Idt]

        # Define initial conditions
        def initial_conditions(t):
            return [5, 10, 1, 0, 0, 0, 0, 20, 40000, 0, 0]
        
        # Run the simulation     
        result = ddeint(model, initial_conditions, self.times)

        # Truncate to specified range
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

        if fitting:
            # Find nearest indices
            indices = np.searchsorted(self.times, times)

            # Pull the filtered values
            filtered_output = result[indices]
            result = filtered_output

        return result

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.parameters[p] = parameters[p]

    def n_outputs(self):
        return 11

    def n_times(self):
        return self.length_model*self.num_days

    def n_parameters(self):
        return self.n_parameters_value

    def get_and_create_boundaries(self):
        lowerbounds = []
        upperbounds = []

        for item in self.parameters.keys():
            lowerbound, upperbound = self.parameter_boundaries.get(item, (None, None))
            if lowerbound is None or upperbound is None: 
                print(f'{item} has no defined bounds. Setting to default (0, 1000)')
                lowerbound, upperbound = (0, 1000)
            
            lowerbounds.append(lowerbound)
            upperbounds.append(upperbound)
        
        return pints.RectangularBoundaries(lowerbounds, upperbounds)

class HPAModelFEInterBothCBGAlbBloodISF(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 times,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()
        self.set_fix_parameters({})

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

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        K_a = self.parameters['K_a']
        K_f = self.parameters['K_f']
        k_mf = self.parameters['k_mf']
        k_me = self.parameters['k_me']
        V_f_b = self.parameters['V_f_b']
        V_e_b = self.parameters['V_e_b']
        V_f_i = self.parameters['V_f_i'] # new param
        V_e_i = self.parameters['V_e_i'] # new param
        k_1 = self.parameters['k_1']
        k_2 = self.parameters['k_2']
        k_3 = self.parameters['k_3']
        k_4 = self.parameters['k_4']
        k_5 = self.parameters['k_5']
        k_6 = self.parameters['k_6']
        k_7 = self.parameters['k_7']
        k_8 = self.parameters['k_8']
        k_BI = self.parameters['k_BI']
        V_B = self.parameters['V_B']
        V_I = self.parameters['V_I']
        alpha = self.parameters['alpha']
        delay = self.parameters['delay']
        lambda_s = self.parameters['lambda_s']
        lambda_a = self.parameters['lambda_a']
        t_s = self.parameters['t_s']
        m_a = self.parameters['m_a']
        m_f = self.parameters['m_f']
        sigma = self.parameters['sigma']
        gamma_a = self.parameters['gamma_a']
        gamma_f_b = self.parameters['gamma_f_b']
        gamma_e_b = self.parameters['gamma_e_b']
        gamma_f_i = self.parameters['gamma_f_i']
        gamma_e_i = self.parameters['gamma_e_i']

        # Define the DDE model
        def model(Y, t):
            A, F_B, E_B, F_CBG, F_Alb, E_CBG, E_Alb, CBG, Alb, F_I, E_I = Y(t)
            F_delay = Y(t - delay)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dF_Bdt = -(gamma_f_b+k_1*CBG+k_3*Alb)*F_B + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_2*F_CBG + k_4*F_Alb + (V_e_b*E_B)/(k_me+E_B) - (V_f_b+F_B)/(k_mf+F_B) - (k_BI/V_B)*(F_B-F_I)
            dE_Bdt = -(gamma_e_b+k_5*CBG+k_7*Alb)*E_B + k_6*E_CBG + k_8*E_Alb - (V_e_b*E_B)/(k_me+E_B) + (V_f_b+F_B)/(k_mf+F_B) - (k_BI/V_B)*(E_B-E_I)
            dF_CBGdt = k_1*F_B*CBG - k_2*F_CBG
            dF_Albdt = k_3*F_B*Alb - k_4*F_Alb
            dE_CBGdt = k_5*E_B*CBG - k_6*E_CBG
            dE_Albdt = k_7*E_B*Alb - k_8*E_Alb
            dCBGdt = k_2*F_CBG - k_1*F_B*CBG + k_6*E_CBG - k_5*E_B*CBG
            dAlbdt = k_4*F_Alb - k_3*F_B*Alb + k_8*E_Alb - k_7*E_B*Alb
            dF_Idt = (k_BI/V_I)*(F_B-F_I) - gamma_f_i*F_I + (V_e_i*E_I)/(k_me+E_I) - (V_f_i+F_I)/(k_mf+F_I)
            dE_Idt = (k_BI/V_I)*(E_B-E_I) - gamma_e_i*E_I - (V_e_i*E_I)/(k_me+E_I) + (V_f_i+F_I)/(k_mf+F_I)

            return [dAdt, dF_Bdt, dE_Bdt, dF_CBGdt, dF_Albdt, dE_CBGdt, dE_Albdt, dCBGdt, dAlbdt, dF_Idt, dE_Idt]

        # Define initial conditions
        def initial_conditions(t):
            return [5, 10, 1, 0, 0, 0, 0, 20, 40000, 0, 0]
        
        # Run the simulation     
        result = ddeint(model, initial_conditions, self.times)

        # Truncate to specified range
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

        if fitting:
            # Find nearest indices
            indices = np.searchsorted(self.times, times)

            # Pull the filtered values
            filtered_output = result[indices]
            result = filtered_output

        return result

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.parameters[p] = parameters[p]

    def n_outputs(self):
        return 11

    def n_times(self):
        return self.length_model*self.num_days

    def n_parameters(self):
        return self.n_parameters_value

    def get_and_create_boundaries(self):
        lowerbounds = []
        upperbounds = []

        for item in self.parameters.keys():
            lowerbound, upperbound = self.parameter_boundaries.get(item, (None, None))
            if lowerbound is None or upperbound is None: 
                print(f'{item} has no defined bounds. Setting to default (0, 1000)')
                lowerbound, upperbound = (0, 1000)
            
            lowerbounds.append(lowerbound)
            upperbounds.append(upperbound)
        
        return pints.RectangularBoundaries(lowerbounds, upperbounds)

class HPAModelCRHSupp(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 times,
                 num_days=6,
                 days_to_keep=1,
                 step=0.1):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()
        self.set_fix_parameters({})

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

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        param_keys = list(self.parameters.keys())
        for i, key in enumerate(param_keys):
            self.parameters[key] = parameters[i]

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        K_a = self.parameters['K_a']
        K_c = self.parameters['K_c']
        K_h = self.parameters['K_h']
        alpha = self.parameters['alpha']
        delay_a = self.parameters['delay_a']
        delay_h = self.parameters['delay_h']
        lambda_s = self.parameters['lambda_s']
        lambda_a = self.parameters['lambda_a']
        t_s = self.parameters['t_s']
        m_a = self.parameters['m_a']
        m_c = self.parameters['m_c']
        m_h = self.parameters['m_h']
        sigma = self.parameters['sigma']
        gamma_a = self.parameters['gamma_a']
        gamma_c = self.parameters['gamma_c']
        gamma_h = self.parameters['gamma_h']

        # Define the DDE model
        def model(Y, t):
            A, C, H = Y(t)
            C_delay_a = Y(t - delay_a)[1]
            C_delay_h = Y(t - delay_h)[1]

            dAdt = -gamma_a*A + ((K_c**m_a)*H)/(K_c**m_a+C_delay_a**m_a)
            dCdt = -gamma_c*C + alpha*((A**m_c)/(K_a**m_c + A**m_c))
            dHdt = -gamma_h*H + ((K_h**m_h)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_h**m_h + C_delay_h**m_h)

            return [dAdt, dCdt, dHdt]

        # Define initial conditions
        def initial_conditions(t):
            return [5, 400, 300]
        
        # Run the simulation     
        result = ddeint(model, initial_conditions, self.times)

        # Truncate to specified range
        result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]

        if fitting:
            # Find nearest indices
            indices = np.searchsorted(self.times, times)

            # Pull the filtered values
            filtered_output = result[indices]
            result = filtered_output

        return result

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.parameters[p] = parameters[p]

    def n_outputs(self):
        return 3

    def n_times(self):
        return self.length_model*self.num_days

    def n_parameters(self):
        return self.n_parameters_value

    def get_and_create_boundaries(self):
        lowerbounds = []
        upperbounds = []

        for item in self.parameters.keys():
            lowerbound, upperbound = self.parameter_boundaries.get(item, (None, None))
            if lowerbound is None or upperbound is None: 
                print(f'{item} has no defined bounds. Setting to default (0, 1000)')
                lowerbound, upperbound = (0, 1000)
            
            lowerbounds.append(lowerbound)
            upperbounds.append(upperbound)
        
        return pints.RectangularBoundaries(lowerbounds, upperbounds)