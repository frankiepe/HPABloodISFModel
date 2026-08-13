import math
import pints
from ddeint import ddeint
import numpy as np
from . import day_len, PARAMETER_BOUNDARIES
from scipy import signal as scipy_signal
from juliacall import Main as jl
from juliacall import JuliaError
jl.seval('using Pkg; Pkg.activate(".")')
jl.seval("using HPADDEModels")
jl.seval("using DelayDiffEq, DifferentialEquations, JSON")
HPADDEModels = jl.seval("HPADDEModels")

class BaseHPAModel(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 fixed_pars,
                 init_conds,
                 times,
                 signal_range = (7,13),
                 num_days=6,
                 days_to_keep=1,
                 step=0.1,
                 reject=True):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.signal_range = signal_range
        self.reject = reject
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.param_keys = list(self.parameters.keys())
        self.fixed_pars = fixed_pars
        self.all_pars = list(parameters.keys()) + list(fixed_pars.keys())
        self.init_conds = init_conds
        self.times = times
        self.tspan = (0.0, day_len*num_days)
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()
        self.alg = jl.MethodOfSteps(jl.Vern6())
        self.model = HPADDEModels.BaseHPAModel
        self.truncate_idx = int((self.length_model/self.step)*(self.num_days-self.days_to_keep))

    def crh(self, t, t_s=None, lambda_a=None, lambda_s=None, sigma=None, T_c=day_len, symmetric=False):
            if symmetric:
                return 70*math.cos(2*math.pi*(t/T_c)) + 75
    
            def _resolve(name, value):
                if value is not None:
                    return value
                if name in self.parameters:
                    return self.parameters[name]
                if name in self.fixed_pars:
                    return self.fixed_pars[name]
                raise KeyError(f"CRH parameter '{name}' is missing")
    
            t_s = _resolve('t_s', t_s)
            lambda_a = _resolve('lambda_a', lambda_a)
            lambda_s = _resolve('lambda_s', lambda_s)
            sigma = _resolve('sigma', sigma)
    
            return lambda_a * math.exp(
                lambda_s * math.cos(2*math.pi * ((t - t_s) / T_c)
                                    + sigma * math.cos(2*math.pi * ((t - t_s) / T_c)))
            )

    def simulate(self, parameters, times, fitting=True):
        # Assign parameters
        for i, key in enumerate(self.param_keys):
            self.parameters[key] = parameters[i]

        par_dict = {}
        for par in self.all_pars:
            if par in self.parameters:
                par_dict[par] = self.parameters[par]
            elif par in self.fixed_pars:
                par_dict[par] = self.fixed_pars[par]

        # HPC parameters
        gamma_a = par_dict['gamma_a'] # ACTH degradation rate
        gamma_c = par_dict['gamma_c'] # Cortisol degradation rate
        K_a = par_dict['K_a'] # ACTH receptor half-saturation constant
        K_c = par_dict['K_c'] # Cortisol receptor half-saturation constant
        m_a = par_dict['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_c = par_dict['m_c'] # Hill coefficient for CORT feedback
        tau = par_dict['tau'] # Feedback delay from CORT to ACTH
        alpha = par_dict['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = par_dict['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = par_dict['lambda_s'] # Circadian modulation strength
        t_s = par_dict['t_s'] # Circadian phase shift
        sigma = par_dict['sigma'] # Asymmetry of circadian drive

        # Initial conditions
        A_0 = self.init_conds['A']
        C_0 = self.init_conds['C']

        # Define the DDE model
        #def model(Y, t):
        #    A, C = Y(t)
        #    C_delay = Y(t - tau)[1]

        #    dAdt = -gamma_a*A + ((K_c**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_c**m_a+C_delay**m_a)
        #    dCdt = -gamma_c*C + alpha*((A**m_c)/(K_a**m_c + A**m_c))

        #    return [dAdt, dCdt]

        # Define initial conditions
        #def initial_conditions(t):
        #    return [A_0, C_0]
        
        # Run the simulation  
        #result = ddeint(model, initial_conditions, self.times)
        
        u0 = [A_0, C_0]
        jl.seval(f"h(p, t) = [{A_0}, {C_0}]")
        h = jl.h
        
        lags = [tau]
        p = (gamma_a, gamma_c, K_a, K_c, m_a, m_c, tau, alpha, lambda_a, lambda_s, t_s, sigma)
        prob = jl.DDEProblem(self.model, u0, h, self.tspan, p, constant_lags = lags, saveat = self.times)
        try:
            result = jl.solve(prob, self.alg, reltol=1e-6, abstol=1e-6)
        except JuliaError:
            result = np.full((len(self.init_conds), len(self.times)), 5000) 
        result = np.asarray(jl.transpose(result[:, self.truncate_idx:]))
        
        # Truncate to specified range
        #result = result[int((self.length_model/self.step)*(self.num_days-self.days_to_keep)):]
        
        if fitting:
            # Find nearest indices
            indices = np.searchsorted(self.times, times)

            # Pull the filtered values
            filtered_output = result[indices]
            result = filtered_output

        if (self.reject == True):
            if (self.reject_parameter_combination(result)):
                return np.full((len(result), np.shape(result)[1]), 5000)
        
        return result

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

    # Function to reject parameter combination if number of peaks are outside a plausible range
    def reject_parameter_combination(self, result): 
        for i in range(result.shape[1]):
            signals, _ = scipy_signal.find_peaks(result[:, i])
            number_of_signals = len(signals)

            lower_bound, upper_bound = self.signal_range

            if not (lower_bound <= number_of_signals <= upper_bound): 
                return True
        
        return False
    
class HPAModelFEInter(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 fixed_pars,
                 init_conds,
                 times,
                 signal_range = (7,13),
                 num_days=6,
                 days_to_keep=1,
                 step=0.1,
                 reject=True):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.signal_range = signal_range
        self.reject = reject
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.param_keys = list(self.parameters.keys())
        self.fixed_pars = fixed_pars
        self.all_pars = list(parameters.keys()) + list(fixed_pars.keys())
        self.init_conds = init_conds
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()

    def crh(self, t, t_s=None, lambda_a=None, lambda_s=None, sigma=None, T_c=day_len, symmetric=False):
        if symmetric:
            return 70*math.cos(2*math.pi*(t/T_c)) + 75

        def _resolve(name, value):
            if value is not None:
                return value
            if name in self.parameters:
                return self.parameters[name]
            if name in self.fixed_pars:
                return self.fixed_pars[name]
            raise KeyError(f"CRH parameter '{name}' is missing")

        t_s = _resolve('t_s', t_s)
        lambda_a = _resolve('lambda_a', lambda_a)
        lambda_s = _resolve('lambda_s', lambda_s)
        sigma = _resolve('sigma', sigma)

        return lambda_a * math.exp(
            lambda_s * math.cos(2*math.pi * ((t - t_s) / T_c)
                                + sigma * math.cos(2*math.pi * ((t - t_s) / T_c)))
        )

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        for i, key in enumerate(self.param_keys):
            self.parameters[key] = parameters[i]

        par_dict = {}
        for par in self.all_pars:
            if par in self.parameters:
                par_dict[par] = self.parameters[par]
            elif par in self.fixed_pars:
                par_dict[par] = self.fixed_pars[par]

        # HPC parameters
        gamma_a = par_dict['gamma_a'] # ACTH degradation rate
        gamma_f = par_dict['gamma_f'] # Cortisol degradation rate
        gamma_e = par_dict['gamma_e'] # Cortisone degradation rate (new param)
        K_a = par_dict['K_a'] # ACTH receptor half-saturation constant
        K_f = par_dict['K_f'] # Cortisol receptor half-saturation constant
        K_mf = par_dict['K_mf'] # Cortisol conc. when F->E reaction rate is half V_f (new param)
        K_me = par_dict['K_me'] # Cortisone conc. when E->F reaction rate is half V_e (new param)
        m_a = par_dict['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_f = par_dict['m_f'] # Hill coefficient for CORT feedback
        V_f = par_dict['V_f'] # Max. cortisol to cortisone rate (new param)
        V_e = par_dict['V_e'] # Max. cortisone to cortisol rate (new param)
        tau = par_dict['tau'] # Feedback delay from CORT to ACTH
        alpha = par_dict['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = par_dict['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = par_dict['lambda_s'] # Circadian modulation strength
        t_s = par_dict['t_s'] # Circadian phase shift
        sigma = par_dict['sigma'] # Asymmetry of circadian drive

        # Initial conditions
        A_0 = self.init_conds['A']
        F_0 = self.init_conds['F']
        E_0 = self.init_conds['E']

        # Define the DDE model
        def model(Y, t):
            A, F, E = Y(t)
            F_delay = Y(t - tau)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dFdt = -gamma_f*F + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + (V_e*E)/(K_me+E) - (V_f+F)/(K_mf+F)
            dEdt = -gamma_e*E - (V_e*E)/(K_me+E) + (V_f+F)/(K_mf+F)

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

        if (self.reject == True):
            if (self.reject_parameter_combination(result)):
                return np.full((len(result), np.shape(result)[1]), 5000)

        return result

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

    # Function to reject parameter combination if number of peaks are outside a plausible range
    def reject_parameter_combination(self, result): 
        for i in range(result.shape[1]):
            signals, _ = scipy_signal.find_peaks(result[:, i])
            number_of_signals = len(signals)

            lower_bound, upper_bound = self.signal_range

            if not (lower_bound <= number_of_signals <= upper_bound): 
                return True
        
        return False

class HPAModelFEInterCBGAlbSimple(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 fixed_pars,
                 init_conds,
                 times,
                 signal_range = (7,13),
                 num_days=6,
                 days_to_keep=1,
                 step=0.1,
                 reject=True):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.signal_range = signal_range
        self.reject = reject
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.param_keys = list(self.parameters.keys())
        self.fixed_pars = fixed_pars
        self.all_pars = list(parameters.keys()) + list(fixed_pars.keys())
        self.init_conds = init_conds
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()

    def crh(self, t, t_s=None, lambda_a=None, lambda_s=None, sigma=None, T_c=day_len, symmetric=False):
            if symmetric:
                return 70*math.cos(2*math.pi*(t/T_c)) + 75
    
            def _resolve(name, value):
                if value is not None:
                    return value
                if name in self.parameters:
                    return self.parameters[name]
                if name in self.fixed_pars:
                    return self.fixed_pars[name]
                raise KeyError(f"CRH parameter '{name}' is missing")
    
            t_s = _resolve('t_s', t_s)
            lambda_a = _resolve('lambda_a', lambda_a)
            lambda_s = _resolve('lambda_s', lambda_s)
            sigma = _resolve('sigma', sigma)
    
            return lambda_a * math.exp(
                lambda_s * math.cos(2*math.pi * ((t - t_s) / T_c)
                                    + sigma * math.cos(2*math.pi * ((t - t_s) / T_c)))
            )

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        for i, key in enumerate(self.param_keys):
            self.parameters[key] = parameters[i]

        par_dict = {}
        for par in self.all_pars:
            if par in self.parameters:
                par_dict[par] = self.parameters[par]
            elif par in self.fixed_pars:
                par_dict[par] = self.fixed_pars[par]

        # HPC parameters
        gamma_a = par_dict['gamma_a'] # ACTH degradation rate
        gamma_f = par_dict['gamma_f'] # Cortisol degradation rate
        gamma_e = par_dict['gamma_e'] # Cortisone degradation rate
        K_a = par_dict['K_a'] # ACTH receptor half-saturation constant
        K_f = par_dict['K_f'] # Cortisol receptor half-saturation constant
        K_mf = par_dict['K_mf'] # Cortisol conc. when F->E reaction rate is half V_f
        K_me = par_dict['K_me'] # Cortisone conc. when E->F reaction rate is half V_e
        k_Fon = par_dict['k_Fon'] # F protein on-binding rate (new param)
        k_Foff = par_dict['k_Foff'] # F protein off-binding rate (new param)
        k_Eon = par_dict['k_Eon'] # E protein on-binding rate (new param)
        k_Eoff = par_dict['k_Eoff'] # E protein off-binding rate (new param)
        m_a = par_dict['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_f = par_dict['m_f'] # Hill coefficient for CORT feedback
        V_f = par_dict['V_f'] # Max. cortisol to cortisone rate
        V_e = par_dict['V_e'] # Max. cortisone to cortisol rate
        tau = par_dict['tau'] # Feedback delay from CORT to ACTH
        alpha = par_dict['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = par_dict['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = par_dict['lambda_s'] # Circadian modulation strength
        t_s = par_dict['t_s'] # Circadian phase shift
        sigma = par_dict['sigma'] # Asymmetry of circadian drive

        # Initial conditions
        A_0 = self.init_conds['A']
        F_0 = self.init_conds['F']
        E_0 = self.init_conds['E']
        F_bound_0 = self.init_conds['F_bound']
        E_bound_0 = self.init_conds['E_bound']

        # Define the DDE model
        def model(Y, t):
            A, F, E, F_bound, E_bound = Y(t)
            F_delay = Y(t - tau)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dFdt = -(gamma_f+k_Fon)*F + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_Foff*F_bound + \
                (V_e*E)/(K_me+E) - (V_f+F)/(K_mf+F)
            dEdt = -(gamma_e+k_Eon)*E + k_Eoff*E_bound - (V_e*E)/(K_me+E) + (V_f+F)/(K_mf+F)
            dF_bounddt = k_Fon*F - k_Foff*F_bound
            dE_bounddt = k_Eon*E - k_Eoff*E_bound

            return [dAdt, dFdt, dEdt, dF_bounddt, dE_bounddt]

        # Define initial conditions
        def initial_conditions(t):
            return [A_0, F_0, E_0, F_bound_0, E_bound_0]
        
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

        if (self.reject == True):
            if (self.reject_parameter_combination(result)):
                return np.full((len(result), np.shape(result)[1]), 5000)
            
        return result

    def n_outputs(self):
        return 5

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

    # Function to reject parameter combination if number of peaks are outside a plausible range
    def reject_parameter_combination(self, result): 
        for i in range(result.shape[1]):
            signals, _ = scipy_signal.find_peaks(result[:, i])
            number_of_signals = len(signals)

            lower_bound, upper_bound = self.signal_range

            if not (lower_bound <= number_of_signals <= upper_bound): 
                return True
        
        return False

class HPAModelFEInterCBGAlb(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 fixed_pars,
                 init_conds,
                 times,
                 signal_range = (7,13),
                 num_days=6,
                 days_to_keep=1,
                 step=0.1,
                 reject=True):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.signal_range = signal_range
        self.reject = reject
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.param_keys = list(self.parameters.keys())
        self.fixed_pars = fixed_pars
        self.all_pars = list(parameters.keys()) + list(fixed_pars.keys())
        self.init_conds = init_conds
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()

    def crh(self, t, t_s=None, lambda_a=None, lambda_s=None, sigma=None, T_c=day_len, symmetric=False):
            if symmetric:
                return 70*math.cos(2*math.pi*(t/T_c)) + 75
    
            def _resolve(name, value):
                if value is not None:
                    return value
                if name in self.parameters:
                    return self.parameters[name]
                if name in self.fixed_pars:
                    return self.fixed_pars[name]
                raise KeyError(f"CRH parameter '{name}' is missing")
    
            t_s = _resolve('t_s', t_s)
            lambda_a = _resolve('lambda_a', lambda_a)
            lambda_s = _resolve('lambda_s', lambda_s)
            sigma = _resolve('sigma', sigma)
    
            return lambda_a * math.exp(
                lambda_s * math.cos(2*math.pi * ((t - t_s) / T_c)
                                    + sigma * math.cos(2*math.pi * ((t - t_s) / T_c)))
            )

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        for i, key in enumerate(self.param_keys):
            self.parameters[key] = parameters[i]

        par_dict = {}
        for par in self.all_pars:
            if par in self.parameters:
                par_dict[par] = self.parameters[par]
            elif par in self.fixed_pars:
                par_dict[par] = self.fixed_pars[par]

        # HPC parameters
        gamma_a = par_dict['gamma_a'] # ACTH degradation rate
        gamma_f = par_dict['gamma_f'] # Cortisol degradation rate
        gamma_e = par_dict['gamma_e'] # Cortisone degradation rate
        K_a = par_dict['K_a'] # ACTH receptor half-saturation constant
        K_f = par_dict['K_f'] # Cortisol receptor half-saturation constant
        K_mf = par_dict['K_mf'] # Cortisol conc. when F->E reaction rate is half V_f
        K_me = par_dict['K_me'] # Cortisone conc. when E->F reaction rate is half V_e
        k_1 = par_dict['k_1'] # F:CBG on-binding rate (new param)
        k_2 = par_dict['k_2'] # F:CBG off-binding rate (new param)
        k_3 = par_dict['k_3'] # F:Alb on-binding rate (new param)
        k_4 = par_dict['k_4'] # F:Alb off-binding rate (new param)
        k_5 = par_dict['k_5'] # E:CBG on-binding rate (new param)
        k_6 = par_dict['k_6'] # E:CBG off-binding rate (new param)
        k_7 = par_dict['k_7'] # E:Alb on-binding rate (new param)
        k_8 = par_dict['k_8'] # E:Alb off-binding rate (new param)
        m_a = par_dict['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_f = par_dict['m_f'] # Hill coefficient for CORT feedback
        V_f = par_dict['V_f'] # Max. cortisol to cortisone rate
        V_e = par_dict['V_e'] # Max. cortisone to cortisol rate
        tau = par_dict['tau'] # Feedback delay from CORT to ACTH
        alpha = par_dict['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = par_dict['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = par_dict['lambda_s'] # Circadian modulation strength
        t_s = par_dict['t_s'] # Circadian phase shift
        sigma = par_dict['sigma'] # Asymmetry of circadian drive

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
            F_delay = Y(t - tau)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dFdt = -(gamma_f+k_1*CBG+k_3*Alb)*F + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_2*F_CBG + k_4*F_Alb + \
                (V_e*E)/(K_me+E) - (V_f+F)/(K_mf+F)
            dEdt = -(gamma_e+k_5*CBG+k_7*Alb)*E + k_6*E_CBG + k_8*E_Alb - (V_e*E)/(K_me+E) + (V_f+F)/(K_mf+F)
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

        if (self.reject == True):
            if (self.reject_parameter_combination(result)):
                return np.full((len(result), np.shape(result)[1]), 5000)

        return result

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

    # Function to reject parameter combination if number of peaks are outside a plausible range
    def reject_parameter_combination(self, result): 
        for i in range(result.shape[1]):
            signals, _ = scipy_signal.find_peaks(result[:, i])
            number_of_signals = len(signals)

            lower_bound, upper_bound = self.signal_range

            if not (lower_bound <= number_of_signals <= upper_bound): 
                return True
        
        return False
    
class HPAModelFEInterCBGAlbBloodISF(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 fixed_pars,
                 init_conds,
                 times,
                 signal_range = (7,13),
                 num_days=6,
                 days_to_keep=1,
                 step=0.1,
                 reject=True):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.signal_range = signal_range
        self.reject = reject
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.param_keys = list(self.parameters.keys())
        self.fixed_pars = fixed_pars
        self.all_pars = list(parameters.keys()) + list(fixed_pars.keys())
        self.init_conds = init_conds
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()

    def crh(self, t, t_s=None, lambda_a=None, lambda_s=None, sigma=None, T_c=day_len, symmetric=False):
            if symmetric:
                return 70*math.cos(2*math.pi*(t/T_c)) + 75
    
            def _resolve(name, value):
                if value is not None:
                    return value
                if name in self.parameters:
                    return self.parameters[name]
                if name in self.fixed_pars:
                    return self.fixed_pars[name]
                raise KeyError(f"CRH parameter '{name}' is missing")
    
            t_s = _resolve('t_s', t_s)
            lambda_a = _resolve('lambda_a', lambda_a)
            lambda_s = _resolve('lambda_s', lambda_s)
            sigma = _resolve('sigma', sigma)
    
            return lambda_a * math.exp(
                lambda_s * math.cos(2*math.pi * ((t - t_s) / T_c)
                                    + sigma * math.cos(2*math.pi * ((t - t_s) / T_c)))
            )

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        for i, key in enumerate(self.param_keys):
            self.parameters[key] = parameters[i]

        par_dict = {}
        for par in self.all_pars:
            if par in self.parameters:
                par_dict[par] = self.parameters[par]
            elif par in self.fixed_pars:
                par_dict[par] = self.fixed_pars[par]

        # HPC parameters
        gamma_a = par_dict['gamma_a'] # ACTH degradation rate
        gamma_f_b = par_dict['gamma_f_b'] # Cortisol degradation rate in BP
        gamma_f_i = par_dict['gamma_f_i'] # Cortisol degradation rate in ISF (new param)
        gamma_e_b = par_dict['gamma_e_b'] # Cortisone degradation rate in BP
        gamma_e_i = par_dict['gamma_e_i'] # Cortisone degradation rate in ISF (new param)
        K_a = par_dict['K_a'] # ACTH receptor half-saturation constant
        K_f = par_dict['K_f'] # Cortisol receptor half-saturation constant
        K_mf = par_dict['K_mf'] # Cortisol conc. when F->E reaction rate is half V_f
        K_me = par_dict['K_me'] # Cortisone conc. when E->F reaction rate is half V_e
        k_Fon = par_dict['k_Fon'] # F protein on-binding rate
        k_Foff = par_dict['k_Foff'] # F protein off-binding rate
        k_Eon = par_dict['k_Eon'] # E protein on-binding rate
        k_Eoff = par_dict['k_Eoff'] # E protein off-binding rate
        k_BI = par_dict['k_BI'] # Permeability constant (new param)
        m_a = par_dict['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_f = par_dict['m_f'] # Hill coefficient for CORT feedback
        V_f = par_dict['V_f'] # Max. cortisol to cortisone rate
        V_e = par_dict['V_e'] # Max. cortisone to cortisol rate
        V_B = par_dict['V_B'] # Vascular distribution volume (new param)
        V_I = par_dict['V_I'] # ISF distribution volume (new param)
        tau = par_dict['tau'] # Feedback delay from CORT to ACTH
        alpha = par_dict['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = par_dict['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = par_dict['lambda_s'] # Circadian modulation strength
        t_s = par_dict['t_s'] # Circadian phase shift
        sigma = par_dict['sigma'] # Asymmetry of circadian drive

        # Initial conditions
        A_0 = self.init_conds['A']
        F_B_0 = self.init_conds['F_B']
        E_B_0 = self.init_conds['E_B']
        F_bound_0 = self.init_conds['F_bound']
        E_bound_0 = self.init_conds['E_bound']
        F_I_0 = self.init_conds['F_I']
        E_I_0 = self.init_conds['E_I']

        # Define the DDE model
        def model(Y, t):
            A, F_B, E_B, F_bound, E_bound, F_I, E_I = Y(t)
            F_delay = Y(t - tau)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dF_Bdt = -(gamma_f_b+k_Fon)*F_B + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_Foff*F_bound + \
                (V_e*E_B)/(K_me+E_B) - (V_f+F_B)/(K_mf+F_B) - (k_BI/V_B)*(F_B-F_I)
            dE_Bdt = -(gamma_e_b+k_Eon)*E_B + k_Eoff*E_bound - (V_e*E_B)/(K_me+E_B) + \
                (V_f+F_B)/(K_mf+F_B) - (k_BI/V_B)*(E_B-E_I)
            dF_bounddt = k_Fon*F_B - k_Foff*F_bound
            dE_bounddt = k_Eon*E_B - k_Eoff*E_bound
            dF_Idt = (k_BI/V_I)*(F_B-F_I) - gamma_f_i*F_I
            dE_Idt = (k_BI/V_I)*(E_B-E_I) - gamma_e_i*E_I

            return [dAdt, dF_Bdt, dE_Bdt, dF_bounddt, dE_bounddt, dF_Idt, dE_Idt]

        # Define initial conditions
        def initial_conditions(t):
            return [A_0, F_B_0, E_B_0, F_bound_0, E_bound_0, F_I_0, E_I_0]
        
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

        if (self.reject == True):
            if (self.reject_parameter_combination(result)):
                return np.full((len(result), np.shape(result)[1]), 5000)

        return result

    def n_outputs(self):
        return 7

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

    # Function to reject parameter combination if number of peaks are outside a plausible range
    def reject_parameter_combination(self, result): 
        for i in range(result.shape[1]):
            signals, _ = scipy_signal.find_peaks(result[:, i])
            number_of_signals = len(signals)

            lower_bound, upper_bound = self.signal_range

            if not (lower_bound <= number_of_signals <= upper_bound): 
                return True
        
        return False

class HPAModelFEInterBothCBGAlbBloodISF(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 init_conds,
                 fixed_pars,
                 times,
                 signal_range = (7,13),
                 num_days=6,
                 days_to_keep=1,
                 step=0.1,
                 reject=True):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.signal_range = signal_range
        self.reject = reject
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.param_keys = list(self.parameters.keys())
        self.fixed_pars = fixed_pars
        self.all_pars = list(parameters.keys()) + list(fixed_pars.keys())
        self.init_conds = init_conds
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()

    def crh(self, t, t_s=None, lambda_a=None, lambda_s=None, sigma=None, T_c=day_len, symmetric=False):
            if symmetric:
                return 70*math.cos(2*math.pi*(t/T_c)) + 75
    
            def _resolve(name, value):
                if value is not None:
                    return value
                if name in self.parameters:
                    return self.parameters[name]
                if name in self.fixed_pars:
                    return self.fixed_pars[name]
                raise KeyError(f"CRH parameter '{name}' is missing")
    
            t_s = _resolve('t_s', t_s)
            lambda_a = _resolve('lambda_a', lambda_a)
            lambda_s = _resolve('lambda_s', lambda_s)
            sigma = _resolve('sigma', sigma)
    
            return lambda_a * math.exp(
                lambda_s * math.cos(2*math.pi * ((t - t_s) / T_c)
                                    + sigma * math.cos(2*math.pi * ((t - t_s) / T_c)))
            )

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        for i, key in enumerate(self.param_keys):
            self.parameters[key] = parameters[i]

        par_dict = {}
        for par in self.all_pars:
            if par in self.parameters:
                par_dict[par] = self.parameters[par]
            elif par in self.fixed_pars:
                par_dict[par] = self.fixed_pars[par]

        # HPC parameters
        gamma_a = par_dict['gamma_a'] # ACTH degradation rate
        gamma_f_b = par_dict['gamma_f_b'] # Cortisol degradation rate in BP
        gamma_f_i = par_dict['gamma_f_i'] # Cortisol degradation rate in ISF
        gamma_e_b = par_dict['gamma_e_b'] # Cortisone degradation rate in BP
        gamma_e_i = par_dict['gamma_e_i'] # Cortisone degradation rate in ISF
        K_a = par_dict['K_a'] # ACTH receptor half-saturation constant
        K_f = par_dict['K_f'] # Cortisol receptor half-saturation constant
        K_mfB = par_dict['K_mfB'] # Cortisol conc. when F->E reaction rate is half V_f_b in BP
        K_meB = par_dict['K_meB'] # Cortisone conc. when E->F reaction rate is half V_e_b in BP
        K_mfI = par_dict['K_mfI'] # Cortisol conc. when F->E reaction rate is half V_f_i in ISF (new param)
        K_meI = par_dict['K_meI'] # Cortisone conc. when E->F reaction rate is half V_e_i in ISF (new param)
        k_Fon = par_dict['k_Fon'] # F protein on-binding rate
        k_Foff = par_dict['k_Foff'] # F protein off-binding rate
        k_Eon = par_dict['k_Eon'] # E protein on-binding rate
        k_Eoff = par_dict['k_Eoff'] # E protein off-binding rate
        k_BI = par_dict['k_BI'] # Permeability constant
        m_a = par_dict['m_a'] # Hill coefficient for ACTH-driven CORT production
        m_f = par_dict['m_f'] # Hill coefficient for CORT feedback
        V_f_b = par_dict['V_f_b'] # Max. cortisol to cortisone rate in BP
        V_e_b = par_dict['V_e_b'] # Max. cortisone to cortisol rate in BP
        V_f_i = par_dict['V_f_i'] # Max. cortisol to cortisone rate in ISF (new param)
        V_e_i = par_dict['V_e_i'] # Max. cortisone to cortisol rate in ISF (new param)
        V_B = par_dict['V_B'] # Vascular distribution volume
        V_I = par_dict['V_I'] # ISF distribution volume
        tau = par_dict['tau'] # Feedback delay from CORT to ACTH
        alpha = par_dict['alpha'] # Maximal rate of ACTH-induced CORT production

        # CRH parameters
        lambda_a = par_dict['lambda_a'] # Baseline amplitude of CRH drive
        lambda_s = par_dict['lambda_s'] # Circadian modulation strength
        t_s = par_dict['t_s'] # Circadian phase shift
        sigma = par_dict['sigma'] # Asymmetry of circadian drive

        # Initial conditions
        A_0 = self.init_conds['A']
        F_B_0 = self.init_conds['F_B']
        E_B_0 = self.init_conds['E_B']
        F_bound_0 = self.init_conds['F_bound']
        E_bound_0 = self.init_conds['E_bound']
        F_I_0 = self.init_conds['F_I']
        E_I_0 = self.init_conds['E_I']

        # Define the DDE model
        def model(Y, t):
            A, F_B, E_B, F_bound, E_bound, F_I, E_I = Y(t)
            F_delay = Y(t - tau)[1]

            dAdt = -gamma_a*A + ((K_f**m_a)*self.crh(t, t_s, lambda_a, lambda_s, sigma))/(K_f**m_a+F_delay**m_a)
            dF_Bdt = -(gamma_f_b+k_Fon)*F_B + alpha*((A**m_f)/(K_a**m_f + A**m_f)) + k_Foff*F_bound + \
                (V_e_b*E_B)/(K_meB+E_B) - (V_f_b+F_B)/(K_mfB+F_B) - (k_BI/V_B)*(F_B-F_I)
            dE_Bdt = -(gamma_e_b+k_Eon)*E_B + k_Eoff*E_bound - (V_e_b*E_B)/(K_meB+E_B) + \
                (V_f_b+F_B)/(K_mfB+F_B) - (k_BI/V_B)*(E_B-E_I)
            dF_bounddt = k_Fon*F_B - k_Foff*F_bound
            dE_bounddt = k_Eon*E_B - k_Eoff*E_bound
            dF_Idt = (k_BI/V_I)*(F_B-F_I) - gamma_f_i*F_I + (V_e_i*E_I)/(K_meI+E_I) - (V_f_i+F_I)/(K_mfI+F_I)
            dE_Idt = (k_BI/V_I)*(E_B-E_I) - gamma_e_i*E_I - (V_e_i*E_I)/(K_meI+E_I) + (V_f_i+F_I)/(K_mfI+F_I)

            return [dAdt, dF_Bdt, dE_Bdt, dF_bounddt, dE_bounddt, dF_Idt, dE_Idt]

        # Define initial conditions
        def initial_conditions(t):
            return [A_0, F_B_0, E_B_0, F_bound_0, E_bound_0, F_I_0, E_I_0]
        
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

        if (self.reject == True):
            if (self.reject_parameter_combination(result)):
                return np.full((len(result), np.shape(result)[1]), 5000)

        return result

    def n_outputs(self):
        return 7

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

    # Function to reject parameter combination if number of peaks are outside a plausible range
    def reject_parameter_combination(self, result): 
        for i in range(result.shape[1]):
            signals, _ = scipy_signal.find_peaks(result[:, i])
            number_of_signals = len(signals)

            lower_bound, upper_bound = self.signal_range

            if not (lower_bound <= number_of_signals <= upper_bound): 
                return True
        
        return False

class HPAModelCRHSupp(pints.ForwardModel):
    def __init__(self,
                 parameters,
                 fixed_pars,
                 times,
                 signal_range = (7,13),
                 num_days=6,
                 days_to_keep=1,
                 step=0.1,
                 reject=True):
        self.num_days = num_days
        self.days_to_keep = days_to_keep
        self.step = step
        self.signal_range = signal_range
        self.reject = reject
        self.n_parameters_value = len(parameters)
        self.parameters = parameters
        self.param_keys = list(self.parameters.keys())
        self.fixed_pars = fixed_pars
        self.all_pars = list(parameters.keys()) + list(fixed_pars.keys())
        self.times = times
        self.length_model = day_len
        self.parameter_boundaries = PARAMETER_BOUNDARIES.copy()

    def crh(self, t, t_s=None, lambda_a=None, lambda_s=None, sigma=None, T_c=day_len, symmetric=False):
            if symmetric:
                return 70*math.cos(2*math.pi*(t/T_c)) + 75
    
            def _resolve(name, value):
                if value is not None:
                    return value
                if name in self.parameters:
                    return self.parameters[name]
                if name in self.fixed_pars:
                    return self.fixed_pars[name]
                raise KeyError(f"CRH parameter '{name}' is missing")
    
            t_s = _resolve('t_s', t_s)
            lambda_a = _resolve('lambda_a', lambda_a)
            lambda_s = _resolve('lambda_s', lambda_s)
            sigma = _resolve('sigma', sigma)
    
            return lambda_a * math.exp(
                lambda_s * math.cos(2*math.pi * ((t - t_s) / T_c)
                                    + sigma * math.cos(2*math.pi * ((t - t_s) / T_c)))
            )

    def simulate(self, parameters, times, fitting=True):
        
        # Assign parameters
        for i, key in enumerate(self.param_keys):
            self.parameters[key] = parameters[i]

        par_dict = {}
        for par in self.all_pars:
            if par in self.parameters:
                par_dict[par] = self.parameters[par]
            elif par in self.fixed_pars:
                par_dict[par] = self.fixed_pars[par]

        K_a = par_dict['K_a']
        K_c = par_dict['K_c']
        K_h = par_dict['K_h']
        alpha = par_dict['alpha']
        tau_a = par_dict['tau_a']
        tau_h = par_dict['tau_h']
        lambda_s = par_dict['lambda_s']
        lambda_a = par_dict['lambda_a']
        t_s = par_dict['t_s']
        m_a = par_dict['m_a']
        m_c = par_dict['m_c']
        m_h = par_dict['m_h']
        sigma = par_dict['sigma']
        gamma_a = par_dict['gamma_a']
        gamma_c = par_dict['gamma_c']
        gamma_h = par_dict['gamma_h']

        # Define the DDE model
        def model(Y, t):
            A, C, H = Y(t)
            C_delay_a = Y(t - tau_a)[1]
            C_delay_h = Y(t - tau_h)[1]

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

        if (self.reject == True):
            if (self.reject_parameter_combination(result)):
                return np.full((len(result), np.shape(result)[1]), 5000)

        return result

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

    # Function to reject parameter combination if number of peaks are outside a plausible range
    def reject_parameter_combination(self, result): 
        for i in range(result.shape[1]):
            signals, _ = scipy_signal.find_peaks(result[:, i])
            number_of_signals = len(signals)

            lower_bound, upper_bound = self.signal_range

            if not (lower_bound <= number_of_signals <= upper_bound): 
                return True
        
        return False