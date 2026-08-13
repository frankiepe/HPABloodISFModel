module Models

export BaseHPAModel, crh

function crh(t, t_s, lambda_a, lambda_s, sigma, T_c=1440)
    lambda_a * exp(lambda_s * cos(2*pi * ((t - t_s) / T_c) + sigma * cos(2*pi * ((t - t_s) / T_c))))
end

function BaseHPAModel(du, u, h, p, t)
    gamma_a, gamma_c, K_a, K_c, m_a, m_c, tau, alpha, lambda_a, lambda_s, t_s, sigma = p
    C_delay = h(p, t - tau)[2]
    phi_t = crh(t, t_s, lambda_a, lambda_s, sigma)
    du[1] = -gamma_a * u[1] + ((K_c^m_a)*phi_t)/((K_c^m_a) + (C_delay^m_a))
    du[2] = -gamma_c * u[2] + alpha*((u[1]^m_c)/((K_a^m_c) + (u[1]^m_c)))
end

end