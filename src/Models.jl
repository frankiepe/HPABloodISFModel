module Models

export BaseHPAModel, HPAModelFEInter, HPAModelFEInterCBGAlbSimple, crh
export HPAModelFEInterCBGAlb, HPAModelFEInterCBGAlbBloodISF
export HPAModelFEInterBothCBGAlbBloodISF

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

function HPAModelFEInter(du, u, h, p, t)
    gamma_a, gamma_f, gamma_e, K_a, K_f, K_mf, K_me, m_a, m_f, V_f, V_e, tau, alpha, lambda_a, lambda_s, t_s, sigma = p
    F_delay = h(p, t - tau)[2]
    phi_t = crh(t, t_s, lambda_a, lambda_s, sigma)
    du[1] = -gamma_a * u[1] + ((K_f^m_a)*phi_t)/((K_f^m_a) + (F_delay^m_a))
    du[2] = -gamma_f * u[2] + alpha*((u[1]^m_f)/((K_a^m_f) + (u[1]^m_f))) + (V_e * u[3])/(K_me + u[3]) - (V_f + u[2])/(K_mf + u[2])
    du[3] = -gamma_e * u[3] - (V_e * u[3])/(K_me + u[3]) + (V_f + u[2])/(K_mf + u[2])
end

function HPAModelFEInterCBGAlbSimple(du, u, h, p, t)
    gamma_a, gamma_f, gamma_e, K_a, K_f, K_mf, K_me, k_Fon, k_Foff, k_Eon, k_Eoff, m_a, m_f, V_f, V_e, tau, alpha, lambda_a, lambda_s, t_s, sigma = p
    F_delay = h(p, t - tau)[2]
    phi_t = crh(t, t_s, lambda_a, lambda_s, sigma)
    du[1] = -gamma_a * u[1] + ((K_f^m_a)*phi_t)/((K_f^m_a) + (F_delay^m_a))
    du[2] = -(gamma_f + k_Fon) * u[2] + alpha*((u[1]^m_f)/((K_a^m_f) + (u[1]^m_f))) + k_Foff * u[4] + (V_e * u[3])/(K_me + u[3]) - (V_f + u[2])/(K_mf + u[2])
    du[3] = -(gamma_e + k_Eon) * u[3] + k_Eoff * u[5] - (V_e*u[3])/(K_me+u[3]) + (V_f+u[2])/(K_mf+u[2])
    du[4] = k_Fon * u[2] - k_Foff * u[4]
    du[5] = k_Eon * u[3] - k_Eoff * u[5]
end

function HPAModelFEInterCBGAlb(du, u, h, p, t)
    gamma_a, gamma_f, gamma_e, K_a, K_f, K_mf, K_me, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, m_a, m_f, V_f, V_e, tau, alpha, lambda_a, lambda_s, t_s, sigma = p
    F_delay = h(p, t - tau)[2]
    phi_t = crh(t, t_s, lambda_a, lambda_s, sigma)
    du[1] = -gamma_a * u[1] + ((K_f^m_a)*phi_t)/((K_f^m_a) + (F_delay^m_a))
    du[2] = -(gamma_f + k_1*u[8] + k_3*u[9]) * u[2] + alpha*((u[1]^m_f)/((K_a^m_f) + (u[1]^m_f))) + k_2 * u[4] + k_4 * u[5] + (V_e * u[3])/(K_me + u[3]) - (V_f + u[2])/(K_mf + u[2])
    du[3] = -(gamma_e + k_5*u[8] + k_7*u[9]) * u[3] + k_6 * u[6] + k_8 * u[7] - (V_e*u[3])/(K_me+u[3]) + (V_f+u[2])/(K_mf+u[2])
    du[4] = k_1 * u[2] * u[8] - k_2 * u[4]
    du[5] = k_3 * u[2] * u[9] - k_4 * u[5]
    du[6] = k_5 * u[3] * u[8] - k_6 * u[6]
    du[7] = k_7 * u[3] * u[9] - k_8 * u[7]
    du[8] = k_2 * u[4] - k_1 * u[2] * u[8] + k_6 * u[6] - k_5 * u[3] * u[8]
    du[9] = k_4 * u[5] - k_3 * u[2] * u[9] + k_8 * u[7] - k_7 * u[3] * u[9]
end

function HPAModelFEInterCBGAlbBloodISF(du, u, h, p, t)
    gamma_a, gamma_f_b, gamma_f_i, gamma_e_b, gamma_e_i, K_a, K_f, K_mf, K_me, k_Fon, k_Foff, k_Eon, k_Eoff, k_BI, m_a, m_f, V_f, V_e, 
                                                                            V_B, V_I, tau, alpha, lambda_a, lambda_s, t_s, sigma = p
    F_delay = h(p, t - tau)[2]
    phi_t = crh(t, t_s, lambda_a, lambda_s, sigma)
    du[1] = -gamma_a * u[1] + ((K_f^m_a)*phi_t)/((K_f^m_a) + (F_delay^m_a))
    du[2] = -(gamma_f_b + k_Fon) * u[2] + alpha*((u[1]^m_f)/((K_a^m_f) + (u[1]^m_f))) + k_Foff * u[4] + (V_e * u[3])/(K_me + u[3]) - (V_f + u[2])/(K_mf + u[2])
            - (k_BI/V_B)*(u[2]-u[6])
    du[3] = -(gamma_e_b + k_Eon) * u[3] + k_Eoff * u[5] - (V_e*u[3])/(K_me+u[3]) + (V_f+u[2])/(K_mf+u[2]) - (k_BI/V_B)*(u[3]-u[7])
    du[4] = k_Fon * u[2] - k_Foff * u[4]
    du[5] = k_Eon * u[3] - k_Eoff * u[5]
    du[6] = (k_BI/V_I)*(u[2]-u[6]) - gamma_f_i*u[6]
    du[7] = (k_BI/V_I)*(u[3]-u[7]) - gamma_e_i*u[7]
end

function HPAModelFEInterBothCBGAlbBloodISF(du, u, h, p, t)
    gamma_a, gamma_f_b, gamma_f_i, gamma_e_b, gamma_e_i, K_a, K_f, K_mfB, K_meB, K_mfI, K_meI, k_Fon, k_Foff, k_Eon, k_Eoff, k_BI, m_a, m_f, V_f_b,
                                                V_e_b, V_f_i, V_e_i, V_B, V_I, tau, alpha, lambda_a, lambda_s, t_s, sigma = p
    F_delay = h(p, t - tau)[2]
    phi_t = crh(t, t_s, lambda_a, lambda_s, sigma)
    du[1] = -gamma_a * u[1] + ((K_f^m_a)*phi_t)/((K_f^m_a) + (F_delay^m_a))
    du[2] = -(gamma_f_b + k_Fon) * u[2] + alpha*((u[1]^m_f)/((K_a^m_f) + (u[1]^m_f))) + k_Foff * u[4] + (V_e_b * u[3])/(K_meB + u[3]) - (V_f_b + u[2])/(K_mfB + u[2])
            - (k_BI/V_B)*(u[2]-u[6])
    du[3] = -(gamma_e_b + k_Eon) * u[3] + k_Eoff * u[5] - (V_e_b*u[3])/(K_meB+u[3]) + (V_f_b+u[2])/(K_mfB+u[2]) - (k_BI/V_B)*(u[3]-u[7])
    du[4] = k_Fon * u[2] - k_Foff * u[4]
    du[5] = k_Eon * u[3] - k_Eoff * u[5]
    du[6] = (k_BI/V_I)*(u[2]-u[6]) - gamma_f_i*u[6] + (V_e_i*u[7])/(K_meI+u[7]) - (V_f_i+u[6])/(K_mfI+u[6])
    du[7] = (k_BI/V_I)*(u[3]-u[7]) - gamma_e_i*u[7] - (V_e_i*u[7])/(K_meI+u[7]) + (V_f_i+u[6])/(K_mfI+u[6])
end

end