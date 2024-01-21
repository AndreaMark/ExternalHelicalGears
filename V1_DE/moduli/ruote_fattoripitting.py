import numpy as np
################################################            FUNZIONI - ISO 6336-2            ################################################  
# Fattore di zona
def calcola_ZH(beta, alpha_t, alpha_wt):                    # eq 16
    """
    Questa funzione calcola il fattore di zona ZH.
    Args: 
        beta [rad]: (float) angolo d'elica.
        alpha_t [rad]: angolo di pressione trasversale di riferimento.
        alpha_wt [rad]: angolo di pressione trasversale di funzionamento.

    Returns: 
        ZH [\\]: fattore di zona.
    """
    num = np.cos(beta)*np.cos(alpha_wt)
    den = (np.cos(alpha_t)**2)*np.sin(alpha_wt)
    fraz = 2*(num/den)
    return np.sqrt(fraz)

# Rapporto di condotta trasversale
def calcola_eps_alpha(d_t_p, d_t_r, d_pp_p, d_pp_r, theta_l, PB) :              # internet
    """
    Questa funzione calcola il rapporto di condotta trasversale.

    Args: 
        d_t_p [mm]: (float), diametro di testa del pignone.
        d_t_r [mm]: diametro di testa della ruota.
        d_pp_p [mm]: diametro di piede del pignone.
        d_pp_r [mm]: diametro di piede della ruota.
        theta_l [rad]: angolo di lavoro (o di pressione nel caso di ruote non corrette).
        PB [mm]: fattore ausiliario.
    
    Returns: 
        eps_alpha: fattore di condotta trasversale.
    """
    g_in = 0.5*(np.sqrt(d_t_r**2 - d_pp_r**2) - d_pp_r*np.tan(theta_l))
    g_out = 0.5*(np.sqrt(d_t_p**2 - d_pp_p**2) - d_pp_p*np.tan(theta_l))
    eps_alpha = (g_in + g_out)/PB
    return eps_alpha

# fattore ausiliario per ZB
def calcola_M1(d_t_p, d_t_r, d_pp_p, d_pp_r, theta_l, PB, z, z_prime, alpha_wt):# eq 17
    """
    Questa funzione calcola il fattore ausiliario M1, utile per ricavare il fattore ZB quando u>1,5.

    Args: 
        d_t_p [mm]: (float), diametro di testa del pignone.
        d_t_r [mm]: diametro di testa della ruota.
        d_pp_p [mm]: diametro di piede del pignone.
        d_pp_r [mm]: diametro di piede della ruota.
        theta_l [rad]: angolo di lavoro (o di pressione nel caso di ruote non corrette).
        PB [mm]: fattore ausiliario.
        z [\\]: numero di denti del pignone.
        z_prime [\\]: numero di denti della ruota.
        alpha_wt [rad]: angolo di pressione trasversale di funzionamento.

    Returns:
        M1: fattore ausiliario.

    """
    eps_alpha = calcola_eps_alpha(d_t_p, d_t_r, d_pp_p, d_pp_r, theta_l, PB)
    fatt1 = np.sqrt((d_t_p/d_pp_p)**2 -1) - (2*np.pi/z)
    fatt2 = np.sqrt((d_t_r/d_pp_r)**2 -1) - (eps_alpha-1)*(2*np.pi/z_prime)
    M1 = np.tan(alpha_wt)/(np.sqrt(fatt1*fatt2))
    return M1

# fattore di lubrificazione
def calcola_ZL(CZL, v_40):                                                         # eq 37                                                                                                 
    """
    Calcola il fattore di lubrificazione ZL.
    Args: 
        CZL [\\]: fattore ausiliario eq 38,39,40.
        v_40 [mm2/s]: viscosità a 40 gradi centigradi.
    Returns: 
        ZL [\\]: fattore di lubrificazione.

    """
    dummy1 = 4*(1-CZL)
    dummy2 = (1.2 + (132/v_40))**2
    ZL = CZL + dummy1/dummy2
    return ZL

# Fattore  di velocità
def calcola_ZV(CZL, v):                                                         # eq 42
    """
    Questa funzione calcola il fattore di velocità.
    Args: 
        CZL [\\]: fattore ausiliario eq 38,39,40.
        v [m/s]: VELOCITA' LINEARE AL RAGGIO PRIMITIVO.
    Returns: 
        ZV: fattore di velocità.
    """
    CZV = CZL + 0.02                                                            # eq 43
    dummy3 = 2*(1-CZV)
    dummy4 = np.sqrt(0.8 + 32/v)
    ZV = CZV + dummy3/dummy4
    return ZV

# Curvatura relativa
def curvatura_relativa(d_pp_p, d_pp_r, alpha_wt):                               # eq 46
    """
    Questa funzione calcola il raggio curvatura relativa.
    Args: 
        d_pp_p [mm]: diametro di piede del pignone.
        d_pp_r [mm]: diametro di piede della ruota.
        alpha_wt [rad]: angolo di pressione trasversale di funzionamento.

    Returns: 
        rho_red [mm]: raggio di curvatura relativa.
    """
    rho_1 = 0.5*d_pp_p * np.tan(alpha_wt)                                       # eq 47
    rho_2 = 0.5*d_pp_r * np.tan(alpha_wt)
    rho_red = (rho_1*rho_2)/(rho_1 + rho_2)
    return rho_red

# Rugosità indotta
def calcola_RzH(Rz, rho_red, v_40, v):                                          # eq 53
    """
    Questa funzione calcola la rugosità ottenuta per funzionamento.
    Args: 
        Rz [mm]: Rugosità relativa.

        rho_red [mm]: raggio di curvatura relativa.

        v_40 [mm2/s]: viscosità a 40 gradi centigradi.

        v [m/s]: VELOCITA' LINEARE AL RAGGIO PRIMITIVO.

    Returns: 
        RHZ: fattore di rugosità relativo al funzionamento
    """
    dummy5 = Rz*(10/rho_red)**(0.33)
    dummy6 = (Rz/Rz)**(0.66)
    dummy7 = dummy5*dummy6
    dummy8 = ((v_40*v)/1500)**(0.33)
    return dummy7/dummy8

# Fattore di work hardening
def calcola_ZW(HB, RZH):                                    # eq 54
    """
    Questa funzione calcola il fattore di indurimento dovuto al funzionamento
    Args: 
        HB: durezza di Brinell.

        RHZ: fattore di rugosità relativo al funzionamento.
    
    Returns: 
        ZW: fattore di indurimento
    """
    dummy9 = 1.2 - (HB-130)/1700
    dummy10 = (3/RZH)**(0.15)
    return dummy9*dummy10