import numpy as np
################################################            FUNZIONI - ISO 6336-3            ################################################  
# Evolvente
def ev(angolo):
    """
    Questa funzione calcola l'evolvente di un angolo. 

    Args:
        angolo: valore angolare espresso in radianti.

    Returns:
        ev: valore della funzione evolvente.
    """
    return np.tan(angolo) - angolo

# Fattore ausiliario E
def calcola_E(modulo, u, theta_p, spr, rho_fp):             # eq 10
    """
    Questa funzione calcola il fattore ausiliario E per il calcolo del fattore di forma YF.

    Args:
        modulo [mm]: modulo unificato.

        u [mm]: dedendum della ruota.

        theta_p [rad]: angolo di pressione.

        spr [mm]: undercut of gears.

        rho_fp [mm]:Raggio di curvatura alla base del dente.
    
    Returns: 
        E [\\]: fattore ausiliario
    """
    fatt1 = (modulo*np.pi/4) - (u*np.tan(theta_p))
    fatt2 = (spr/np.cos(theta_p))
    fatt3 = (1 - np.sin(theta_p))*(rho_fp/np.cos(theta_p))
    E_factor = fatt1 + fatt2 - fatt3
    return E_factor

# Fattore ausiliario G
def calcola_G(modulo, u, rho_fp, x):                        # eq 12
    """
    Questa funzione calcola il fattore ausiliario G per il calcolo del fattore di forma YF.

    Args:
        modulo [mm]: modulo unificato.

        u [mm]: dedendum della ruota.

        rho_fp [mm]:Raggio di curvatura alla base del dente.

        x [\\]: fattore corretivo della specifica ruota
    
    Returns: 
        G [\\]: fattore ausiliario.
    """
    return ((rho_fp-u)/modulo + x)

# Fattore ausiliario H
def calcola_H(modulo, E, z):                                # eq 13
    """
    Questa funzione calcola il fattore ausiliario H per il calcolo del fattore di forma YF.

    Args:
        modulo [mm]: modulo unificato.

        E [\\]: fattore ausiliario.

        z [\\]: numero di denti della specifica ruota.
    
    Returns: 
        H [\\]: fattore ausiliario.
    """
    T = np.pi/3                                             # for spur gear
    return ((2/z)*((np.pi/2) - (E/modulo)) - T)

# Fattore ausiliario theta
def calcola_theta_aus(toll, guess, z, H, G):                # eq 14
    """
    Questa funzione calcola il fattore ausiliario theta per il calcolo del fattore di forma YF.

    Args:
        toll [\\]: tolleranza richiesta dell' approssimazione.

        guess [rad]: valore iniziale.

        z [\\]: numero di denti della specifica ruota.
     
        H [\\]: fattore ausiliario.

        G [\\]: fattore ausiliario.

    Returns:
        theta_aus [rad]: fattore ausiliario.

        iterazioni [\\]: numero di iterazioni effettuate per arrivare al risultato con la tolleranza richeista.
    """
    iterazioni = 0
    while True:
        # Valore della funzione in guess
        f_theta = ((2 * G) / z) * np.tan(guess) - H
    
        # Prossima stima di theta 
        theta_next = f_theta
    
        iterazioni += 1  # Incremento il contatore 
    
        # Errore relativo tra le due stime
        error = abs((theta_next - guess) / theta_next)
    
        if error <= toll:
            break

        guess = theta_next
        risultato = guess
    return risultato, iterazioni

# Corda normale/spessore alla base  
def calcola_sfn(modulo, z, theta_aus, G, rho_fp):           # eq 15
    """
    Questa funzione calcola  la corda normale alla base del dente.

    Args:
        modulo [mm]: modulo unificato.

        z [\\]: numero di denti della specifica ruota.

        guess [rad]: valore iniziale.

        theta_aus [rad]: fattore ausiliario.

        G [\\]: fattore ausiliario.

        rho_fp [mm]:Raggio di curvatura alla base del dente.

    Returns:
        sfn [mm]: corda normale alla base del dente.
        sfn_norm [\\]: corda NORMALIZZATA per il modulo.
    """
    fatt3 = z*np.sin(np.pi/3 - theta_aus)
    fatt4 = np.sqrt(3)*(G/np.cos(theta_aus) - rho_fp/modulo)
    sfn = modulo*(fatt3 + fatt4)
    sfn_norm = sfn/modulo
    return sfn, sfn_norm 

#  radius of root fillet, ρF
def calcola_rhoF(modulo, z, theta_aus, G, rho_fp):          # eq 17
    """
    Questa funzione calcola  la corda normale alla base del dente.

    Args:
        modulo [mm]: modulo unificato.

        z [\\]: numero di denti della specifica ruota.

        theta_aus [rad]: fattore ausiliario.
        
        G [\\]: fattore ausiliario.

        rho_fp [mm]:Raggio di curvatura alla base del dente.

    Returns:
        rhoF [mm]:  radius of root fillet, rhoF
    """
    fatt5 = 2*G**2
    fatt6 = np.cos(theta_aus)*(z*np.cos(theta_aus)**2 - 2*G)
    rho_F = rho_fp + modulo*fatt5/fatt6
    return rho_F

# form-factor pressure angle
def calcola_alphaen(dan, dbn, d_p_p, beta, theta_l, z, eps_gamma):
    """
    Questa funzione calcola il form-factor pressure angle.

    Args:
        dan [mm]: fattore geometrico ausiliario.

        dbn [mm]: fattore geometrico ausiliario.

        d_p_p [mm]: diametro primitivo della ruota.

        beta [rad]: angolo d'elica.

        theta_l [rad]: angolo di lavoro.

        z [\\]: numero di denti.
        
        eps_gamma [\\]: fattore di ricoprimento totale.

    Returns:
        alpha_en [rad]:  form-factor pressure angle.

        den [mm]: fattore geometrico ausiliario.
    """
    fatt7 = np.sqrt((dan/2)**2 - (dbn/2)**2)
    fatt8 = (np.pi*d_p_p*np.cos(beta)*np.cos(theta_l)/z)*(eps_gamma - 1)
    fatt9 = (dbn/2)**2
    den = 2*np.sqrt((fatt7 - fatt8)**2 + fatt9)             # eq 28
    alpha_en = np.arccos(dbn/den)                           # eq 29
    return alpha_en, den

# gamma_e
def calcola_gamma_e(x, z, theta_p, alpha_en):               # eq 30
    """
    Questa funzione calcola l' angolo gamma_e.

    Args: 
        x [\\]: fattore correttivo della ruota specifica.

        z [\\]: numero di denti della specifica ruota.

        theta_p [rad]: angolo di pressione.

        alpha_en [rad]: form-factor pressure angle
    
    Returns:
        gamma_e [rad]: angolo ausiliario.
    """
    fatt10 = (0.5*np.pi + 2*x*np.tan(theta_p))/z
    gamma_e = fatt10 + ev(theta_p) - ev(alpha_en)
    return gamma_e

# Bending moment arm
def calcola_hFE(z, m, den, rho_fp, theta_aus, gamma_e, alpha_fen, G):# eq 18
    """
    Questa funzione calcola il bending moment

    Args: 
        z [\\]: numero di denti.

        m [mm]: modulo unificato.

        den [mm]: fattore geometrico ausiliario.

        rho_fp [mm]: Raggio di curvatura alla base del dente.

        theta_aus [rad]: fattore ausiliario.

        gamma_e [rad]: angolo ausiliario.

        alpha_fen [rad]: angolo ausiliario.

        G [\\]: fattore ausiliario

    Returns:
        hFe [mm]: bending moment arm for tooth root stress.

        hFe_norm [\\]: bending moment arm, normalized.
    """
    fatt12 = (np.cos(gamma_e) - np.sin(gamma_e)*np.tan(alpha_fen))*(den/m)
    fatt13 = z*np.cos((np.pi/3) - theta_aus)
    fatt14 = (G/np.cos(theta_aus) - (rho_fp/m))
    hFe = (m/2)*(fatt12 - fatt13*fatt14)
    hFe_norm = hFe/m
    return hFe, hFe_norm

# Fattore di forma
def calcola_YF(hFE_norm, sfn_norm, theta_l, alpha_fen):     # eq 9
    """
    Questa funzione calcola il fattore di forma YF

    Args: 
        hFe_norm [\\]: bending moment arm, normalized.

        sfn_norm [\\]: corda NORMALIZZATA per il modulo.

        theta_l [rad]: angolo di lavoro.

        alpha_fen [rad]: angolo ausiliario.

    Returns: 
        YF [\\]: fattore di forma
    """
    fatt15 = 6*(hFE_norm)*np.cos(alpha_fen)
    fatt16 = np.cos(theta_l)*(sfn_norm)**2
    YF = fatt15/fatt16
    return YF

# Stress correction factor
def calcola_YS(sfn, hFe, rho_F):
    """
    Questa funzione calcola il fattore di correzione dello stress YS

    Args:
        sfn [mm]: corda normale alla base del dente.

        hFe [mm]: bending moment arm for tooth root stress.

        rhoF [mm]:  radius of root fillet, rhoF.

    Returns: 
        YS [\\]: fattore correttivo YS 

        L: fattore ausiliario

        qs: fattore ausiliario
    """
    qs = sfn/(2*rho_F)
    L = sfn/hFe
    exp = 1/(1.21 + 2.3/L)
    YS = (1.2 + 0.13*L)*qs**(exp)
    return YS, L, qs