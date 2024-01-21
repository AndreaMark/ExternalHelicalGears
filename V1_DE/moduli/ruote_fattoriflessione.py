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

# Angolo dall'evolvente
def angolodaevolvente(evtheta):
        '''
        Questa funzione calcola l'angolo dal valore di evolvente(angolo). 

        Args:
            evtheta [\\]: evolvente di cui si vuole conoscere l'angolo. 

        Returns 
        '''
        # Risolvo numericamente a partire dalla funzione evolvente per individuare l'angolo di lavoro. 
        # Scelgo Newton - Raphson
        angolo = 1.0        # Inizializzo 
        toll= 1e-3                # Precisione che richiedo 
        iter = 0
        while True:
            evangolowhile = np.tan(angolo) - angolo                         # Funzione evolvente
            errore = evangolowhile - evtheta                  # Differenza tra il valore appena calcolato e il valore target evtheta_lavoro
            if abs(errore) < toll:
                break 
            # Derivata!
            derivata = -1 + 1/np.cos(angolo)**2 
            iter +=1
            # Aggiorna theta_lavoro usando il metodo di Newton-Raphson
            angolo = angolo - errore / derivata

        return angolo, iter

# Fattori ausiliari E-G-H
import numpy as np

def calcola_EGH(rhoFp, hfP, m, x, alpha, spr, z):
    """
    Questa funzione calcola i fattori E, G, e H per il calcolo del fattore di forma YF.

    Args:
        rhoFp (mm): Raggio di curvatura alla base del dente.
        hfP (mm): Dedendum del dente.
        m (mm): Modulo unificato.
        x [\\]: Fattore di correzione.
        alpha (rad): Angolo di pressione normale.
        spr (mm): Undercut dei denti.
        z [\\]: Numero di denti.
    
    Returns: 
        E (\\): Fattore ausiliario E.
        G (\\): Fattore ausiliario G.
        H (\\): Fattore ausiliario H.
    """
    # Calcola E
    pi = np.pi
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    tan_alpha = np.tan(alpha)

    E = (pi / 4) * m - hfP * tan_alpha + spr / cos_alpha - (1 - sin_alpha) * rhoFp / cos_alpha

    # Calcola G
    G = rhoFp / m - hfP / m + x

    # Calcola H
    T = pi / 3
    H = (2) * (pi / 2 - E / m)/z - T

    return E, G, H


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
def calcola_sFn(G, rhoFp, m, theta, z):           # eq 15
    """
    Questa funzione calcola  la corda normale alla base del dente.

    Args:
        G [\\]: fattore ausiliario.

        rhoFp [mm]: Raggio di curvatura alla base del dente.

        modulo [mm]: modulo unificato.

        theta_aus [rad]: fattore ausiliario.

        z [\\]: numero di denti della specifica ruota.

    Returns:
        sfn [mm]: corda normale alla base del dente.
        sfn_norm [\\]: corda NORMALIZZATA per il modulo.
    """
    pi = np.pi
    cos_theta = np.cos(theta)
    numeratore = z * np.sin(pi / 3 - theta) + np.sqrt(3) * (G / cos_theta - rhoFp / m)
    rapporto = numeratore / m

    return numeratore, rapporto

#  radius of root fillet, ρF
def calcola_rhoF(modulo, z, theta_aus, G, rho_fp):          # eq 17
    """
    Questa funzione calcola  raggio di raccordo alla base del dente.

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
def calcola_alphaen_den(d_an, d_bn, d, alpha, beta, z, epsilon_alpha):
    """
    Questa funzione calcola il form-factor pressure angle.

    Args:
        d_an [mm]: fattore geometrico ausiliario.

        d_bn [mm]: fattore geometrico ausiliario.

        d [mm]: diametro primitivo della ruota.

        alpha [rad]: angolo di pressione.

        beta [rad]: angolo d'elica.
        
        z [\\]: numero di denti.
        
        epsilon_alpha [\\]: fattore di ricoprimento totale.

    Returns:
        alpha_en [rad]:  form-factor pressure angle.

        d_en [mm]: fattore geometrico ausiliario.
    """   
    pi = np.pi
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)

    # Calcola d_en
    inner_sqrt = np.sqrt((np.sqrt((d_an / 2) ** 2 - (d_bn / 2) ** 2) - (pi * d * cos_beta * cos_alpha / z) * (epsilon_alpha - 1)) ** 2 + (d_bn / 2) ** 2)
    d_en = 2 * inner_sqrt

    # Calcola alpha_en
    alpha_en = np.arccos(d_bn / d_en)

    return alpha_en, d_en

# gamma_e ed alphaFen
def calcola_alphaFen(alpha_en, gamma_e):
    """
    Questa funzione calcola l' angolo gamma_e.

    Args: 
        x [\\]: fattore correttivo della ruota specifica.
        alpha [rad]: angolo di pressione.
        z [\\]: numero di denti della specifica ruota.
        alpha_en [rad]: form-factor pressure angle
    
    Returns:
        alpha_Fen [rad]: angolo di applicazione del carico.
    """
    # Calcola alpha_Fen
    alpha_Fen = alpha_en - gamma_e

    return alpha_Fen

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
    fatt10 = ((0.5*np.pi) + (2*x*np.tan(theta_p)))/z
    gamma_e = fatt10 + ev(theta_p) - ev(alpha_en)
    return gamma_e

# Bending moment arm
def calcola_hFe(gamma_e, alpha_Fen, d_en, m, theta, z, G, rhoFp):# eq 18
    """
    Questa funzione calcola il bending moment

    Args: 
        gamma_e [rad]: angolo ausiliario.

        alpha_Fen [rad]: angolo ausiliario.

        d_en [mm]: fattore geometrico ausiliario.

        m [mm]: modulo unificato.

        theta [rad]: fattore ausiliario.

        z [\\]: numero di denti.

        G [\\]: fattore ausiliario.

        rhoFp [mm]: Raggio di curvatura alla base del dente.

    Returns:
        hFe [mm]: bending moment arm for tooth root stress.

        hFe_norm [\\]: bending moment arm, normalized.
    """
    pi = np.pi
    cos_theta = np.cos(theta)
    sin_gamma_e = np.sin(gamma_e)
    cos_gamma_e = np.cos(gamma_e)
    tan_alpha_Fen = np.tan(alpha_Fen)

    hFe_su_m = 0.5 * ((cos_gamma_e - sin_gamma_e * tan_alpha_Fen) * (d_en / m) - z * np.cos(pi / 3 - theta) - (G / cos_theta - rhoFp / m))

    return hFe_su_m*m, hFe_su_m

# Fattore di forma
def calcola_YF(hFe_norm, sFn_norm, alpha_Fen, alpha):
    """
    Questa funzione calcola il fattore di forma YF

    Args: 
        hFe_norm [\\]: bending moment arm, normalized.

        sfn_norm [\\]: corda NORMALIZZATA per il modulo.

        alpha_fen [rad]: angolo ausiliario.

        alpha [rad]: angolo di lavoro.

    Returns: 
        YF [\\]: fattore di forma.
    """
    cos_alpha = np.cos(alpha)
    cos_alpha_Fen = np.cos(alpha_Fen)

    Y_F = (6 * (hFe_norm) * cos_alpha_Fen) / (((sFn_norm) ** 2 )* cos_alpha)
    return Y_F

# def calcola_YF(hFE_norm, sfn_norm, theta_l, alpha_fen):     # eq 9
    
#     fatt15 = 6*(hFE_norm)*np.cos(alpha_fen)
#     fatt16 = np.cos(theta_l)*(sfn_norm)**2
#     YF = fatt15/fatt16
#     return YF

# Stress correction factor
def calcola_YS(sfn, hFe, rho_F):
    """
    Questa funzione calcola il fattore di correzione dello stress YS.

    Args:
        sfn [mm]: corda normale alla base del dente.

        hFe [mm]: bending moment arm for tooth root stress.

        rhoF [mm]:  radius of root fillet, rhoF.

    Returns: 
        YS [\\]: fattore correttivo YS.

        L: fattore ausiliario.

        qs: fattore ausiliario.
    """
    qs = sfn/(2*rho_F)
    L = sfn/hFe
    exp = 1/(1.21 + 2.3/L)
    YS = (1.2 + 0.13*L)*qs**(exp)
    return YS, L, qs