import numpy as np
################################################            FUNZIONI - ISO 6336-1            ################################################  
# massa specifica
def massa_rel(da, df, db, rho, u):                          # [kg/mm] eq. 30 
    """
    Questa funzione calcola la massa specifica in base alla norma ISO 6336-1:2006. 

    Args: 
        da [mm]: diametro di testa. 
        df [mm]: diametro primitivo/di lavoro. 
        db [mm]: diametro di base. 
        rho [kg/mm3]: densità.
        u [\\]: rapporto z2/z1. 

    Returns: 
        m_red [kg/mm]: massa relativa. 
    """
    dm = (da + df)/2
    fatt1 = (np.pi/8)*(dm/db)**2
    num2 = (1/rho) + (1/(rho*u**2))
    fatt2 = (dm**2)/num2
    fatt = fatt1*fatt2
    return fatt

# theoretical Tooth stiffness
def calcola_cthp(z, z_prime, x, x_prime):                   # eq 82,81 
    """
    Questa funzione calcola il fattore TEORICO di rigidità del dente (Tooth stiffness).

    Args: 
        z [\\]: numero di denti del pignone.
        z_prime [\\]: numero di denti della ruota.
        x [\\]: fattore correttivo del pignone.
        x_prime [\\]: fattore correttivo della ruota.
    
    Returns:
        qp [\\]: valore minimo per la flessibilità di una coppia di denti.
        c_th [N/mm*um]: rigidità singola teorica.
    """
    qp = 0.04723 + 0.15551/z + 0.25791/z_prime- 0.00635*x - (0.11654*x)/z - 0.00193*x_prime - (0.24188*x_prime)/z_prime + 0.00529*(x)**2 +  0.00182*(x_prime)**2  
    c_th = 1/qp
    return qp, c_th

# Basic rack factor
def calcola_CB(u, m, angolo):                               # eq 86 
    """
    Questa funzione calcola il Basic rack factor.

    Args: 
        u [mm]: dedendum.
        m [mm]: modulo.
        angolo [grad]: angolo di lavoro (o di pressione nel caso di ruota non corretta).

    Returns:
        CB [\\]: Basic rack factor.
    """
    return (1+0.5*(1.25 - (u/m)))*(1-0.2*(20-np.degrees(angolo)))

# single stiffness
def calcola_c_prime(c_teo, cm, cr, cb, beta, carico_specifico):                 # eq 90 
    """
    Questa funzione calcola la rigidità singola del dente (single stiffness).
    
    Args: 
        c_teo [N/mm*um]: fattore TEORICO di rigidità del dente.
        cm [\\]: Correction factor.
        cr [\\]: Gear blank factor.
        cb [\\]: Basic rack factor.
        beta [rad]: angolo d'elica.
        carico_specifico [N/mm]: specific load.
    Returns: 
        c_prime [N/mm*um]: rigidità singola del dente.
    """                              
    return c_teo*cm*cb*cr*np.cos(beta) * (carico_specifico/100)**(0.25)

# Mesh stiffness
def calcola_c_gammaalpha(c_primo, fatt_ricoprimento):                           # eq 91 
    """
    Questa funzione calcola il fattore di Mesh stiffness (rigidità dell'insieme).

    Args: 
        c_primo [N/mm*um]: rigidità singola del dente.
        fatt_ricoprimento [\\]: fattore di ricoprimento associato alle ruota.

    Returns: 
        c_gammaalpha [N/mm*um]: fattore di rigidità dei denti.
    """
    return c_primo*(0.75*fatt_ricoprimento + 0.25)

# Resonance running speed
def calcola_vel_ris(DentiP,cgammaalpha, m_s):                                        # eq 6 
    """
    Questa funzione calcola il numero di giri teorico di risonanza DEL SOLO PIGNONE.

    Args: 
        z [\\]: numero di denti
        cgammaalpha [N/mm*um]: fattore di rigidità dei denti.
        m_s [kg/mm]: massa specifica.

    Returns: 
        N [1/s]: numero di giri di risonanza.
    """
    return (30000/(np.pi*DentiP)) * np.sqrt(cgammaalpha/m_s)   

# Fattori ausiliarii calcolo KV
def calcola_BP(c_primo, fbp, sigma_lim, carico_specifico):                      # eq 15 
    """
    Questa funzione calcola il fattore ausiliario BP per il calcolo di KV.

    Args:
        c_primo [N/mm*um]: rigidità singola del dente.
        fbp [mm]: transverse base pitch deviation. 
        carico_specifico [N/mm]: specific load.
        sigma_lim [N/mm2]: tensione ammissibile in base alle tabelle specifiche.

    Returns:
        BP [\\]: fattore ausiliario.
    """
    yp = (160/sigma_lim)*fbp                                                    # eq 75
    f_bp_eff =  fbp - yp                                                        # [mm] f_bp - yp transverse base pitch deviation - estimated running-in allowances
    return (c_primo*f_bp_eff)/carico_specifico

def calcola_BF(c_primo, fbp, sigma_lim, carico_specifico):                      # eq 16
    """
    Questa funzione calcola il fattore ausiliario BF per il calcolo di KV.

    Args:
        c_primo [N/mm*um]: rigidità singola del dente.
        fbp [mm]: transverse base pitch deviation.
        carico_specifico [N/mm]: specific load.
        sigma_lim [N/mm2]: tensione ammissibile in base alle tabelle specifiche.

    Returns:
        BF [\\]: fattore ausiliario.
    """
    yp = (160/sigma_lim)*fbp                                                    # eq 75
    f_bp_eff =  fbp - yp                                                        # [mm] f_bp - yp transverse base pitch deviation - estimated running-in allowances
    f_alphaeff = f_bp_eff 
    return (c_primo*f_alphaeff)/carico_specifico

def calcola_BK(c_primo, sigma_lim, carico_specifico):                           # eq 17
    """
    Questa funzione calcola il fattore ausiliario BK per il calcolo di KV.

    Args:
        c_primo [N/mm*um]: rigidità singola del dente.
        sigma_lim [N/mm]: tensione ammissibile in base alle tabelle specifiche.        
        carico_specifico [N/mm]: specific load.

    Returns:
        BK [\\]: fattore ausiliario.
    """
    Ca = (1/18)*((sigma_lim/97) - 18.45)**2 + 1.5                               # [mm] tip relief
    return np.abs(1 - (c_primo*Ca)/carico_specifico)

# Fattore dinamico
def calcola_KV(BP, BF, BK, N, eps_gamma):                                                  # eq 14,13 tab4
    """
    Questa funziona calcola il fattore dimanico KV in range di risonanza subrictico

    Args:    
        BP [\\]: fattore ausiliario BP.
        BF [\\]: fattore ausiliario BF.
        BF [\\]: fattore ausiliario BK.
        N [\\]: rapporto di risonanza.
        eps_gamma [\\]: rapporto di condotta totale (evolvente e d'elica)
    Returns: 
        K [\\]: fattore ausiliario per il calcolo di KV.
        KV [\\]: fattore dinamico.
    """
    if eps_gamma>2:
        cv1 = 0.32
        cv2 = 0.57/(eps_gamma-0.3)
        cv3 = 0.096/(eps_gamma-1.56)
    else: 
        cv1 = 0.32
        cv2 = 0.34
        cv3 = 0.23
    K = cv1*BP + cv2*BF + cv3*BK
    
    return K, (N*K)+1, cv1, cv2, cv3

# fattore di carico alla faccia, KHbeta
def calcola_KHbeta(sigma_lim, Fbetax, c_gammaalpha, Fm, b):                     # eq 41
    """
    Questa funzione calcola il fattore di carico alla faccia, KHbeta

    Args:
        sigma_lim [N/mm]: tensione ammissibile in base alle tabelle specifiche.
        Fbetax [mm]: diseallinamento massimo iniziale prima del rodaggio.
        c_gammaalpha [N/mm*um]: valore medio della rigidezza per unità di fascia.
        Fm [N/mm]: mean transverse tangential load.
        b [mm]: larghezza di fascia.

    Returns: 
        KHbeta [\\]: face load factor
    """
    x_beta = 1 - 320/sigma_lim
    F_betay = Fbetax * x_beta
    c_gammabeta = 0.85*c_gammaalpha
    KHbeta = 1 + (F_betay*c_gammabeta)/(2*(Fm/b))
    return KHbeta

# fattore di carico trasversale
def calcola_Kalpha(sigma_lim, fbp, eps, c_gammaalpha, FTH, b):                  # eq 71
    """
    Questa funzione calcola il fattore di carico trasversale Kalpha.
    Args:
        sigma_lim [N/mm]: tensione ammissibile in base alle tabelle specifiche.
        fbp [mm]: transverse base pitch deviation.
        eps [\\]: fattore di ricoprimento.
        c_gammaalpha [N/mm*um]: valore medio della rigidezza per unità di fascia.
        FTH [N/mm]: Tangential load in a transverse plane. 
        b [mm]: larghezza di fascia.

    Returns:
        Kalpha [\\]: fattore di carico trasversale KHalpha = KHbeta
    """
    yp = (160/sigma_lim)*fbp                                                    # eq 75
    y_a = yp                                                                    # eq.75
    f_alpha_k = fbp - y_a
    Kalpha = (eps/2)*(0.9 + 0.4 * ((c_gammaalpha*f_alpha_k)/(FTH/b)))
    return Kalpha 