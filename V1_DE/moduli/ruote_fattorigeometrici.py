import numpy as np


################################################            FUNZIONI            ################################################  
# Numero minimo di denti ideale   
def z_min_ideale(k_prime, t, theta_0):
    """
    Questa funzione calcola il numero di denti di primo tentativo, si basa su mere considerazioni geometriche di non interferenza.

    Args: 
        k_prime: fattore geometrico di intaglio, per il dimensionamento normale k_prime_pignone = 1, k_prime_ruota = 1.2
        t: rapporto di trasmissione ideale, richiesto.
        theta_0: RADIANTI, angolo di pressione, solitamente 20 gradi.
    
    Returns: 
        z_min: numero minmo di denti della specifica ruota.
    """
    num = 2 * k_prime * (1 + np.sqrt(1 + t * (1 + t) * np.sin(theta_0)**2))
    den = (2 + t) * np.sin(theta_0)**2
    z_min = num / den
    return z_min

# Fattore di ricoprimento ideale
def epsilon_ideale(z_prime, k_prime, z, k, theta_0):
    """
    Questa funzione calcola il fattore di ricoprimento relativo al numero di denti ideale calcolato con la funzione 'calcola_z_min'.

    Args: 
        z_prime: numero di denti della ruota.
        k_prime: fattore moltiplicitivo della ruota (1.25).
        z: numero di denti del pignone.
        k: fattore moltiplicativo del pignone (1).
        theta_0: angolo di pressione.

    Returns: 
        epsilon: fattore di ricoprimento teorico.

    """
    cos_theta = np.cos(theta_0)
    term1 = np.sqrt(((z_prime + 2 * k_prime) / cos_theta)**2 - z_prime**2)
    term2 = np.sqrt(((z + 2 * k) / cos_theta)**2 - z**2)
    term3 = (z + z_prime) * np.tan(theta_0)
    
    epsilon = (1 / (2 * np.pi)) * (term1 + term2 - term3)
    return epsilon

# Numero minimo di denti intagliabili
def z_min_intagliabili(k_prime, theta_0, beta_0):
    """
    Questa funzione calcola il numero di denti effettivamente intagliabili mendiante dentiera MAAG (utensile più utilizzato). 
    È come 'calcola_z_min' ma con rapporto di trasmissione ora nullo.

    Args: 
        k_prime: fattore moltiplicitivo (1 per pignone, 1.25 per ruota).
        theta_0: angolo di pressione normale.
        beta_0: angolo d'elica fondamentale

    Returns: 
        z: numero di denti effettivamente intagliebili a quell'angolo di pressione.

    """
    num = 2*k_prime
    den = np.sin(theta_0)**2
    fatt = np.cos(beta_0)**3
    z = (num/den)*fatt
    return z

# Numero minimo di denti reale
def z_min_reale(DentiP, DentiR_guess, t, toll):
    """
    Il reale numero di denti, da usare nei successivi calcoli, si ottiene per convergenza del valore del rapporto di trasmissione tau, da dover sempre garantire. 
    Dopo aver calcolato 'z_min_reale' e 'z_min_pratico' è necessario controllare che sia rispettato il rapporto di trasmissione richiesto, se questo non accade, si usa questa funzione.

    Args: 
        DentiP: numero di denti del pignone, decisi a partire da 'z_min_pratico' (solitamente numeri interi primi).
        DentiR_guess: numero di denti della ruota di primo tentativo, supposto.
        t: valore ideale del rapporto di trasmissione da dover rispettare.
        toll: tolleranzia richiesta dal calcolo, non più di e-5.

    Returns: 
    DentiR_guess: numero di denti della ruota relativi al numero di denti del pignone inseriti in input atti a mantenere quel determinato tau.

    """
    while True:
        tau_while = DentiP / DentiR_guess
        errore = tau_while - t              # Differenza tra tau_while e il valore target tau
        
        if abs(errore) < toll:
            break 

        derivata = -DentiP / (DentiR_guess ** 2)
        
        # Aggiorno DentiR_guess usando il metodo di Newton-Raphson
        DentiR_guess = DentiR_guess - errore / derivata
    
    return DentiR_guess

# Modulo unificato, primo tentativo
def calcola_modulo(k, Mt, lambda_val, sigma_amm):
    """
    Questa funzione calcola il modulo secondo Lewis.
    Il modulo definitivo si sceglie come il massimo tra i moduli di ruota e pignone e viene individuato nella tabella dei moduli unificati più comuni.

    Args: 
        k: fattore moltiplicativo da individuare nelle tabelle per denti, tipologia di ruota e angolo di pressione. 
        Mt: momento torcente relativo alla specifica ruota
        lamda_val: larghezza di fascia
        sigma_amm: tansione ammissibile del materiale 

    Returns:
        m_lewis: modulo minimo da rispettare per la resistenza flessionale del dente. 
    """
    m_lewis = k * (Mt / (lambda_val * sigma_amm))**(1/3)
    return m_lewis

# Primitive di funzionamento
def raggi_primitive(modulo, z):
    """
    Questa funzione calcola i raggi delle primitive dal modulo unificato e dal numero di denti.

    Args: 
        modulo: modulo unificato.
        z: numero di denti, diverso per pignone e ruota.

    Returns:
        r_primitiva: raggio della primitiva.       
    """
    r_primitiva = modulo * (z/2)
    return r_primitiva

# Funzione evolvente
def ev(angolo):
    """
    Questa funzione calcola l'evolvente di un angolo. 

    Args:
        angolo: valore angolare espresso in radianti.

    Returns:
        ev: valore della funzione evolvente.
    """
    return np.tan(angolo) - angolo

# Funzione evolvente ad un determinato spessore
def ev_gamma(spessore, raggio, angolo):
    """
    Questa funzione calcola l'evolvente ad un determinato spessore. 
    Solitamente si usa per calcolare l'evolvente sulla primitiva, ma può essere usata in qualinque punto del dente, basta conoscerne lo spessore ed il raggio. 
    Solitamente: spessore = spessore primitiva, raggio = raggio primitiva, angolo = angolo di pressione.

    Args: 
        spessore: spessore noto del dente.
        raggio: raggio al quale si trova quello spessore noto.
        angolo: angolo a cui si trova quello spessore.

    Returns: 
        Incognita: valore dell'evolvente incognita.
    """
    incognita = spessore/(2*raggio) + ev(angolo)
    return incognita

# Strisciamenti 
def ks_pignone(t, delta_re, R_pignone, theta_0):
    """
    Questa funzione calcola gli strisciamenti specifici dei denti del pignone lungo tutto il segmento dei contatti. 

    Args: 
        t_re: rapporto di trasmissione reale.
        delta_re: distanza reale tra il punto in cui avviene il contatto e C, va da (-accesso_reale:+recesso_reale).
        R_pignone: raggio di primitiva del pignone
        theta_0: angolo di pressione

    Returns: 
        ks: funzione strisciamento specifico, per tale funzione va ricercato il massimo valore in modulo alle estremità, questo deve essere minore di 1.4
    """
    ks = (1 + t)*(delta_re/(R_pignone*np.sin(theta_0)+delta_re))
    return ks

def ks_ruota(t, delta_re, R_ruota, theta_0):
    """
    Questa funzione calcola gli strisciamenti specifici dei denti della ruota lungo tutto il segmento dei contatti. 

    Args: 
        t_re: rapporto di trasmissione reale.
        delta_re: distanza reale tra il punto in cui avviene il contatto e C, va da (-accesso_reale:+recesso_reale).
        R_ruota: raggio di primitiva della ruota
        theta_0: angolo di pressione

    Returns: 
        ksp: funzione strisciamento specifico, per tale funzione va ricercato il massimo valore in modulo alle estremità, questo deve essere minore di 1.4
    """
    ksp = -(1 + (1/t))*(delta_re/(R_ruota*np.sin(theta_0)-delta_re))
    return ksp