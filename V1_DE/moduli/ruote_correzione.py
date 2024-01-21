import numpy as np
from . import ruote_fattorigeometrici as rfge


def correzione_dentature(t_re, delta_iniziale, theta_0, x_pignone, x_ruota, rho_pignone, rho_ruota, r_pignone, r_ruota, DentiP, DentiR, modulo, beta, theta_0n):
    """
    Il mio obiettivo è quello di minimizzare gli strisciamenti specifici ks facendo il possibile per mantenerli minori di 1.4 e la loro differenza minore di 0.2.
    Per fare questo mi affido ad un while loop a tre condizioni da soddisfare tutte e tre.

    Args:
        t_re (float): reale rapporto di trasmissione, dato dal rapporto tra il numero di denti del pignone e quello della ruota.
        delta_iniziale (array): distanza di primo tentativo tra il punto in cui avviene il contatto e C, va da (-accesso_reale:+recesso_reale). 
        theta_0 (float): angolo di pressione frontale IN RADIANTI.
        x_pignone (float): valore correttivo del pignone di primo tentativo, da scegliere con le carte di Neimann.
        x_ruota (float): valore correttivo della ruota di primo tentativo, da scegliere con le carte di Neimann.
        rho_pignone (float): raggio FONDAMENTALE del pignone.
        rho_ruota (float): raggio FONDAMENTALE della ruota.
        r_pignone (float): raggio PRIMITIVO del pignone.
        r_ruota (float): raggio PRIMITIVO della ruota.
        DentiP (int): numero di denti del pignone.
        DentiR (int): numero di denti della ruota.
        m (float): modulo unificato. 
        beta (rad): angolo d'elica.
        theta_0n: angolo di pressione di realizzazione (della dentiera utensile)

    Returns:
        x_pignone_finale (0): valore correttivo finale del pignone.
        x_ruota_finale (1): valore correttivo finale della ruota.
        theta_lavoro (2): angolo di lavoro, identifica la primitiva di lavoro.
        accesso_corretto (3): segmento di accesso corretto.
        recesso_corretto(4): segmento di recesso corretto.
        Rtp_corretto (5): raggio di testa del pignone corretto.
        Rtr_corretto (6): raggio di testa della ruota corretto.
        m_lavoro (7): modulo di lavoro
        Rpp_corretto (8): raggio primitiva pignone corretto
        Rpr_corretto (9): raggio primitiva ruota corretto
        R_piede_pc (10): raggio di piede del pignone corretto
        R_piede_rc (11): raggio di piede della ruota corretto
        y (12): distanza radiale tra la primitiva di taglio e di lavoro, pignone
        y_prime (13): distanza radiale tra la primitiva di taglio e di lavoro, ruota
    """
    # Variabili introduttive, parto dai valori iniziali, notoriamente sbilanciati
    ks_pignone_funz = rfge.ks_pignone(t_re, delta_iniziale, r_pignone, theta_0)  
    ks_ruota_funz = rfge.ks_ruota(t_re, delta_iniziale, r_ruota, theta_0)

    # Definisco le variabili da minimizzare, quelle che entrano come condizioni all'interno del ciclo while
    ks_max_pignone = max(np.abs(np.min(ks_pignone_funz)), np.abs(np.max(ks_pignone_funz)))
    ks_max_ruota = max(np.abs(np.min(ks_ruota_funz)), np.abs(np.max(ks_ruota_funz)))
    delta_ks = np.abs(ks_max_pignone - ks_max_ruota)
    iter = 1
    while (
         ks_max_pignone > 1.4
        or ks_max_ruota > 1.4
        or delta_ks > 0.2 
        ):
        ######  RISOLUZIONE PER L'ANGOLO DI LAVORO
        # Posso calcolarmi dalla teoria:
        evtheta_lavoro = rfge.ev(theta_0) + 2 * np.tan(theta_0n) * (x_pignone + x_ruota) / (DentiP + DentiR)

        # Risolvo numericamente a partire dalla funzione evolvente per individuare l'angolo di lavoro. 
        # Scelgo Newton - Raphson
        theta_lavoro = 1.0        # Inizializzo 
        toll= 1e-3                # Precisione che richiedo 
        while True:
            evtheta_lavoro_while = np.tan(theta_lavoro) - theta_lavoro      # Funzione evolvente
            errore = evtheta_lavoro_while - evtheta_lavoro                  # Differenza tra il valore appena calcolato e il valore target evtheta_lavoro
            if abs(errore) < toll:
                break 
            # Derivata!
            derivata = -1 + 1/np.cos(theta_lavoro)**2 
        
            # Aggiorno theta_lavoro usando il metodo di Newton-Raphson
            theta_lavoro = theta_lavoro - errore / derivata                 # Angolo di lavoro frontale

        # Angolo di lavoro normale
        theta_lavoro_n = np.arctan(np.tan(theta_lavoro*np.cos(beta)))

        ######  CARATTERISTICHE GEOMETRICHE RUOTE CORRETTE, slide parte 18
        # Raggi delle primitive di lavoro
        Rp_lavoro = rho_pignone / np.cos(theta_lavoro)
        Rr_lavoro = rho_ruota / np.cos(theta_lavoro)
        
        # Modulo di lavoro
        m_lavoro_n =  modulo * np.cos(theta_0) / np.cos(theta_lavoro_n)         # Modulo di lavoro normale
        m_lavoro_f = m_lavoro_n/np.cos(beta)                                    # modulo di lavoro frontale

        # Coefficienti di correzione FRONTALI
        x_pignone_f = x_pignone*np.cos(beta)
        x_ruota_f = x_ruota*np.cos(beta)
  
        # Coefficienti eta per addendum e dedendum
        eta_p = (np.cos(theta_0)/np.cos(theta_lavoro) - 1) * DentiP/np.cos(beta)
        eta_r = (np.cos(theta_0)/np.cos(theta_lavoro) - 1) * DentiR/np.cos(beta)

        # Addendum e dedendum di lavoro - EFFETTIVI
        a_p = (1 + x_pignone_f - eta_p)*modulo
        a_r = (1 + x_ruota_f - eta_r)*modulo
        u_p = (1.25-x_pignone_f + eta_p)*modulo
        u_r = (1.25-x_ruota_f + eta_r)*modulo
        
        # Grandezze corrette tutte riferite al piano frontale
        # Raggi di testa CORRETTI 
        Rtp_lavoro = Rp_lavoro + a_p
        Rtr_lavoro = Rr_lavoro + a_r

        # Raggi di piede CORRETTI
        Rp_piede_lavoro = Rp_lavoro - u_p
        Rt_piede_lavoro = Rr_lavoro - u_r

        # Distanza radiale tre le primitive di taglio e dicontatto
        y = eta_p*modulo
        y_prime = eta_r*modulo

        # Arco dei contatti CORRETTO   
        phi_while = np.pi/2 - theta_lavoro
        accesso_reale_while = np.sqrt(np.abs(Rtr_lavoro**2 - rho_ruota**2)) - r_ruota*np.cos(phi_while)
        recesso_reale_while = np.sqrt(np.abs(Rtp_lavoro**2 - rho_pignone**2)) - r_pignone*np.cos(phi_while)

        delta_while = np.arange(-np.abs(accesso_reale_while), recesso_reale_while, 0.01)
        
        # Calcola i valori di ks_pignone e ks_ruota CORRETTI per questa iterazione
        ks_pignone_plot_while = rfge.ks_pignone(t_re, delta_while , Rp_lavoro, theta_lavoro)
        ks_ruota_plot_while = rfge.ks_ruota(t_re, delta_while , Rr_lavoro, theta_lavoro)

        # Aggiorna i valori di ks per la prossima iterazione
        ks_max_pignone = max(np.abs(np.min(ks_pignone_plot_while)), np.abs(np.max(ks_pignone_plot_while)))
        ks_max_ruota = max(np.abs(np.min(ks_ruota_plot_while)), np.abs(np.max(ks_ruota_plot_while)))
        delta_ks = np.abs(ks_max_pignone - ks_max_ruota)

        # Aggiorna i valori di x per la prossima iterazione
        if x_pignone > x_ruota:
            x_pignone -= 0.1
            x_ruota +=  0.1
        else:
            x_pignone += 0.1
            x_ruota -=  0.1

        # Aggiorna i valori di x_pignone e x_ruota prima di ritornarli
        x_pignone_finale = x_pignone_f
        x_ruota_finale = x_ruota_f

        # Aggiorna i valori da restituire
        accesso_corretto = np.abs(accesso_reale_while)
        recesso_corretto = recesso_reale_while
        Rtp_corretto = Rtp_lavoro
        Rtr_corretto = Rtr_lavoro
        Rpp_corretto = Rp_lavoro
        Rpr_corretto = Rr_lavoro
        R_piede_pc = Rp_piede_lavoro
        R_piede_rc = Rt_piede_lavoro
        iter += 1
    # Restituisci x e x_prime alla fine
    return x_pignone_finale, x_ruota_finale, theta_lavoro, accesso_corretto, recesso_corretto, Rtp_corretto, Rtr_corretto, m_lavoro_f, Rpp_corretto, Rpr_corretto, R_piede_pc, R_piede_rc, y, y_prime, iter