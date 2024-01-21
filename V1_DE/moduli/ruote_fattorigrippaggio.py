import numpy as np

# Slip velovcity 
def vg(vt, g, beta, ds):
    """
    Questa funzione calcola la velocità di strisciamento dei denti.

    Args: 
        vt [m/s]: velocità lineare a cerchio primitivo.

        g [mm]: arco di reccesso (per vgt1) o di accesso (per vgt2).

        beta [rad]: angolo d'elica.

        ds [mm]: diametro primitivo.

    Returns:
        vgt [m/s]: velocità di strisciamento dei denti
    """
    fatt1 = 2*vt*g
    fatt2 = np.cos(beta)/ds
    return fatt1*fatt2

# geometric factor
def calcola_XG(x_betaalpha, u, gamma_yA, gamma_yE):
    """
    Questa funzione calcola il fattore geometrico.

    Args: 
        x_betaalpha [\\]: fattore d'angolo tabulato.

        u [\\]: gear ratio z2/z1.

        gamma_y [\\]: posizione del punto arbitrario sulla linea d'azione (T1=-1, C=0, T2 = u).

    Returns:
        XG [\\]: Fattore geometrico.

    """
    fatt1 = 0.51*x_betaalpha*np.sqrt(u+1)
    num = np.abs(1+ gamma_yA - 1 - (gamma_yE/u))
    den = ((1+gamma_yA)**(1/4)) * ((u-gamma_yE)**(1/4))
    XG =  fatt1 * num/den
    return XG

# coefficent of friction 
def calcola_mum(XL, XR, vsigmac, rho_red, wbt):
    """
    Questa funzione calcola il coefficiente d'attrito. 

    Args: 
        XL [\\]: coefficiente del sistema di lubrificazione.

        XR [\\]: coefficiente di rugosità.

        vsigmac [m/s]: cumulative velocity vector.

        rho_red [mm]: local relative radius of curvature.

        wbt [N/mm]: Transverse unit load.
    
        Returns: 

        mum [\\]: coefficiente d'attrito
    """
    fatt = XL*XR
    den = vsigmac * rho_red
    mum = (60*0.001)*((wbt/den)**(0.2))*fatt
    return mum

# calcola thetafl

def calcola_thetafl(mum, XM, XJ, XG, X_gamma, wbt, v_t, diametro_p):
    """
    Questa funzione calcola la temperatura di flash.

    Args: 
        mum [\\]: coefficiente d'attrito.

        XM [\\]:  Thermoelastic factor.

        XJ [\\]: Approacch factor.

        XG [\\]: Geometrical factor

        X_gamma [\\]: Load sharing factor.

        wbt [N/mm]: Transverse unit load.

        vt [m/s]: velocità lineare a cerchio primitivo.

       diametro_p [mm]: diametro primitivo della ruota considerata.

    Returns: 
        theta_fl [gradi]: temperatura flash.

    """
    fatt1 = mum*XM*XJ*XG
    fatt2 = (X_gamma*wbt)**(3/4)
    fatt3 = np.sqrt(v_t)/(diametro_p/2)**(1/4)
    theta_fl = fatt1*fatt2*fatt3
    return theta_fl

# banda di pressione hertziana
def calcola_bh(Ft, Er, b, theta_p, rho_rel):
    """
    Questa funzione calcola la banda di pressione hertziana.

    Args: 
        Ft [N]: forza circonferenziale.

        Er [N/mm2]: modulo di elasticità ridotto.

        b [mm]: lasrghezza di fascia.

        theta_p [rad]: angolo di pressione.

        rho_rel [mm]: raggio di curvatura.

    Returns: 
        bh [mm]: semiampiezza della banda di pressione herztiana

    """
    fatt1 = 2*np.sqrt(2)/np.sqrt(np.pi)
    fatt2 = Ft/(Er*b*np.cos(theta_p))
    bh = fatt1 * np.sqrt(fatt2*rho_rel)
    return bh

# Péclet numbers
def calcola_Pe(vg, bh, rho, c, lam, gamma):
    """
    Questa funzione calcola i numeri di Péclet per mostrare se la relazione 
    7.20 del Vullo è utilizzabile. 

    Args:
        vg [m/s]: velocità di strisciamento.

        bh [mm]: larghezza della banda Hertziana.

        rho [kg/m3]: densità.

        c [Nm/kgK]: calore specifico.

        lam [N/sK]: capacità termica

        gamma [rad]: angolo ausiliario
    Returns: 
        Pe [\\]: numero di Péclet
    """
    num = vg*bh*rho*c
    den = lam*np.sin(gamma)
    return num/den

#Verifica e grafica thetafl
def verifica_thetafl(mum, X_gamma, XJ, wbn, bh, vg1, vg2, BM1, BM2):
    """
    Questa funzione verifica e rende graficabile la temperatura di flash.

    Args: 
        mum [\\]: coefficiente d'attrito.

        X_gamma [\\]: Load sharing factor.

        XJ [\\]: Approacch factor.

        wbn [N/mm]: Normal unit load.

        bh [mm]: semiampiezza della banda hertziana.

        vg1, vg2 [m/s]: velocità di strisciamento.

        BM1, BM2 [?]: Coefficienti di contatto.

    Returns: 
        thetafl [gradi]: temperatura flash.

    """
    num1 = mum*X_gamma*XJ*wbn
    den1 = np.sqrt(2*bh)
    fatt1 = num1/den1
    num2 = np.abs(vg1-vg2)
    den2 = (BM1*np.sqrt(vg1)) + (BM2*np.sqrt(vg2))
    fatt2 = num2/den2
    thetafl = 1.11*fatt1*fatt2
    return thetafl

def calcola_L(u, beta, KV, epsilon_alpha, W_kW):
    """Questa funzione calcola il fattore di rumorosità secondo la formula di Kato.

    Args: 
        u [\\]: rapporto di trasmissione.

        beta [rad]: angolo d'elica.

        KV: fattore dnamico.

        epsilon_alpha [\\]: rapporto di condotta trasversale. 

        W_kW [kW]: potenza in ingresso al riduttore, espressa in kilowatt.

    Returns: 
        L [Db]: fattorei 
    """
    term1 = 20 * (1 - np.tan(beta / 2)) * u ** (1/8)
    term2 = 20 * np.log(W_kW * 1.34102)
    
    L = term1 / (KV * epsilon_alpha ** (1/4)) + term2

    return L
