import numpy as np
import matplotlib.pyplot as plt

# Questo script forrebbe ricreare il profilo di un dente con dentatura ad evolvente. 
# Per farlo ho preso a larghe mani spunto dall'articolo "A Mathematical Model for Parametric Tooth Profile of Spur Gears" 
# Che per brevità d'ora in poi chiamerò Articolo1. 
# Questo articolo definisce le funzioni matematiche di seguito indicate

# Funzione evolvente
def ev(angolo):
    return np.tan(angolo) - angolo

# Creazione del profilo ad evolvente 
def coordinate_x(alpha_t, rb, z, ry):
    """
    Funzione per il calcolo delle coordinate x del dente. 

    Args: 
        alpha_t: is the pressure angle
        rb: radius of the base circle (fondamentale)
        z: numero  di denti
        ry: radius of any circle (variabile)

    Returns: 
        x, y: coordinate per plottare il profilo del dente
    """
    alpha_t = np.radians(alpha_t)
    inv_at = ev(alpha_t)
    alpha_y = np.arccos(rb/ry)
    inv_ay = ev(alpha_y)
    frac = np.pi/(2*z)
    sin_value = np.sin(frac + inv_at - inv_ay)

    x = ry*sin_value
    return x

def coordinate_y(alpha_t, rb, z, ry):
    """
    Funzione per il calcolo delle coordinate y del dente. 

    Args: 
        alpha_t: is the pressure angle
        rb: radius of the base circle (fondamentale)
        z: numero  di denti
        ry: radius of any circle (variabile)

    Returns: 
        x, y: coordinate per plottare il profilo del dente
    """   
    alpha_t = np.radians(alpha_t)
    inv_at = ev(alpha_t)
    alpha_y = np.arccos(rb/ry)
    inv_ay = ev(alpha_y)
    frac = np.pi/(2*z)
    cos_value = np.cos(frac + inv_at - inv_ay)

    y = ry*cos_value
    return y

# Coefficiente k
def coefficiente_k(alpha_t, rb, z, ry):
    """
    Funzione per il calcolo del coefficiente -(1/k) utile per il calcolo dell'arco di sottotaglio/interferenza di taglio
    Sempre secondo l'Articolo1. 

    Args: 
        alpha_t: is the pressure angle
        rb: radius of the base circle (fondamentale)
        z: numero  di denti
        ry: radius of any circle (variabile)

    Returns: 
        -(1/k): Valore da utilizzare per il calcolo di theta
    """   
    alpha_t = np.radians(alpha_t)
    inv_at = ev(alpha_t)
    alpha_y = np.arccos(rb/ry)
    inv_ay = ev(alpha_y)
    frac = np.pi/(2*z)
    cos_value = np.cos(frac + inv_at - inv_ay)
    sin_value = np.sin(frac + inv_at - inv_ay)

    k_frac = -(sin_value/cos_value)
    return k_frac

# Distanza verticale
def coefficiente_H(alpha_t, m, z):
    """
    Funzione per il calcolo della distanza verticale H. 

    Args: 
        alpha_t: is the pressure angle
        m: modulo unificato
        z: numero  di denti

    Returns: 
        H: Distanza verticale
    """
    alpha_t = np.radians(alpha_t)
    fatt1 = 1.25*m
    fatt2 = (m*z)/2
    fatt3 = (1-np.cos(alpha_t))
    
    H = fatt1 - fatt2*fatt3

    return H


def parametric_circle(radius, num_points, x_c, y_c):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = x_c + radius * np.cos(theta)
    y = y_c + radius * np.sin(theta)
    return x, y

# Ora inizializzo i parametri della ruota dentata progettata.
# Angolo d'elica: nullo, angolo di pressione: 20grad, 
# Parametri della ruota dentata - SOPRA PRIMITIVA
r_b = 27.02             # Raggio di base (fondamentale)
R_min = r_b             # Raggio minimo (fondamenyale)
R_max = 31.25           # Raggio massimo (testa)
R_valori = np.arange(R_min, R_max, 0.001)

# Coordonate del profilo
x = np.array([coordinate_x(20, r_b, 23, rr) for rr in R_valori])
y = np.array([coordinate_y(20, r_b, 23, rr) for rr in R_valori])

# Archi che individuano la circonerenza primitiva, quella fondamentale e quella di piede
angolo = np.linspace(1.74533, 1.39626, 100)
# Primitiva
arc_x_prim = 28.75 * np.cos(angolo)
arc_y_prim = 28.75 * np.sin(angolo)
# Fondamentale
arc_x_fond = 27.02 * np.cos(angolo)
arc_y_fond = 27.02 * np.sin(angolo)
# Di piede
arc_x_pied = 25.62 * np.cos(angolo)
arc_y_pied = 25.62 * np.sin(angolo)
# Valori Che individuano la spoglia superiore, la testa del dente
x_tip_plot = np.sqrt(R_max**2 - R_valori**2)
y_tip_plot= np.sqrt(R_max**2 - x**2)


####    Plot sopra primitiva
# Profilo dx & sx sopra primitiva
plt.figure(1)
plt.plot(x, y, 'k') 
plt.plot(-x, y, 'k')
# Circonferenze caratteristiche
plt.plot(arc_x_prim, arc_y_prim, 'r')
plt.plot(arc_x_fond, arc_y_fond, 'b')
plt.plot(arc_x_pied, arc_y_pied, 'g')
# Testa del dente
plt.plot([-x[-1], x[-1]], [y_tip_plot[-1], y_tip_plot[-1]], 'k')
# Ammennicoli 
plt.title('Forma superiore del Dente del pignone')
plt.xlabel('Coordinata x')
plt.ylabel('Coordinata y')
plt.grid(True)
plt.axis('equal')
#plt.show()
print("CONTROLLO GRAFICO: SPESSORE IN TESTA", round(2*x[-1],2))
# Dalla verifica precedentemente fatta (in un altro script) lo spessore del dente viene identico,
# Questa parte secondo me è corretta

# Da qui in poi vengono i dolori

# Parametri della ruota dentata - SOTTO PRIMITIVA
R_min_new = 25.62       # Raggio minimo (di piede?)
R_max_new = r_b         # Raggio massimo (fondamentale)
R_valori_new = np.arange(R_min_new, R_max_new, 0.001)

# Calcolo i valori come spiegato nell'Articolo1
meno_uno_su_k = coefficiente_k(20, r_b, 23, r_b) 
#print("-1/k", meno_uno_su_k)
theta = np.arctan(meno_uno_su_k)
#print("theta", theta)
H = coefficiente_H(20, 2.5, 23)
#print("H", H)
# Ho printato i valori per vedere non fossero strani, sembrano corretti

# raggio  r  
r = H/(1-np.sin(theta))
#print("r",r)

X_A, Y_A = x[0], y[0]
#print(X_A, Y_A)
# Questo punto A, sempre dall'Articolo1 è l'ultimo punto dell'evolvente
# Per via grafica, sembra essere corretto

# Centro del raggio di raccordo
X_C = X_A + r * np.cos(theta)
Y_C = Y_A + r * np.sin(theta)
r_racc = r * np.cos(theta)
#print(r_racc)
# Questo punto C sembra essere corretto, perchè per via grafica appare similmente
# Al punto C dell'Articolo1 figura14

# Coordinate del punto di minima sezione alla base del dente
X_D = X_A + r_racc * np.sin(theta)  
Y_D = Y_A - r_racc * np.cos(theta)
# Questo punto l'ho aggiuto io, dovrebbe essere corretto.

# Ora, il punto B la figura14 dell'articolo1 lo interpreta come proiezione ortogonale del 
# Punto C sulla circonferenza di Piede? Perchè se è cosi io non riesco proprio a farla questa proiezione. 

# In un altro Articolo, "Orthogonal projection of points in CAD/CAM applications: an overview" (Articolo2)
# Dice che per trovare questo punto devo eseguire l'iterazione dell'equazione 15, come la imposto? Non ne ho
# La più pallida idea
# Punto B
X_B = X_C + r * np.sin(theta)
Y_B = Y_C - r * np.cos(theta)
# Per ora il punto B l'ho visto così, come una traslazione verticale...

# Ora il problema più grosso sta nel plot di questo maledetto raggio di raccordo, 
# Secondo l'articolo1 "it is easy to compute the graphic curve" ma a me non sembra proprio
# Anche perchè se provo a fare un fit polinomiale di questi tre punti, mi viene (giustamente, per python)
# Una parabola... 
# Dove posso agire? Cosa non sto considerando? Cosa sto sbagliando? 
x_base, y_base = parametric_circle(r, 100, X_C, Y_C)

# coordinate i punti A, D, e B
x_points = np.array([X_A, X_D, X_B])
y_points = np.array([Y_A, Y_D, Y_B])

# coefficienti del polinomio
coefficients = np.polyfit(x_points, y_points, 2)
poly = np.poly1d(coefficients)

# curva interpolata
x_curve = np.linspace(min(x_points), max(x_points), 100)
y_curve = poly(x_curve)

plt.figure(2)
plt.plot(x, y, 'k')  # Profilo sopra primitiva
plt.plot(-x, y, 'k')  # Profilo sopra primitiva (simmetrico)
plt.plot(arc_x_prim, arc_y_prim, 'r')  # Primitiva
plt.plot(arc_x_fond, arc_y_fond, 'b')  # Fondamentale
plt.plot(arc_x_pied, arc_y_pied, 'g')  # Piede
plt.plot([-x[-1], x[-1]], [y_tip_plot[-1], y_tip_plot[-1]], 'k')  # Testa del dente
#plt.plot(X_A, Y_A, 'yo', label = "A") 
plt.plot(X_C, Y_C, 'bo', label="C")  # Punto di inizio del raggio di raccordo
#plt.plot(X_D, Y_D, 'go', label="D")  # Raggio di raccordo al di sotto della fondamentale
#plt.plot(X_B, Y_B, 'mo', label="B")  # Punto di inizio del raggio di raccordo
plt.plot(x_base, y_base, label='Circonferenza')
plt.scatter(x_points, y_points, color='red', label='Punti dati')
plt.plot(x_curve, y_curve, label='Curva interpolata')
plt.legend()  # Aggiunto per mostrare le etichette delle legende
plt.title('Forma totale del Dente del pignone')
plt.xlabel('Coordinata x')
plt.ylabel('Coordinata y')
plt.grid(True)
plt.axis('equal')
plt.show()



