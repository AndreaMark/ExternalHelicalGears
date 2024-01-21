import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

# Definisci le coordinate dei tre punti
x1, y1 = 1, 2
x2, y2 = 3, 4
x3, y3 = 5, 6

# Definisci le variabili simboliche
x, y, r = symbols('x y r')

# Equazioni del sistema
eq1 = Eq((x - x1)**2 + (y - y1)**2, r**2)
eq2 = Eq((x - x2)**2 + (y - y2)**2, r**2)
eq3 = Eq((x - x3)**2 + (y - y3)**2, r**2)

# Risolvi il sistema di equazioni
sol = solve((eq1, eq2, eq3), (x, y, r), dict=True)

# Ottieni i valori delle variabili
x_center, y_center, radius = sol[0][x], sol[0][y], sol[0][r]

print(f"L'equazione della circonferenza è: (x - {x_center})^2 + (y - {y_center})^2 = {radius}^2")

# Involute function
def evolute(angle):
    return np.tan(angle) - angle

# Create the involute profile 
def coordinate_x(pressure_angle, base_radius, num_teeth, variable_radius):
    """
    Function to calculate tooth coordinates.

    Args: 
        pressure_angle: pressure angle
        base_radius: radius of the base circle (fundamental)
        num_teeth: number of teeth
        variable_radius: radius of any circle (variable)

    Returns: 
        x: x-coordinate for tooth profile
    """
    pressure_angle = np.radians(pressure_angle)
    inv_pressure_angle = evolute(pressure_angle)
    angle_y = np.arccos(base_radius/variable_radius)
    inv_angle_y = evolute(angle_y)
    frac = np.pi/(2*num_teeth)
    sin_value = np.sin(frac + inv_pressure_angle - inv_angle_y)

    x = variable_radius * sin_value
    return x

def coordinate_y(pressure_angle, base_radius, num_teeth, variable_radius):
    """
    Function to calculate tooth coordinates.

    Args: 
        pressure_angle: pressure angle
        base_radius: radius of the base circle (fundamental)
        num_teeth: number of teeth
        variable_radius: radius of any circle (variable)

    Returns: 
        y: y-coordinate for tooth profile
    """   
    pressure_angle = np.radians(pressure_angle)
    inv_pressure_angle = evolute(pressure_angle)
    angle_y = np.arccos(base_radius/variable_radius)
    inv_angle_y = evolute(angle_y)
    frac = np.pi/(2*num_teeth)
    cos_value = np.cos(frac + inv_pressure_angle - inv_angle_y)

    y = variable_radius * cos_value
    return y

# Coefficient k
def coefficient_k(pressure_angle, base_radius, num_teeth, variable_radius):
    """
    Function to calculate the coefficient (-1/k) for the undercut/interference angle calculation.

    Args: 
        pressure_angle: pressure angle
        base_radius: radius of the base circle (fundamental)
        num_teeth: number of teeth
        variable_radius: radius of any circle (variable)

    Returns: 
        -(1/k): Value to use in the theta calculation
    """   
    pressure_angle = np.radians(pressure_angle)
    inv_pressure_angle = evolute(pressure_angle)
    angle_y = np.arccos(base_radius/variable_radius)
    inv_angle_y = evolute(angle_y)
    frac = np.pi/(2*num_teeth)
    cos_value = np.cos(frac + inv_pressure_angle - inv_angle_y)
    sin_value = np.sin(frac + inv_pressure_angle - inv_angle_y)

    k_frac = -(sin_value/cos_value)
    return k_frac

# Vertical distance
def vertical_distance(pressure_angle, module, num_teeth):
    """
    Function to calculate the vertical distance H.

    Args: 
        pressure_angle: pressure angle
        module: module
        num_teeth: number of teeth

    Returns: 
        H: Vertical distance
    """
    pressure_angle = np.radians(pressure_angle)
    term1 = 1.25 * module
    term2 = (module * num_teeth) / 2
    term3 = (1 - np.cos(pressure_angle))
    
    H = term1 - term2 * term3

    return H


def parametric_circle(radius, num_points, x_c, y_c):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = x_c + radius * np.cos(theta)
    y = y_c + radius * np.sin(theta)
    return x, y


# Gear parameters - ABOVE PITCH CIRCLE
base_radius = 27.02  # Base radius (fundamental)
R_min = base_radius  # Minimum radius (fundamental)
R_max = 31.25  # Maximum radius (head)
R_values = np.arange(R_min, R_max, 0.001)
# Profile values
x = np.array([coordinate_x(20, base_radius, 23, rr) for rr in R_values])
y = np.array([coordinate_y(20, base_radius, 23, rr) for rr in R_values])
# Primitive values
angle = np.linspace(1.74533, 1.39626, 100)
arc_x_prim = 28.75 * np.cos(angle)
arc_y_prim = 28.75 * np.sin(angle)
arc_x_fund = 27.02 * np.cos(angle)
arc_y_fund = 27.02 * np.sin(angle)
arc_x_foot = 25.62 * np.cos(angle)
arc_y_foot = 25.62 * np.sin(angle)
# Tip values
x_tip_plot = np.sqrt(R_max**2 - R_values**2)
y_tip_plot = np.sqrt(R_max**2 - x**2)

# Plot above pitch circle
plt.figure(1)
plt.plot(x, y, 'k')
plt.plot(-x, y, 'k')
plt.plot(arc_x_prim, arc_y_prim, 'r')
plt.plot(arc_x_fund, arc_y_fund, 'b')
plt.plot(arc_x_foot, arc_y_foot, 'g')
plt.plot([-x[-1], x[-1]], [y_tip_plot[-1], y_tip_plot[-1]], 'k')
plt.title('Top shape of Gear Tooth')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.axis('equal')
#print("GRAPHICAL CHECK: HEAD THICKNESS", round(2*x[-1],2))

# Gear parameters - BELOW PITCH CIRCLE
R_min_new = 25.62  # Minimum radius (root?)
R_max_new = base_radius  # Maximum radius (fundamental)
R_values_new = np.arange(R_min_new, R_max_new, 0.001)

minus_one_over_k = coefficient_k(20, base_radius, 23, base_radius) 
print("-1/k", minus_one_over_k)
theta = np.arctan(minus_one_over_k)
print("theta", theta)
H = vertical_distance(20, 2.5, 23)
print("H", H)

# Auxiliary radius r
r = H / (1 - np.sin(theta))
print("r", r)

X_A, Y_A = x[0], y[0]
print(X_A, Y_A)

# Rounding radius calculations
X_C = X_A + r * np.cos(theta)
Y_C = Y_A + r * np.sin(theta)
r_round = r * np.cos(theta)
print(r_round)

# Coordinates of the point of minimum section at the base of the tooth
X_D = X_A + r_round * np.sin(theta)
Y_D = Y_A - r_round * np.cos(theta)

# Point B
X_B = X_C + r * np.sin(theta)
Y_B = Y_C - r * np.cos(theta)

x_base, y_base = parametric_circle(r, 100, X_C, Y_C)

plt.figure(2)
plt.plot(x, y, 'k')  
plt.plot(-x, y, 'k')  
plt.plot(arc_x_prim, arc_y_prim, 'r')  
plt.plot(arc_x_fund, arc_y_fund, 'b')  
plt.plot(arc_x_foot, arc_y_foot, 'g')  
plt.plot([-x[-1], x[-1]], [y_tip_plot[-1], y_tip_plot[-1]], 'k')  
plt.plot(X_A, Y_A, 'yo', label="A") 
plt.plot(X_C, Y_C, 'bo', label="C")  
plt.plot(X_D, Y_D, 'go', label="D")  
plt.plot(X_B, Y_B, 'mo', label="B")  
plt.plot(x_base, y_base, label='Circle')
plt.legend()  
plt.title('Total shape of Gear Tooth')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.axis('equal')
plt.show()
