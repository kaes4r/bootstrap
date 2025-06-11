import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

def is_castable_to_int(value):
    if isinstance(value, float):
        return value.is_integer()
    return False

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def compute_if_matrix_is_ok(x_pows_dict, k):
    M = np.zeros((k+1,k+1))
    for i in range(k+1):
        for j in range(k+1):
            M[i, j] = x_pows_dict[i+j]
    
    return is_pos_def(M)
        
def recursion_osc_anarmonico(xsq_init:float, E_init:float, k:int, g:float):
    # H = P^2 + X^2 + gX^4
    x_pows_dict = {
        -5:0,
        -4:0,
        -3:0,
        -2:0,
        -1:0,
        0:1,
        1:0,
        2:xsq_init,
        3:0
    }
    max_power = 2*k
    for i in range (4, max_power+1):
        x_pows_dict[i] = (E_init/g)*((i-3)/(i-1))*x_pows_dict[i-4]+ ((i-3)*(i-4)*(i-5)/(4*g*(i-1)))*x_pows_dict[i-6] - (i-2)/(g*(i-1))*x_pows_dict[i-2]
        
    return x_pows_dict

def recursion_osc_armonico(xsq_init:float, E_init:float, k:int, g:float):
    # H = P^2 + X^2
    x_pows_dict = {
        -5:0,
        -4:0,
        -3:0,
        -2:0,
        -1:0,
        0:1,
        1:0,
        2:xsq_init,
        3:0
    }
    max_power = 2*k
    for i in range (4, max_power+1):
        x_pows_dict[i] = E_init*((i-1)/(i))*x_pows_dict[i-2]+ (((i-1)*(i-2)*(i-3))/(4*(i)))*x_pows_dict[i-4]
        
    return x_pows_dict

def recursion_osc_armonico_susy(xsq_init:float, E_init:float, k:int, g:float):
    # H = (P^2 + X^2)(id) - (id)(sigma^3)
    x_pows_dict = {
        -5:0,
        -4:0,
        -3:0,
        -2:0,
        -1:0,
        0:1,
        1:0,
        2:xsq_init,
        3:0
    }
    max_power = 2*k
    for i in range (4, max_power+1):
        x_pows_dict[i] = -(((-1 + i) * ((-2520 + 2754 * i - 1175 * i**2 + 245 * i**3 - 25 * i**4 + i**5) * x_pows_dict[-8 + i] + 
                         8 * (-60 + 47 * i - 12 * i**2 + i**3) * E_init * x_pows_dict[-6 + i] - 
                         8 * (-3 + i) * (12 - 6 * i + i**2 - 2 * E_init**2) * x_pows_dict[-4 + i] - 
                         32 * (-2 + i) * E_init * x_pows_dict[-2 + i])) / (16 * (-2 + i) * i))
    return x_pows_dict

def recursion_coulomb_ordine4(xinitial:float, E_init:float, k:int, g:float):
    # H = P^2 + 1/X espanso attorno a X=1
    x_pows_dict = {
        -5:0,
        -4:0,
        -3:0,
        -2:0,
        -1:0,
        0:1,
        1:xinitial,
    }
    max_power = 2*k
    for i in range (2, max_power+1):
        x_pows_dict[i] = E_init*((i-3)/2)*x_pows_dict[i-4]- ((i+7)/2)*x_pows_dict[i-2]+ (((i-3)*(i-4)*(i-5))/8)*x_pows_dict[i-6] + (5/2)*x_pows_dict[i-4] + (15/4)*x_pows_dict[i-1]
        
    return x_pows_dict


def bootstrap(E_min:float, E_max:float, xsq_min:float, xsq_max:float, E_epsilon:float, xsq_epsilon:float, g:float, k:int):
    x_values = np.arange(start=E_min, stop=E_max + E_epsilon, step=E_epsilon)
    y_values = np.arange(start=xsq_min, stop=xsq_max + xsq_epsilon, step=xsq_epsilon)
    
    map = np.zeros((len(y_values), len(x_values)), dtype='bool')
    
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            powers = recursion_osc_armonico_susy(xsq_init=y, E_init=x, k=k, g=g)
            #powers = recursion_coulomb_ordine4(xinitial=y, E_init=x, k=k, g=g)
            map[i, j] = compute_if_matrix_is_ok(x_pows_dict=powers, k=k)
            
    return map
    
    

#Initialize values
E_min=0
E_max=1
xsq_min=0
xsq_max=1
E_epsilon=5e-1
xsq_epsilon=5e-1
g=1

#PLOTTO LA MAPPA
plt.figure(figsize=(8, 8))
map_7 = bootstrap(E_min=E_min,E_max=E_max, xsq_min=xsq_min, xsq_max=xsq_max, E_epsilon=E_epsilon, xsq_epsilon=xsq_epsilon, g=g, k=5)
cmap_7 = LinearSegmentedColormap.from_list("custom", ["white", "orange"])
#map_8 = bootstrap(E_min=E_min,E_max=E_max, xsq_min=xsq_min, xsq_max=xsq_max, E_epsilon=E_epsilon, xsq_epsilon=xsq_epsilon, g=g, k=13)
#cmap_8 = LinearSegmentedColormap.from_list("custom", ["white", "blue"])
plt.imshow(map_7, cmap=cmap_7, extent=(E_min, E_max, xsq_min, xsq_max), origin='lower', alpha=0.7)
#plt.imshow(map_8, cmap=cmap_8, extent=(E_min, E_max, xsq_min, xsq_max), origin='lower', alpha=0.7)

legend_elements = [
    Patch(facecolor='orange', edgecolor='orange', label='K=11'),
    Patch(facecolor='blue', edgecolor='blue', label='K=13')
]

plt.legend(handles=legend_elements, loc='lower right')
plt.xlabel("E")
plt.ylabel(r"$< X >$")
plt.title("Zone permesse")
plt.show()
