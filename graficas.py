import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import os

def plot_convergencia(funcion_str: str, res: dict, guardar_pdf: bool = True):
    """
    Dibuja la función, el eje X y el historial de iteraciones (versión limpia sin superposiciones).
    """
    x = sp.symbols('x')
    expr_f = sp.sympify(funcion_str)
    f_num = sp.lambdify(x, expr_f, 'numpy')
    
    puntos = res['historial_puntos']
    raiz_final = res['raiz']
    nombre_metodo = res['nombre']
    
    rango = max(puntos) - min(puntos)
    margen = max(rango * 1.0, 1.5)
    
    if margen == 0: margen = 1.0 
    
    x_vals = np.linspace(min(puntos) - margen, max(puntos) + margen, 400)
    y_vals = f_num(x_vals)
    
    # --- PALETA DE COLORES "AESTHETIC" Y FORMAL ---
    COLOR_CURVA = '#2C3E50'   # Azul marino oscuro
    COLOR_PUNTOS = '#7F8C8D'  # Gris azulado
    COLOR_RAIZ = '#E74C3C'    # Rojo carmesí
    COLOR_EJE = '#BDC3C7'     # Gris claro
    
    plt.figure(figsize=(8, 5))
    
    # 1. Curva y Ejes
    plt.plot(x_vals, y_vals, label=f'$f(x) = {funcion_str}$', color=COLOR_CURVA, linewidth=2)
    plt.axhline(0, color=COLOR_EJE, linewidth=1.5, linestyle='--')
    
    y_puntos = [f_num(p) for p in puntos]
    
    # 2. Puntos previos (Sin etiquetas, con un poco de transparencia para ver superposiciones)
    plt.scatter(puntos[:-1], y_puntos[:-1], color=COLOR_PUNTOS, zorder=5, s=50, 
                edgecolor='white', alpha=0.6, label='Iteraciones')

    # 3. La Raíz final (Destacada)
    plt.scatter(raiz_final, f_num(raiz_final), color=COLOR_RAIZ, s=120, zorder=6, 
                edgecolor='white', linewidth=1.5, label=f'Raíz $\\approx {raiz_final:.4f}$')
    
    # --- FORMATO ACADÉMICO ---
    plt.title(f'Convergencia geométrica: {nombre_metodo}', fontsize=14, fontweight='bold', color='#2C3E50', pad=15)
    plt.xlabel('x', fontsize=12, color='#2C3E50')
    plt.ylabel('f(x)', fontsize=12, color='#2C3E50')
    
    # Cuadrícula y bordes limpios
    plt.grid(True, alpha=0.3, color=COLOR_EJE, linestyle='-')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color(COLOR_EJE)
    plt.gca().spines['bottom'].set_color(COLOR_EJE)
    
    plt.legend(frameon=True, edgecolor=COLOR_EJE, fancybox=True)
    
    # --- GUARDAR ---
    if guardar_pdf:
        os.makedirs("graficas_tfg", exist_ok=True)
        nombre_archivo = f"graficas_tfg/{nombre_metodo}_{funcion_str.replace('**', '^')}.pdf"
        plt.savefig(nombre_archivo, format='pdf', bbox_inches='tight')
        print(f"% [OK] Gráfica vectorial guardada en: {nombre_archivo}")
    
    plt.close()