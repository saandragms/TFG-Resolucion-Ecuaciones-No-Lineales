import sympy as sp
import pandas as pd
import numpy as np
from graficas import plot_convergencia
from metodos import metodo_biseccion, metodo_newton_raphson, metodo_secante # Asegúrate de importar tu método

def ejecutar_analisis(f_str, metodo_func, args_metodo, usa_derivada=False, generar_grafica=False):
    """
    Generar tabla LaTeX y (opcionalmente) gráficas adaptándose a tu metodos.py
    """
    x = sp.symbols('x')
    expr_f = sp.sympify(f_str)
    f_num = sp.lambdify(x, expr_f, 'numpy')

    if usa_derivada:
        expr_df = sp.diff(expr_f, x)
        df_num = sp.lambdify(x, expr_df, 'numpy')
        res = metodo_func(f=f_num, df=df_num, **args_metodo)
    else: res = metodo_func(f=f_num,  **args_metodo)
    
    # 4. Formatear las columnas
    puntos_formateados = [f"{p:.6f}" for p in res['historial_puntos']]
    errores_formateados = [f"{e:.2e}" for e in res['historial_errores']]
    
    # 5. Crear el DataFrame
    df = pd.DataFrame({
        'n': range(1, len(puntos_formateados) + 1),
        'p_n': puntos_formateados,
        'Error (|p_n - p_{n-1}|)': errores_formateados
    })
    
    # 6. Truncar tablas largas con puntos suspensivos
    if len(df) > 10:
        df_head = df.head(4).copy()
        df_tail = df.tail(4).copy()
        df_medio = pd.DataFrame([{"n": "\\vdots", "p_n": "\\vdots", "Error (|p_n - p_{n-1}|)": "\\vdots"}])
        df = pd.concat([df_head, df_medio, df_tail], ignore_index=True)
        
    # 7. Generar LaTeX
    nombre_metodo = res['nombre'] # Coge la key "nombre" de tu diccionario
    
    codigo_latex = (
        df.style
        .hide(axis="index")
        .to_latex(
            caption=f"Convergencia del método de {nombre_metodo} para $f(x)={f_str}$.",
            label=f"tab:analisis_{nombre_metodo.lower()}",
            hrules=True,
            column_format="c c c"
        )
    )
    
    # Centrar en LaTeX
    codigo_latex = codigo_latex.replace("\\begin{table}", "\\begin{table}[h!]\n\\centering")
    
    print(codigo_latex)
    print("\n" + "%" * 50 + "\n")
    
    # 8. Generar gráfica
    if generar_grafica:
        plot_convergencia(f_str, res, guardar_pdf=True)

if __name__ == "__main__":
    # --- ZONA DE CONFIGURACIÓN ---
    MI_FUNCION = "x**2 - 2"

    MI_METODO = metodo_newton_raphson
    
    if MI_METODO == metodo_newton_raphson:  
        derivada = True
    else: derivada = False
        

    # Argumentos específicos.
    MIS_ARGUMENTOS = {
        "x0": 1.0,
        "tol": 1e-8
    }

    ejecutar_analisis(
        f_str=MI_FUNCION,
        metodo_func=MI_METODO, 
        args_metodo=MIS_ARGUMENTOS, 
        usa_derivada=derivada,
        generar_grafica=True
    )