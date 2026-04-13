from tabulate import tabulate
import pandas as pd

def tabla(f :str, resultados: dict, guardar_archivo: bool = False):

    df = pd.DataFrame({"Aproximación x_n":resultados["historial_puntos"],
                       "Evaluación f(x_n)": resultados["historial_evaluaciones"],
                       "Error de aproximación": resultados["historial_errores"]})

    df.index.name = "Iteración"

    encabezado = f"--- Tabla de la ecuación {f} = 0 ---\n----- Método de {resultados['nombre']} -----\n"
    tabla = tabulate(df, 
                    headers='keys',
                    tablefmt = "pipe", # Cambiar "pipe" por "latex" para formato LaTeX.
                    colalign=("center", "center", "center", "center"))
    
    if guardar_archivo:
        with open(f"tabla_{f}_{resultados['nombre']}.txt", "w") as f:
            f.write(encabezado + tabla)
    
    print(encabezado + tabla)
