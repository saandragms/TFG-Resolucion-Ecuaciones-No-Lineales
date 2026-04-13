# import math
import metodos as mt
import sympy as sp
from tablar import tabla


def run(f_str : str, metodo : str, cond_iniciales : list, genera_tabla = True):

    dicc_metodos = {"Bisección": mt.biseccion,
               "Newton-Raphson": mt.newton_raphson,
               "Secante": mt.secante,
               "Regula-Falsi": mt.regula_falsi,
               "Newton Modificado": mt.newton_modificado,
                "Steffensen": mt.steffensen
               }
    
    # Convertir la función a formato numérico  
    x = sp.symbols('x')
    f_sp = sp.sympify(f_str)
    f_num = sp.lambdify(x, f_sp, 'numpy')

    # Si el método requiere derivada, calcularla y convertirla a formato numérico.
    if metodo in ["Newton-Raphson", "Newton Modificado"]:
        # Calcular derivada
        df_sp = sp.diff(f_sp)
        df_num = sp.lambdify(x, df_sp, 'numpy')
        # Si el metodo requiere segunda derivada, calcularla y convertirla a formato numérico.
        if metodo in ["Newton Modificado",]: 
            ddf_sp = sp.diff(df_sp)
            ddf_num = sp.lambdify(x, ddf_sp, 'numpy')
            cond_iniciales.insert(0,ddf_num)
        cond_iniciales.insert(0,df_num)

    # Ejecutar el método seleccionado con la función y las condiciones iniciales.
    res = dicc_metodos[metodo](f_num, *cond_iniciales)

    # Mensaje de convergencia o no convergencia
    print(res["mensaje"])

    # Si se ha alcanzado la convergencia, mostrar resultados y generar tabla.
    if res["convergencia"]:
        print(f"Se ha utilizado el método de {res['nombre']} para aproximar la solución de la ecuación {f_str} = 0.")
        print(f"La solución aproximada calculada es: {res['raiz']}.")
        print(f"Se ha calculado en {res['iteraciones']} iteraciones.\n")

        # Generar tabla de resultados. Por defecto, se muestra por consola. 
        # Añadir el argumento guardar_archivo=True, para guardar en un archivo de texto. 
        if genera_tabla:
            tabla(f_str, resultados=res, guardar_archivo=False)


if __name__ == "__main__":
    # --- ZONA DE CONFIGURACIÓN ---
    # Función a resolver. Tipo string. Ejemplo: "cos(x) - x"
    MI_FUNCION = "cos(x) - x"
   
    # Nombre de los métodos disponibles. Tipo string. 
    metodos = ["Bisección", "Newton-Raphson", "Secante", "Regula-Falsi", "Newton Modificado", "Steffensen"]
    MI_METODO = metodos[5]

    """Argumentos específicos. Datos iniciales para cada método. Tipo lista. 
       Bisección: [a, b]
       Newton-Raphson: [x0]
       Secante: [x0, x1]
       Regula-Falsi: [a, b]
       Newton Modificado: [x0]
       Steffensen: [x0]"""
    
    MIS_ARGUMENTOS = [2,]

    #Función para testear cada método. Recibe la función, el método a usar y los argumentos específicos.
    run(f_str=MI_FUNCION, metodo=MI_METODO, cond_iniciales=MIS_ARGUMENTOS)

    