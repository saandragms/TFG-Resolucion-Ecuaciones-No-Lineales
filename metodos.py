
import time
from typing import Callable, Dict, Any
import numpy as np

"""
while i < max_iter and not conv:
    #calulo del nuevo punto
    #calculo del error
    if error < tolx or abs(f(nuevo_punto)) < tolf:
        conv = True
    #actualización de variables para la siguiente iteración
"""
def biseccion(f: Callable[[float], float], a: float, b: float,
              tolx =1e-6, tolf = 1e-8, max_iter: int=100) -> Dict[str, Any]:
    if f(a) * f(b) >= 0:  raise ValueError("Error: La condición inical no satisface el teorema de Bolzano.")
    errores, raices, evaluaciones, conv = [abs(b - a)/2,], [a,], [f(a),], False
    t0 = time.perf_counter()
    i, fa = 0, f(a)

    while i < max_iter and not conv:
        x = a + (b-a) / 2
        fx = f(x)

        raices.append(x)
        evaluaciones.append(fx)
        errores.append(abs((b - a) / 2))

        if errores[-1] < tolx and abs(fx) < tolf:
            conv = True
        else:   
            if fa * fx < 0: b = x
            else: a, fa = x, fx
        i += 1
    t1 = time.perf_counter()
    return {
        "nombre": "Bisección", "funcion": f, "raiz": x, "iteraciones": i,
        "historial_errores": errores, "historial_puntos": raices, "historial_evaluaciones": evaluaciones,
        "convergencia": conv, "tiempo": t1 - t0, "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."}


def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], x0: float,
                     tolx: float = 1e-6, tolf: float = 1e-8, max_iter: int=100) -> Dict[str, Any]:
    
    errores, raices, evaluaciones, conv = [np.nan,], [x0,],  [f(x0),], False
    t0 = time.perf_counter()
    i, fx0 = 0, f(x0)
    
    while i < max_iter and not conv:
        dfx0 = df(x0)
        
        if dfx0 == 0: raise ValueError("Error: Derivada nula.")

        x1 = x0 - (fx0 / dfx0)
        fx1 = f(x1)
        raices.append(x1)
        evaluaciones.append(fx1)
        errores.append(abs(x1 - x0))

        if errores[-1] < tolx and abs(fx1) < tolf:
            conv = True
        else:
            x0 = x1
            fx0 = fx1
        i += 1
    t1 = time.perf_counter()
    return {
        "nombre": "Newton-Raphson", "funcion": f, "raiz": x1, "iteraciones": i,
        "historial_errores": errores, "historial_puntos": raices,
        "historial_evaluaciones": evaluaciones, "convergencia": conv,
        "tiempo": t1 - t0, "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."}


def secante(f: Callable[[float], float], x0: float, x1: float,
            tolx: float = 1e-6, tolf: float = 1e-8, max_iter: int=100) -> Dict[str, Any]:   
    
    errores, raices, evaluaciones, conv = [np.nan, abs(x1 - x0)], [x0,x1], [], False
    t0 = time.perf_counter()    
    fx0, fx1 = f(x0), f(x1)
    evaluaciones.append(fx0)
    evaluaciones.append(fx1)
    i = 0
    while i < max_iter and not conv:
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = f(x2)
        raices.append(x2)
        evaluaciones.append(fx2)
        errores.append(abs(x2 - x1))
        if errores[-1] < tolx and abs(fx2) < tolf:
            conv = True 
        else:
            x0, x1 = x1, x2
            fx0, fx1 = fx1, fx2
        i += 1
    t1 = time.perf_counter()
    return {
        "nombre": "Secante", "funcion": f, "raiz": x2, "iteraciones": i,
        "historial_errores": errores, "historial_puntos": raices,
        "historial_evaluaciones": evaluaciones, "convergencia": conv,
        "tiempo": t1 - t0, "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."}    
        
def regula_falsi(f: Callable[[float], float], x0: float, x1: float,
                 tolx: float = 1e-6, tolf: float = 1e-8, max_iter: int=100) -> Dict[str, Any]:      
    errores, raices, evaluaciones, conv = [np.nan, abs(x1 - x0)], [x0,x1], [], False
    t0 = time.perf_counter()
    fx0, fx1 = f(x0), f(x1)
    evaluaciones.append(fx0)
    evaluaciones.append(fx1)

    if fx0 * fx1 >= 0:  raise ValueError("Error: La condición inical no satisface el teorema de Bolzano.")
    
    i = 0
    while i < max_iter and not conv:
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = f(x2)
        raices.append(x2)
        evaluaciones.append(fx2)
        errores.append(abs(x2 - x1))
        if errores[-1] < tolx and abs(fx2) < tolf:
            conv = True 
        else:
            if fx2 * fx1 < 0: # Raiz el intervalo entre x1 y x2
                x0, fx0 = x1, fx1
                
            x1, fx1 =  x2, fx2
        i += 1
    t1 = time.perf_counter()
    return {
        "nombre": "Regula Falsi", "funcion": f, "raiz": x2, "iteraciones": i,
        "historial_errores": errores, "historial_puntos": raices,
        "historial_evaluaciones": evaluaciones, "convergencia": conv,
        "tiempo": t1 - t0, "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."}


#Newton modificado para raices múltiples
def newton_modificado(f: Callable[[float], float], df: Callable[[float], float], ddf: Callable[[float], float],
                     x0: float, tolx: float = 1e-6, tolf: float = 1e-8, max_iter: int=100) -> Dict[str, Any]:
    
    errores, raices, evaluaciones, conv = [np.nan,], [x0,],  [f(x0),], False
    t0 = time.perf_counter()
    i, fx0 = 0, f(x0)
    
    while i < max_iter and not conv:
        dfx0 = df(x0)
        ddfx0 = ddf(x0)
        
        if (dfx0**2 - fx0 * ddfx0)==0: raise ValueError(f"Error en interación {i}: Denominador nulo.")

        x1 = x0 - (fx0 * dfx0) / (dfx0**2 - fx0 * ddfx0)
        fx1 = f(x1)
        raices.append(x1)
        evaluaciones.append(fx1)
        errores.append(abs(x1 - x0))

        if errores[-1] < tolx or abs(fx1) < tolf:
            conv = True
        else:
            x0 = x1
            fx0 = fx1
        i += 1
    t1 = time.perf_counter()
    return {
        "nombre": "Newton Modificado", "funcion": f, "raiz": x1, "iteraciones": i,
        "historial_errores": errores, "historial_puntos": raices,
        "historial_evaluaciones": evaluaciones, "convergencia": conv,
        "tiempo": t1 - t0, "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."}

# Variaciones Newton sin derivadas
def steffensen(f: Callable[[float], float], x0: float, 
               tolx: float = 1e-6, tolf: float = 1e-8, max_iter: int=100) -> Dict[str, Any]:
    errores, raices, evaluaciones, conv = [np.nan,], [x0,],  [f(x0),], False
    t0 = time.perf_counter()
    i, fx0 = 0, f(x0)
    
    while i < max_iter and not conv:
        y0 = x0 + fx0
        fy0 = f(y0)

        if (fy0 - fx0) == 0: raise ValueError(f"Error en interación {i}: Denominador nulo.")

        x1 = x0 - (fx0**2) / (fy0 - fx0)
        fx1 = f(x1)
        raices.append(x1)
        evaluaciones.append(fx1)
        errores.append(abs(x1 - x0))

        if errores[-1] < tolx and abs(fx1) < tolf:
            conv = True
        else:
            x0 = x1
            fx0 = fx1
        i += 1
    t1 = time.perf_counter()
    return {
        "nombre": "Steffensen", "funcion": f, "raiz": x1, "iteraciones": i,
        "historial_errores": errores, "historial_puntos": raices,
        "historial_evaluaciones": evaluaciones, "convergencia": conv,
        "tiempo": t1 - t0, "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."}