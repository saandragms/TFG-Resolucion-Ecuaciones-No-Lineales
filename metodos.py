import numpy as np
import time
from typing import Callable, Dict, Any

def metodo_biseccion(f: Callable[[float], float], a: float, b: float,
                      tol: float = 1e-10, max_iter: int=50) -> Dict[str, Any]:

    if f(a) * f(b) >= 0:  raise ValueError("Error: No se cumple el teorema de Bolzano.")
    errores, raices, conv = [], [], False
    t0,= time.perf_counter()
    fa = f(a)
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        raices.append(c)
        errores.append(abs(c - a))

        if errores[-1] < tol or fc == 0:
            conv = True
            break

        if fa * fc < 0: b = c
        else: a, fa = c, fc
        x0 = c

    return {
        "nombre": "Bisección", "funcion": f, "raiz": c, "iteraciones": i + 1,
        "historial_errores": errores, "historial_puntos": raices, "convergencia": conv,
        "tiempo": time.perf_counter() - t0, 
        "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."
    }

def metodo_newton_raphson(f: Callable[[float], float], df: Callable[[float], float], x0: float, 
                          tol: float = 1e-10, max_iter: int=50) -> Dict[str, Any]:

    errores, raices, conv = [], [], False
    t0 = time.perf_counter()
    for i in range(max_iter):
        fx0 = f(x0)
        dfx0 = df(x0)

        if dfx0 == 0: raise ValueError("Error: Derivada nula en el punto x0.")

        x1 = x0 - (fx0 / dfx0)
        raices.append(x1)
        errores.append(abs(x1 - x0))

        if errores[-1] < tol or f(x1) == 0:
            conv = True
            break

        x0 = x1

    return {
        "nombre": "Newton-Raphson", "funcion": f, "raiz": x1, "iteraciones": i + 1,
        "historial_errores": errores, "historial_puntos": raices, "convergencia": conv,
        "tiempo": time.perf_counter() - t0,
        "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."
    }

def metodo_secante(f: Callable[[float], float], x0: float, x1: float, 
                   tol: float = 1e-10, max_iter: int=50) -> Dict[str, Any]:

    errores, raices, conv = [], [], False
    t0 = time.perf_counter()

    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if fx1 - fx0 == 0: raise ValueError("Error: División por cero en el método de la secante.")
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        raices.append(x2)
        errores.append(abs(x2 - x1))
        if errores[-1] < tol or f(x2) == 0:
            conv = True
            break
        x0, x1 = x1, x2
    
    return {
        "nombre": "Secante", "funcion": f, "raiz": x1, "iteraciones": i + 1,
        "historial_errores": errores, "historial_puntos": raices, "convergencia": conv,
        "tiempo": time.perf_counter() - t0, 
        "mensaje": "Convergencia alcanzada." if conv else "Máximo de iteraciones alcanzado."
    }