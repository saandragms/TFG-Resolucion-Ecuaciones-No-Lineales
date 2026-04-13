# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:15:30 2026

@author: saand
"""
import numpy as np
import matplotlib.pyplot as plt

def biseccion():
    
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Fuentes elegantes
    "mathtext.fontset": "cm",                          # 'cm' = Computer Modern (de LaTeX)
})

    def f(x): return x**3 - x - 1
    
    xpoints = np.linspace(-0.25,2)
    ypoints = f(xpoints)
    
    fig, ax = plt.subplots()
    
    ax.plot(xpoints, ypoints)
    ax.annotate("y=f(x)", (1.55,3), color ='C0', weight='bold')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_xlim(0.3,2)
    ax.set_ylim(-4.5,5)
    
    ax.plot([-10,10],[0,0], color = 'k', lw = '0.6')
    
    a= 0.5
    b= 1.5
    ax.plot(a, 0, marker='o',  markersize=4, color = 'k')
    ax.plot(b, 0, marker='o',  markersize=4, color = 'k')
    ax.annotate("a=a0", (a, 0), (a-0.15,-0.45), fontsize=8)
    ax.annotate("b=b0", (b, 0), (b+0.05,-0.45), fontsize=8)
    
    ax.plot([a,b],[-1.7,-1.7], color='0.5',linestyle='-', lw = 0.8)
    ax.annotate("a0", (a, -1.7), (a-0.07,-1.7-0.03), fontsize=6.5)
    ax.annotate("b0", (b, -1.7), (b+0.03,-1.7-0.03), fontsize=6.5)
    ax.plot([a,a], [0,-1.7], linestyle='--', lw = 0.8, color = '0.8')
    ax.plot([b,b], [f(b),-3.3], linestyle='--', lw = 0.8, color = '0.8')
    
    
    c0 = (a+b)/2
    ax.plot(c0,0, marker='o',  markersize=4, color = 'k')
    ax.plot(c0,-1.7, color ='0.5', marker = '|')
    ax.annotate("x0", (c0, 0), (c0,-0.45), fontsize=8)
    
    ax.plot([c0,b],[-2.5,-2.5], color='0.5', linestyle='-', lw = 0.8)
    ax.annotate("a1", (c0, -2.5), (c0-0.07,-2.5-0.03), fontsize=6.5)
    ax.annotate("b1", (b, -2.5), (b+0.03,-2.5-0.03), fontsize=6.5)
    ax.plot([c0,c0], [0,-2.5], linestyle='--', lw = 0.8, color = '0.8')

    
    c1 = (c0+b)/2
    ax.plot(c1,0,marker='o',  markersize=4, color = 'k')
    ax.plot(c1,-2.5, color ='0.5', marker = '|')
    ax.annotate("x1", (c1, 0), (c1,-0.45), fontsize=8)
    
    ax.plot([c1,b],[-3.3,-3.3], color='0.5', linestyle='-', lw = 0.8)
    ax.annotate("a2", (c1, -3.3), (c1-0.07,-3.3-0.03), fontsize=6.5)
    ax.annotate("b2", (b, -3.3), (b+0.03,-3.3-0.03), fontsize=6.5)
    ax.plot([c1,c1], [0,-4.1], linestyle='--', lw = 0.8, color = '0.8')
    
    
    c2 = (c1+b)/2
    
    ax.plot(c2,0, marker='o',  markersize=4, color = 'k')
    ax.annotate("x2", (c2, 0), (c2,-0.45), fontsize=8)
    ax.plot(c2,-3.3, color ='0.5', marker = '|')
    
    ax.plot([c1,c2],[-4.1,-4.1], color='0.5', linestyle='-', lw = 0.8)
    ax.annotate("a3", (c1, -4.1), (c1-0.07,-4.1-0.03), fontsize=6.5)
    ax.annotate("b3", (c2, -4.1), (c2+0.03,-4.1-0.03), fontsize=6.5)
    ax.plot([c2,c2], [f(c2),-4.1], linestyle='--', lw = 0.8, color = '0.8')
    
    ax.plot((c2+c1)/2,-4.1, color ='0.5', marker = '|')
    
    for p in [a,b,c0,c1,c2]:
        ax.plot(p,f(p), color = 'C0', marker = 'o', markersize=3)


    fig.savefig('proceso_biseccion.png', dpi=300, bbox_inches='tight')    

    plt.show()
    
def newton():
    
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Fuentes elegantes
    "mathtext.fontset": "cm",                          # 'cm' = Computer Modern (de LaTeX)
})
    
    tam_eje0=4
    tam_f=3
    
    def f(x): return 0.3*x**3 + 0.5*x**2 -2*x -0.5
    
    def df(x): return 0.9*x**2 + x - 2
    
    xpoints = np.linspace(0.25,3)
    ypoints = f(xpoints)
    
    fig, ax = plt.subplots()
    
    ax.plot(xpoints, ypoints)
    ax.annotate("y=f(x)", (2.5,4), color ='C0', weight='bold')
    
    ax.set_xlim(0.5,3.5)
    ax.set_ylim(-2.3,5)
    
    ax.plot([-10,10],[0,0], color = 'k', lw = '0.6')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    x0 = 1.5
    m0 = df(x0)
    tangente0_x=np.linspace(x0-0.75,3)
    tangente0_y= f(x0) + m0 *(tangente0_x - x0)
    
    ax.plot(tangente0_x,tangente0_y, color = '0.5', lw = 0.8)
    ax.plot(x0,0, color = '0.3', marker ='o', markersize=tam_eje0)
    ax.annotate("x0", (x0,0), (x0,-0.45), fontsize=8)
    ax.plot([x0,x0], [0,f(x0)], linestyle='--', lw = 0.8, color = '0.8')
    
    x1 = x0 - f(x0)/m0
    m1 = df(x1)
    tangente1_x=np.linspace(x1-1,3)
    tangente1_y= f(x1) + m1 *(tangente1_x - x1)
    
    ax.plot(tangente1_x,tangente1_y, color = '0.5', lw = 0.8)
    ax.plot(x1, 0, color = '0.3', marker = 'o', markersize=tam_eje0)
    ax.annotate("x1", (x1,0), (x1,-0.45), fontsize=8)
    ax.plot([x1,x1], [0,f(x1)], linestyle='--', lw = 0.8, color = '0.8')
    
    
    x2 = x1 - f(x1)/m1
    ax.plot(x2, 0, color = '0.3', marker = 'o', markersize=tam_eje0)
    ax.annotate("x2", (x2,0), (x2,-0.45), fontsize=8)
    
    
    i = 0
    for p in [x0,x1,x2]:
        ax.plot(p, f(p),color = 'C0', marker = 'o', markersize=tam_f)
        if not i==2: ax.annotate(f"(x{i},f(x{i}))", (p+0.05,f(p)-0.1), fontsize=6, color = "C0")
        i +=1
        
    fig.savefig('proceso_newton.png', dpi=300, bbox_inches='tight')

    plt.show()    
    
    
def secante():
    
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Fuentes elegantes
    "mathtext.fontset": "cm",                          # 'cm' = Computer Modern (de LaTeX)
})
    
    tam_eje0=4
    tam_f=3
    
    def f(x): return 2*x**2 - 2

    
    xpoints = np.linspace(0,3)
    ypoints = f(xpoints)
    
    fig, ax = plt.subplots()
    
    ax.plot(xpoints, ypoints)
    ax.annotate("y=f(x)", (1.9,4.7), color ='C0', weight='bold')
    

    ax.plot([-10,10],[0,0], color = 'k', lw = '0.6')
    
    ax.set_ylim(-2.2, 6)
    ax.set_xlim(-0.3,2.5)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    x0 = 0.2
    x1 = 1.8
    
    secante0_x= np.linspace(x0-0.75,x1)
    secante0_y=  f(x0) +((f(x0)-f(x1))/(x0-x1))*(secante0_x - x0)
    
    ax.plot(secante0_x,secante0_y, color = '0.5', lw = 0.8)
    ax.plot(x0,0, color = '0.3', marker ='o', markersize=tam_eje0)
    ax.plot(x1,0, color = '0.3', marker ='o', markersize=tam_eje0)
    
    ax.annotate("x0", (x0,0), (x0,-0.45), fontsize=8)
    ax.annotate("x1", (x1,0), (x1,-0.45), fontsize=8)
    
    ax.plot([x0,x0], [0,f(x0)], linestyle='--', lw = 0.8, color = '0.8')
    ax.plot([x1,x1], [0,f(x1)], linestyle='--', lw = 0.8, color = '0.8')
    
    x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
    
    secante12_x= np.linspace(x1,x2)
    secante12_y=  ((f(x1)-f(x2))/(x1-x2))*(secante12_x - x1) + f(x1)
    ax.plot(secante12_x,secante12_y, color = '0.5', lw = 0.8)
    ax.plot(x2,0, color = '0.3', marker ='o', markersize=tam_eje0)
    ax.annotate("x2", (x2,0), (x2,-0.45), fontsize=8)
    
    ax.plot([x2,x2], [0,f(x2)], linestyle='--', lw = 0.8, color = '0.8')
    
    
    x3 = x2 - f(x2)*(x2-x1)/(f(x2)-f(x1))
    ax.plot(x3,0, color = '0.3', marker ='o', markersize=tam_eje0)
    ax.annotate("x3", (x3,0), (x3+0.05,+0.15), fontsize=8)
    
    secante23_x= np.linspace(x2-0.1,x3+0.3)
    secante23_y=  ((f(x2)-f(x3))/(x2-x3))*(secante23_x - x2) + f(x2)
    ax.plot(secante23_x,secante23_y, color = '0.5', lw = 0.8)
    
    x4 = x3 - f(x3)*(x3-x2)/(f(x3)-f(x2))
    ax.plot(x4,0, color = '0.3', marker ='o', markersize=tam_eje0)
    ax.annotate("x4", (x4,0), (x4+0.07,+0.15), fontsize=8)

    
    i = 0
    for p in [x0,x1,x2,x3]:
        ax.plot(p, f(p),color = 'C0', marker = 'o', markersize=tam_f)
        ax.annotate(f"(x{i},f(x{i}))", (p+0.05,f(p)-0.1), fontsize=6, color = "C0")
        i +=1
        
    fig.savefig('proceso_secante.png', dpi=300, bbox_inches='tight')

    plt.show() 
