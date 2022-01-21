import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

############################Methods for NumericalIntegration###############################

def numpyEquation(equation):
    fun={"sin":"np.sin","cos":"np.cos","tan":"np.tan","sqrt":"np.sqrt","exp":"np.exp","ln":"np.log","pi":"np.pi","log10":"np.log10"}
    for i in fun:
        if i in equation:
            equation = equation.replace(i,fun[i])
    return equation

def dataEntryNumericalIntegration():
    a = 0
    b = 0.8
    sections = 1
    h = (b-a)/sections
    integralValue = 1.640533
    return a,b,h,sections,integralValue

def functionEvaluation(a,sections,h):
    print("-----------Evaluate Function---------")
    xi = a
    for _ in range(sections + 1):
        print(f"f({xi})={fx2(xi)}")
        xi = xi + h

def areaCalculate(equationList, a,sections,h,b):
    xi = a
    add = 0
    print("--------Trapezium area---------")
    if(equationList == "h*(fx2(xi)+4*fx2(xi+h)+fx2(xi+2*h))*(1/3)"): sections = sections - 1
    elif(equationList == "3*h*(fx2(a)+3*fx2(a+h)+3*fx2(a+2*h)+fx2(a+3*h))*(1/8)"): sections = sections -2
    else: sections
    for _ in range(sections):
        trapeziumArea = eval(equationList)
        add = add + trapeziumArea
        print(add)
        xi = xi + h
    print(f"I = {add}")
    return add

def error(integralValue, I):
    E = integralValue - I
    Et = ((integralValue - I)/integralValue)*100
    print("------------Getting the Error-----------")
    print(f"Et={E}, Et%={Et}%")

def nDerivative(fx, nderivative):
    for _ in range(nderivative):
        y = sp.diff(fx,sp.Symbol('x'))
        fx = y
    print(f"---------------Getting the {nderivative} derivative-----------")
    print(f"f{nderivative}(x)'={y}")
    return y

def definiteIntegral(y,a,b):
    print("------------Definite Integral-----------")
    Id = sp.integrate(y, (sp.Symbol('x'),a,b))
    print(f"Id={Id.evalf(6)}")
    return Id

def averageValueOfTheDerivative(Id,a,b):
    print("--------Average value of the derivative ------")
    P2D = Id/(b-a)
    print(f"faverae(x)={P2D.evalf(6)}")
    return P2D

def trapezoidRuleError(equationErrorList,P2D,a,b,tramos):
    print("------------Trapezoid rule error----------")
    Et = eval(equationErrorList)
    print(f"Ea={Et.evalf(6)}")

#Special Method to compund simpsion 1/3
def areaCalculateSpecial(equationList, a,sections,h,b):
    xi = a+h
    evenAdd = 0
    oddAdd = 0
    print("--------Trapezium area---------")
    for _ in range(1,sections):
        if(_%2 != 0):
            oddAdd = oddAdd+fx2(xi)
        else:
            evenAdd = evenAdd+fx2(xi)
        xi = xi + h
    Integral = eval(equationList)
    print(f"I = {Integral}")
    return Integral

def graficate(a,b,sections):
    xi = np.linspace(a,b,sections+1) 
    fi = fx2(xi)
    linesSample = (sections+1)*10
    xk = np.linspace(a,b,linesSample)
    fk = fx2(xk)
    plt.plot(xi,fi, 'ro')
    plt.plot(xk, fk)
    plt.fill_between(xi,0,fi, color='g')
    for i in range(sections+1):
        plt.axvline(xi[i], color='w')
    plt.show()

#######################################Taylor Serie############################
def dataEntryTaylorSerie():
    nderivative = 4
    x = 0
    h = 1
    taylorSerie = [fx2(x)]
    approach = [fx2(x)]
    truncationError = [fx2(1)-fx2(0)]
    equationsList=[equation]
    return nderivative, x, h, taylorSerie, approach, truncationError, equationsList

def graficateTaylor(x,h):
    delta = 100
    xp = np.linspace(x, h, delta)
    plt.plot(xp, fx2(xp), label=("f(x)"))
    plt.plot()                    
    plt.xlabel("x axis")       
    plt.ylabel("y axis")       
    plt.title("Graph of the functions")
    plt.legend()           
    plt.show()     

###########################Newton's Interpolation########################
def dataEntryNewtonInterpolation():
    xi = np.array([1, 4, 6, 5])
    fi = np.array([0, 1.386294, 1.791759, 1.609438])
    return xi, fi

def graficateNewtonInterpolation(xi, fi, pxi, pfi):
    plt.plot(xi,fi,'o', label = 'Points')
    plt.plot(pxi,pfi, label = 'Polynomial')
    plt.legend()
    plt.xlabel('xi')
    plt.ylabel('fi')
    plt.title("Newtons's Interpolation")
    plt.show()
###################################################

def menu():
    """
    Clean Screen
    """
    # os.system('cls') # NOTA para windows tienes que cambiar clear por cls
    print("Select a method")
    print("\t1 - Simple Trapezoid")
    print("\t2 - Compound Trapezoid")
    print("\t3 - Simple Simpson 1/3")
    print("\t4 - Compound Simpson 1/3")
    print("\t5 - Simpson 3/8")
    print("\t6 - Taylor Serie")
    print("\t7 - Newton's Interpolation")
    print("\t9 - exit")

diccionary = {
        "1": lambda: simpleTrapezoid(),
        "2": lambda: compundTrapezoid(),
        "3": lambda: simpleSimpson1_3(),
        "4": lambda: compoundSimpson1_3(),
        "5": lambda: simpleSimpson3_8(),
        "6": lambda: TaylorSerie(),
        "7": lambda: newtonInterpolation()
    }

def simpleTrapezoid():
    a,b,h,sections,integralValue = dataEntryNumericalIntegration()
    functionEvaluation(a,sections,h)
    I = areaCalculate(equationList[0],a,sections,h,b)
    error(integralValue, I)
    y = nDerivative(equation,2)
    Id = definiteIntegral(y,a,b)
    P2D = averageValueOfTheDerivative(Id, a, b)
    trapezoidRuleError(equationErrorList[0],P2D,a,b, sections)
    graficate(a,b,sections)

def compundTrapezoid():
    a,b,h,sections,integralValue = dataEntryNumericalIntegration()
    functionEvaluation(a,sections,h)
    I = areaCalculate(equationList[1],a,sections,h,b)
    error(integralValue, I)
    y = nDerivative(equation,2)
    Id = definiteIntegral(y,a,b)
    P2D = averageValueOfTheDerivative(Id, a, b)
    trapezoidRuleError(equationErrorList[1],P2D,a,b,sections)
    graficate(a,b,sections)

def simpleSimpson1_3():
    a,b,h,sections,integralValue = dataEntryNumericalIntegration()
    functionEvaluation(a,sections,h)
    I = areaCalculate(equationList[2],a,sections,h,b)
    error(integralValue, I)
    y = nDerivative(equation,4)
    Id = definiteIntegral(y,a,b)
    P2D = averageValueOfTheDerivative(Id, a, b)
    trapezoidRuleError(equationErrorList[2],P2D,a,b,sections)
    graficate(a,b,sections)

def compoundSimpson1_3():
    a,b,h,sections,integralValue = dataEntryNumericalIntegration()
    functionEvaluation(a,sections,h)
    I = areaCalculateSpecial(equationList[3],a,sections,h,b)
    error(integralValue, I)
    y = nDerivative(equation,4)
    Id = definiteIntegral(y,a,b)
    P2D = averageValueOfTheDerivative(Id, a, b)
    trapezoidRuleError(equationErrorList[3],P2D,a,b,sections)
    graficate(a,b,sections)
    
def simpleSimpson3_8():
    a,b,h,sections,integralValue = dataEntryNumericalIntegration()
    functionEvaluation(a,sections,h)
    I = areaCalculate(equationList[4],a,sections,h,b)
    error(integralValue, I)
    y = nDerivative(equation,4)
    Id = definiteIntegral(y,a,b)
    P2D = averageValueOfTheDerivative(Id, a, b)
    trapezoidRuleError(equationErrorList[4],P2D,a,b,sections)
    graficate(a,b,sections)

def TaylorSerie():
    nderivative, x, h, taylorSerie, approach, truncationError, equationsList = dataEntryTaylorSerie()
    global equation
    for i in range(1,nderivative+1):
        y = sp.diff(equation,sp.Symbol('x'))
        fx2 = lambda x: eval(numpyEquation(str(y)))
        evaluar = fx2(x)
        taylorSerie.append(evaluar*(h**2)/factorial(abs(i)))
        equation = y
        equationsList.append(str(equation))
        approach.append(sum(taylorSerie))
        truncationError.append(fx2(1)-approach[i])
        print(f"Ecuacion {y} con evaluar f({x}) = {evaluar}")
    print(taylorSerie)
    graficateTaylor(x,h)

def newtonInterpolation():
    xi, fi = dataEntryNewtonInterpolation()
    # Newtons Interpolation Table
    title = ['i   ','xi  ','fi  ']
    n = len(xi)
    ki = np.arange(0,n,1)
    table = np.concatenate(([ki],[xi],[fi]),axis=0)
    table = np.transpose(table)

    # Newton's Interpolation Empty
    dfinite = np.zeros(shape=(n,n),dtype=float)
    table = np.concatenate((table,dfinite), axis=1)

    # Calculate table, start in column 3
    [n,m] = np.shape(table)
    diagonal = n-1
    j = 3
    while (j < m):
        # Add title for ech column
        title.append('F['+str(j-2)+']')
        # each column row
        i = 0
        paso = j-2 # start in 1
        while (i < diagonal):
            denominator = (xi[i+paso]-xi[i])
            numerator = table[i+1,j-1]-table[i,j-1]
            table[i,j] = numerator/denominator
            i = i+1
        diagonal = diagonal - 1
        j = j+1

    # Polynomial
    # Equidistant points on the x-axis
    dDivided = table[0,3:]
    n = len(dfinite)

    # Polynomial expression with Sympy
    x = sp.Symbol('x')
    polynomial = fi[0]
    for j in range(1,n,1):
        factor = dDivided[j-1]
        term = 1
        for k in range(0,j,1):
            term = term*(x-xi[k])
        polynomial = polynomial + term*factor

    # Simplify by multiplying by (x-xi)
    polisimple = polynomial.expand()

    # Polynomial for numerical evaluation
    px = sp.lambdify(x,polisimple)

    # Points for the graph
    samples = 101
    a = np.min(xi)
    b = np.max(xi)
    pxi = np.linspace(a,b,samples)
    pfi = px(pxi)

    # Exit
    np.set_printoptions(precision = 4)
    print('********Newtons Interpolation Table')
    print([title])
    print(table)
    print('dDivided: ')
    print(dDivided)
    print('polynomial: ')
    print(polynomial)
    print('simplified polynomial: ' )
    print(polisimple)
    graficateNewtonInterpolation(xi, fi, pxi, pfi)

def menuBucle():
    while True:
        menu()
        option = input("Enter the option >> ")
        if (option != "9"): diccionary.get(option, lambda: None)()
        else: break
    

if __name__ == '__main__':
    equationList = [
            "(b-a)*(fx2(xi)+fx2(xi+h))/2",
            "h*(fx2(xi)+fx2(xi+h))/2",
            "h*(fx2(xi)+4*fx2(xi+h)+fx2(xi+2*h))*(1/3)",
            "(b-a)*(1/(3*sections))*(fx2(a)+4*sumaImpar+2*sumaPar+fx2(b))",
            "3*h*(fx2(a)+3*fx2(a+h)+3*fx2(a+2*h)+fx2(a+3*h))*(1/8)"
        ]
    equationErrorList = [
            "-(1/12)*(P2D)*(b-a)**3",
            "-(1/(12*sections**2))*(P2D)*(b-a)**3",
            "-(1/2880)*(P2D)*(b-a)**5",
            "-(1/(180*sections**4))*(P2D)*(b-a)**5",
            "-(1/6480)*(P2D)*(b-a)**5"
            ]
    global equation, equationToEvaluate
    equation = "0.2+25*x-200*x**2+675*x**3-900*x**4+400*x**5"
    # equation = "x**3/(1+x**(1/2))"
    # equation = "(1/sqrt(2*pi))*e(-(x**2)/2)"
    # equation = "ln(1+x)"
    # equation = "exp(x)"
    # equation = "sin(x)"
    # equation = "exp(2*x)"
    # equation = "exp(x**2)"
    # equation = "cos(x**2)"
    # equation = "1/x"
    equationToEvaluate = numpyEquation(equation)
    fx = lambda x: eval(equation)
    fx2 = lambda x: eval(equationToEvaluate)
    menuBucle()