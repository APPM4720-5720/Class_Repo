""" 
Author: Tyler Reiser
Date Created: 2/3/22
Date Modified: 2/3/22 @ 20:55 MST

Summary: Implementation of Newton's method to find the root of a system of nonlinear equations.


- The nonlinear equations used are:
def f(x,y):  
    return x**2-1+y*x**3

def g(x,y): 
    return x*y+y+x 
- The partial derivatives were calculated by hand and stored in the Jacobian matrix
- Both equations are stored in the vector function as an array.
- x0 and x1 are used to store the users input of the initial condition to be called on at the end in the printed result. 
- x is the initial condition used by the algorithm which stores x0 and x1 in an array.
- The two variables in the nonlinear equations are stored in x after each iteration.
- Count stores the number of iterations. 
- The max counter is put in for the case that the initial guess does not result in a solution. 
- Epsilon is defined as machine epsilon. 
- fval is the value of y or f(x). 
- Fnorm is the 2 norm calculated by NumPy

Note: since x,y values are stored in an array called x, x[0]=x and x[1]=y
"""

import numpy as np


x0 = input('choose a starting value for x: ')  # choose initial condition, x
x1 = input('choose a starting value for y: ')  # choose initial condition, y
x = np.array([int(x0), int(x1)])


def Vect_Function(x):  # the vector function stores f and g
    return np.array([x[0]**2-1+x[1]*x[0]**2, x[0]*x[1] + x[1] + x[0]])


def Jacobian(x):  # partial derivatives calculated/added by hand
    return np.array([[2*x[0]+2*x[1]*x[0], x[0]**2], [x[1]+1, x[0]+1]])


def Newton_SystEq(Vect_Function, Jacobian, x, epsilon):

    fval = Vect_Function(x)
    Fnorm = np.linalg.norm(fval, ord=2)  # L2 norm by numpy
    count = 0

    # loop until epsilon or max iterations is reached
    while abs(Fnorm) > epsilon and count < 50:
        term = np.linalg.solve(Jacobian(x), -fval)
        x = x + term
        fval = Vect_Function(x)
        Fnorm = np.linalg.norm(fval, ord=2)
        count += 1  # add one iteration to the counter

        # not necessary to print but fun to look at for small approx.
        print('We get ', fval, 'after ', count, 'iterations')

    if abs(Fnorm) > epsilon:
        count = -1  # go back one iteration once epsilon is passed

    return x, count  # x output is an 1x2 array, count is an int


x, count = Newton_SystEq(Vect_Function, Jacobian, x, epsilon=1.0e-6)


# results
print('\nThe root value is ', x)
print('\nThe number of iterations to reach a solution is ', count,
      ' when initial conditions x and y are ', x0, ' and ', x1)
