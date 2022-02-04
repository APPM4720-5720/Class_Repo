""" 
Author: Tyler Reiser
Date Created: 2/3/22
Date Modified: 2/3/22 @ 17:50

Summary: Implementation of Newton's method to find the root of a nonlinear equation.

Variables:
- f(x) is the nonlinear equation and it's derivative dfdx is calculated by hand. 
- x0 is used to store the users input of the initial condition so it can be called on at the end in the printed result. 
- x is the initial condition used by the algorithm.
- Count stores the number of iterations. 
- The max counter is put in for the case that the initial guess does not result in a solution. 
- Epsilon is defined as machine epsilon. 
- fval is the value of y or f(x). 

Notes: 
- Storing the result in an array and printing each iteration would probably be unnecessary in for most applications.
- There is no need to waste memory by storing every iteration, so the initial condition, x, is replaced after each iteration. 
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):  # nonlinear equation
    return x**2-1


def dfdx(x):  # could be automated but a lot of extra work + error analysis
    return 2*x


# we only need one x, there is no need to store all iterations
# but for x0 is used to save the initial promp to print at the end
x0 = input('choose a starting value for x: ')  # choose an initial condition
x = float(x0)


def Newton(f, dfdx, x, epsilon):

    fval = f(x)
    count = 0

    # loop until epsilon or max iterations is reached
    while abs(fval) > epsilon and count < 100:
        try:
            x = x - float(fval)/float(dfdx(x))
        except:  # skip vals that cause div by zero, this might be helpful?
            dfdx = 0
            print('There is division by zero when x equals', x)

        fval = f(x)
        count += 1  # add one iteration to the counter

        # not necessary to print but fun to look at for small approx.
        print('We get ', fval, 'after ', count, 'iterations')

    if abs(fval) > epsilon:
        count = -1  # go back one iteration once epsilon is passed

    # stored as an array is also not necessary
    return np.array([x, f(x), count])


solution_final = Newton(f, dfdx, x, epsilon=1.0e-6)  # machine epsilon defined
x = solution_final[0]  # needed because solution is array, can be taken out
y = solution_final[1]
count = solution_final[2]


# results
print('\nThe root value is ', x)
print('\nThe value of f(x) is ', y)
print('\nThe number of iterations to reach a solution is ',
      count, ' when x=', x0, 'is chosen as an initial condition')
