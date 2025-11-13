import math
import random 


def accept_reject_gen(f,g,sample_g,a):
    """
    f  is the pdf f(.) of X random var that we want to estimate but its hard to
    compute in a closed form
    
    g  is the pdf g(.) of Y rand.var that it is easy to find
    
    We do the accept-reject approach : If a =const exist : f(.)/g(.)  <=a
    
    sample_g : Function that draws Y~g
    
    """
    
    while True:
        Y = sample_g #generate Y
        U = random.random() #generate Uniform(0,1) U
        
        if U < f(Y)/(a*g(Y)):
            return Y
        
"""
Notes:
    Let N the number of loops until 1st Acceptance: This means the following
    N~Geom(p) , where p=1/a
    E[N]=1/p = 1/a. Thus we need to choose an a that rises the acceptance prob
    or accept ratio. So a = supremum(f(x)/g(x)) with respect to x. Also to
    achieve this a we need the ratio to be propotionally ->1 so we need to also 
    choose a g(.) similar to f(.)


"""