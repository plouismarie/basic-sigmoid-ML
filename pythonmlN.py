import numpy as np
import matplotlib.pyplot as plt

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# needed for the graph
listX = []
listY = []

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

##var part you can play with from 3 to anything bigger
# START
# number of neural layers
n = 4 # can not be smaller than 4
#number of training iteration
trainingit=6000 # big valur will impact duration of calculation
#every trainingitsplit you will see the error value
myslice=25
trainingitsplit=trainingit / myslice
#debug mode
debug=1 # 0 for just the result, and 2 for very verbose
# END
##var part you can play with

n = n - 1
nmin = n - 1
nminmin = n - 2
nplus = n + 1
#trick to allow array in array meaning here multi-dimensional matrix in dictionary
# do not try to initialize all this in one line, you will earn hours of debugging ;)
syn = {}
l = {}
l_error = {}
l_delta = {}
# initialize weights randomly with mean 0
#syn0 = 2*np.random.random((3,1)) - 1

##syn0 = np.array([ [0,0,0.5] ]).T

#syn0 = 2*np.random.random((3,4)) - 1
#syn1 = 2*np.random.random((4,4)) - 1
#syn2 = 2*np.random.random((4,1)) - 1

syn[0] = 2*np.random.random((3,4)) - 1

# this is the part that we can put in a loop, and we want to see flexible when we play with the value n number of layers
for i in xrange(1,nmin):
    syn[i] = 2*np.random.random((4,4)) - 1

syn[nmin] = 2*np.random.random((4,1)) - 1

#number of layers here
print "+----+"
print "Number of layers in use:",nplus
print "Number of iterations in use:",trainingit

#verbose value of syn when it is initiated
if ( debug == 2 ):
    print "+----+"
    print "Here is the initial value of weights from syn0 to synX"
    for i in xrange(n):
        print "syn{}_init : {} - type: {} - shape: {}" .format(i,syn[i],syn[i].dtype,syn[i].shape)
    print "+----+"
    print 'Here is the Error value ( in Percentage ) based on number of iteration - smaller is better'

for iter in xrange(trainingit):
    # forward propagation

    l[0] = X
    for i in xrange(1,nplus):
        imin = i - 1
        l[i] = nonlin(np.dot(l[imin],syn[imin]))
    # how much did we miss?
    l_error[n] = y - l[n]
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l_delta[n] = l_error[n] * nonlin(l[n],True)
    #update weights
    syn[nmin] += np.dot(l[nmin].T,l_delta[n])

    if ( debug > 0 ):
        if (iter% trainingitsplit) == 0:
            print "iter:",iter,
            print "Error%:",float(np.mean(np.abs(l_error[n]))) * 100
            listX.append(iter)
            listY.append(float(np.mean(np.abs(l_error[n]))) * 100)


    for i in xrange(n,1,-1):
        imin=i-1
        iminmin=imin-1
        # how much did each l2 value contribute to the l3 error (according to the weights)?
        l_error[imin] = l_delta[i].dot(syn[imin].T)
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l_delta[imin] = l_error[imin] * nonlin(l[imin],True)
        # update weights
        syn[iminmin] += np.dot(l[iminmin].T,l_delta[imin])
    

if ( debug == 2 ):
    print "+----+"
    print "Here is the final value of weights from syn0 to synX"
    for i in xrange(n):
        print "syn{}_end : {}" .format(i,syn[i])

    print "+----+"
    print "Here is the value of l[N]_error from l0_error to l[X]_error"
    for i in xrange(1,nplus):
        print "error l{} : {}" .format(i,l_error[i])

print "+----+"
print "Output After full iter:",trainingit
print "Final Error%:",float(np.mean(np.abs(l_error[n]))) * 100
listX.append(trainingit)
listY.append(float(np.mean(np.abs(l_error[n]))) * 100)

if ( debug == 2 ):
    for i in xrange(1,nplus):
        print "l{} : {}" .format(i,l[i])
    print listX
    print listY

if ( debug < 2 ):
    print "l{} : {}" .format(nplus,l[n])


if ( debug > 0 ):
    plt.plot(listX, listY, 'ro')
    plt.axis([0, trainingit, 0, 10])
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Error in % (smaller is better)')
    plt.title('number of layers: '+str(nplus))
    plt.show()
