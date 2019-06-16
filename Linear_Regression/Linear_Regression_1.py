#%matplotlib inline
#imports
from numpy import *
import matplotlib.pyplot as plt
points = genfromtxt('C:/Users/New User/Desktop/ML_Practice/Data_Resources/Linear_Reg_Dataset_1.csv', delimiter = ',')
print(points)

x = array(points[:,0])
y = array(points[:,1])


%matplotlib inline
from numpy import *
import matplotlib.pyplot as plt
points = genfromtxt('C:/Users/New User/Desktop/Linear_Regression/data.csv', delimiter = ',')
#print(points)

x = array(points[:,0])
y = array(points[:,1])
#plot the dataset
plt.scatter(x,y)
plt.xlabel('Hours of Study')
plt.ylabel('Test Scores')
plt.title('Dataset')

#plt.show() will display the current figure that you are working on.
#plt.draw() will re-draw the figure. This allows you to work in interactive mode and, should you have changed your data or formatting, allow the graph itself to change.

plt.show()


#define the parameters

learning_rate = 0.0001
initial_b = 0
initial_m = 0
num_iterations = 10

#define the cost function

#In ML, Cost Functions are used to estimate how badly models are performing.
#A Cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between X and y.

def compute_cost(b,m,points):
    total_cost = 0
    N = float(len(points))
    
    #compute the sum of squared errors
    
    for i in range(0, len(points)):
        x = points[i , 0]
        y = points[i , 1]
        total_cost += (y-(m * x + b)) ** 2
        
    #return the average of squared error
    return total_cost/N


#Run gradient_descent_runner() to get optimized parameters b and m
#Gradient Descent is a first-order iterative optimization algorithm for finding the minimum of a function.


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    cost_graph = []
    
    #For every iteration, optimize b,m and compute its cost
    for i in range(num_iterations):
        cost_graph.append(compute_cost(b, m, points))
        b, m = step_gradient(b, m, array(points), learning_rate)
    
    return [b, m, cost_graph] 


def step_gradient(b_current, m_current, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    
    N = float(len(points))
    
    #Calculate gradient
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        m_gradient += - (2/N) * x * (y - (m_current * x + b_current))
        b_gradient += - (2/N) * (y - (m_current * x + b_current))
    
    #Update current m and b
    m_updated = m_current - learning_rate * m_gradient
    b_updated = b_current - learning_rate * b_gradient
    
    #Return updated Parameters
    return b_updated, m_updated

b, m, cost_graph = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

#Print optimized parameteres
print('Optimized b:', b)
print('Optimized m:', m)

#Print error with optimized parameteres
print('Minimized cost:', compute_cost(b, m, points))


plt.plot(cost_graph)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost Per Iteration')
plt.show()



plt.scatter(x, y)

#prefidct y values

pred = m * x + b

#plot predictions as line of best fit

#line of best fit

plt.plot(x, pred, c = 'r')
plt.xlabel('Hours of Study')
plt.ylabel('Test scores')
plt.title('Line of Best Fit')
plt.show()



    