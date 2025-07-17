#number of possible outcomes 
num_of_points = 8
#training data
num_symbols = 100
#control data for the loops
num_epochs= 100
learning_rate=0.1

# nedded info for multilayer 
a = 1
b = 1
# why 2? because i have 2 componenst, inphase and quadrature, if i increse it i will need to implement a multiple point comparison 
input_dim = 2
#hidden dimention i want it to be dynamic, soo it will change with the numbers of possible labels soo will change will depend on the num_of_points since 
#each point is 100% independed, each have its own code with gray coding of error reduction "not implemented yet"
hidden_dim = round(num_of_points*1.5)


