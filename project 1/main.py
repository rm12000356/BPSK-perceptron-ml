from data_uti import create_data_set_normal
from config import num_of_points
from data_uti import labels_insetion
from perceptron_model import train_perceptron
from perceptron_model import test_training
from plot_util import plot_grapgh

#here the data is created and the targets is determine 
Data = create_data_set_normal()
Data_with_noise = create_data_set_normal(noise_power = 0.6 ,phase_noise = 0)
Data_with_phase = create_data_set_normal(phase_noise = 0.2,noise_power = 0)
Data_with_phase_and_noise = create_data_set_normal(phase_noise = 0.2,noise_power = 0.2)
target = labels_insetion(num_of_points, Data)

# plotting of the data to see the graphs 
plot_grapgh(Data , target)
plot_grapgh(Data_with_noise , target)
plot_grapgh(Data_with_phase , target)
plot_grapgh(Data_with_phase_and_noise , target)

#trainning of the perceptron with the generated data
weights = train_perceptron(Data)
#results after the training
error= test_training(Data ,weights,target)
print('pure=',error)
error1= test_training(Data_with_noise ,weights,target)
print('noice=',error1)
error2= test_training(Data_with_phase_and_noise ,weights,target)
print('phase=',error2)
error3= test_training(Data_with_phase_and_noise ,weights,target)
print('both=',error3)