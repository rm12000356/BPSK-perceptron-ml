from data_uti import create_data_set_normal
from config import num_of_points
from MultiLayerPerceptron_model import training_multilayer
from MultiLayerPerceptron_model import testing
from data_uti import labels_insetion
from data_uti import data_separation
from plot_util import plot_grapgh

#here the data is created and the targets is determine 
Data = create_data_set_normal()
Data_with_noise = create_data_set_normal(noise_power = 0.2 ,phase_noise = 0)
Data_with_phase = create_data_set_normal(phase_noise = 0.2,noise_power = 0)
Data_with_phase_and_noise = create_data_set_normal(phase_noise = 0.2,noise_power = 0.1)
target = labels_insetion(num_of_points, Data)

# plotting of the data to see the graphs 
plot_grapgh(Data , target)
plot_grapgh(Data_with_noise , target)
#plot_grapgh(Data_with_phase , target)
#plot_grapgh(Data_with_phase_and_noise , target)

#trainning of the perceptron with the generated data
hidden_weight,output_weight,binary_target =training_multilayer(Data,target)

#results after the training
#there is a way to introduce a no rounding, that is at the testing function testing(testing_Data,hidden_weight,output_weight,binary_target,"0 for no rounding"-"1 for rounding(default)")
error= testing(Data,hidden_weight,output_weight,binary_target)
print('pure=',error)
error1= testing(Data_with_noise,hidden_weight,output_weight,binary_target)
print('noice=',error1)
error2= testing(Data_with_phase,hidden_weight,output_weight,binary_target)
print('phase=',error2)
error3= testing(Data_with_phase_and_noise,hidden_weight,output_weight,binary_target)
print('both=',error3)