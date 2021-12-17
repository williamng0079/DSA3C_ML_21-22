import numpy as np

# this code is made to demontrated mutation procedure in report

def Mutation_Demo(x):
    mutation_chance_per_entry = 0.01
            
    mutation_mask = np.random.rand(x.shape[0], x.shape[1])     
    mutation_mask = np.where(mutation_mask < mutation_chance_per_entry, (np.random.rand() - 0.5)/2, 0)
    #updated_array = x + mutation_mask


    return mutation_mask

example_array = np.ones((5, 5))
print(example_array)

print("\n", Mutation_Demo(example_array))