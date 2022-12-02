# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:39:53 2021

@author: chen_hung
"""


import numpy as np
import matplotlib.pyplot as plt



total_loss = np.load('display_loss.npy')

training_neglogev = total_loss[:,0]
validation_neglogev = total_loss[:,1]

plt.plot(range(len(total_loss)), training_neglogev, 'b-', label='Training_neglogev')
plt.plot(range(len(total_loss)), validation_neglogev, 'g-', label='validation_neglogev')
plt.title('Training & Validation neglogev')
plt.xlabel('Number of epochs')
plt.ylabel('Neglogev')
plt.legend()
plt.show()

