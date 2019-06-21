'''Test script for development of solar gridgraph'''

from solargrid import solar_grid

rows = 3
cols = 3

parameters = {'Voc': 1,         # Volts
              'Jsc': 1,      # A/cm^2
              'Pwire': 10**-5,  # Ohm-cm
              'Psheet': 0.1,    # Ohm/square
              'element_size': 1,  # cm element size
              'w_min': 1e-4}         # cm smallest wire thickness
# parameters = {'Voc': 1,         # Volts
#               'Jsc': 0.02,      # A/cm^2
#               'Pwire': 10**-5,  # Ohm-cm
#               'Psheet': 100,    # Ohm/square
#               'element_size': 200e-4,  # cm element size
#               'w_min': 1e-4}         # cm smallest wire thickness

cell_area = rows * cols * parameters['element_size']  # cm**2

model = solar_grid((rows, cols), parameters)

ideal_power = parameters['Jsc'] * cell_area * parameters['Voc']
power = model.power()
