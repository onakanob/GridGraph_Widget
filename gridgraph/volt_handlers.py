# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:49:34 2020
Objects to solve solar cell equations
@author: Oliver
"""
import autograd.numpy as np


class Dual_Diode_Handler():
    '''2-diode current, voltage, and wire calculations in a square element.'''
    def __init__(self, params):
        self.params = params
        q = 1.602176634e-19  # elemental charge [C]
        k = 1.380649e-23  # Boltzmann constant [J/K]
        self.q_kT = q / (k * self.params['T'])  # exponential term [C/J = 1/V]

    def local_Jsol(self, V):
        '''Local base-level current density based on overvoltage, using a
        2-diode model for solar current.'''
        return self.params['Jsc'] -\
            self.params['J1'] * np.exp(self.q_kT * V) -\
            self.params['J2'] * np.exp((self.q_kT / 2) * V) -\
            (V / self.params['Rshunt'])

    def I_generated(self, V):
        '''May not need to use sheet loss here: 2-diode appropriately tuned
        would account for sheet recomb. But how does it scale with 'a'?'''
        J = self.local_Jsol(V)
        # sheet_power_loss = ((J**2) * self.params['Rsheet'] *
        #                     self.params['a']**4) / 12
        # return J * (self.params['a'] ** 2) -\
        #     (sheet_power_loss / params['Voc'])

        # Simpler version with no sheet spreading losses
        return J * (self.params['a'] ** 2)

    def I_shadowed(self, I, V):
        '''Amount of current to subtract due to wire shadowing.'''
        return self.local_Jsol(V) * self.params['a'] * self.w(I)

    def volt_drop(self, I):
        if I < 0:
            print('uh - voltage rose?', str(I))
        return I * self.R(I) + 1e-10

    def R(self, I):
        R_wire = self.params['Pwire'] * self.params['a'] /\
            (np.square(self.w(I)) * self.params['h_scale'] + 1e-20)
        return np.minimum(R_wire, self.params['Rsheet'])

    def w(self, I):
        '''Wire scaling rule as a function of I.
        Current version: shadow & power scaling.'''
        return ((2 * np.square(I) * self.params['Pwire']) /
            (self.params['Voc'] * self.params['Jsol'] *
             self.params['h_scale'])) ** (1 / 3.)


if __name__ == '__main__':
    from utils import param_loader
    from matplotlib import pyplot as plt

    print('Debugging voltage-based current handlers.')

    params = param_loader('../recipes/1 cm test.csv')
    params['elements_per_side'] = 100
    params['a'] = params['L'] / params['elements_per_side']

    MAX_V = 0.7

    h = Dual_Diode_Handler(params)

    V = np.arange(0, MAX_V, .01)
    J = h.local_Jsol(V)
    I = h.I_generated(V)
    P = V * J

    plt.plot(V, J)
    plt.plot(V, P)
    plt.xlabel('V')
    plt.ylabel('[A/cm2] / [W/cm2]')
    plt.xlim([0, MAX_V])
    plt.ylim([0, 1.1 * params['Jsc']])
    plt.show()
    plt.close()

    I = np.arange(0, .00005, .0000001)
    V = h.volt_drop(I)
    plt.plot(I, V)
    plt.xlabel('Amps')
    plt.ylabel('Volts')
    plt.title('Voltage drop across an element')
    plt.show()
    plt.close()

    # Try out a full solve cycle with 1 'element'.
