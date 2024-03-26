# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:17:27 2021

@author: ajaj3
"""



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#Importing required libraries


class Sim(object):
    
    def __init__(self, filename):
        """
        Method to initialise the class, assinging values to all the required
        variables.
        """
        filein = open(filename, "r")
        lines = filein.readlines()
        self.tStep =  float(lines[0])
        self.inc =  int(lines[1])
       
        
    def Return(self):
        """
        Method to return the number of increments.
        """
        return self.inc , self.tStep
    
    
    def init(self):
        """
        Method to initialise the patches that will be plotted.
        """
        return self.patches
    
    
    def EnergyCon(self, T):
        """
        Method used to check for total energy conservation within the simulation.
        """
        tData = []
        eData = T
        #Defining the lists that store the x-y data
        fileout = open("Tenergy.txt", "w")
        #The file that which will be used to write energies out is opened
        for i in range(len(eData)):  
            tData.append(self.tStep * i)
            fileout.write(str(eData[i]) + " ")    
            #The x data is created and the total energy along each timestep is printed to file
        fileout.close()
        plt.plot( tData, eData)
        plt.title("Total Energy against time")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")
        plt.show
        #A plot of energy against time is created
    
    
    def Periods(self, orbPeriods, eYear):
        """
        Method to write out the orbital Periods to file.
        """
        fileout = open("Periods.txt", "w")
        fileout.write("An Earth year was determined to be " + str(eYear) + " days. \n" )
        fileout.write("Mercury has an orbital period of " + str(orbPeriods[0]) + " Earth years. \n" )
        fileout.write("Venus has an orbital period of " + str(orbPeriods[1]) + " Earth years. \n")
        fileout.write("Earth has an orbital period of " + str(orbPeriods[2]) + " year. \n")
        fileout.write("Mars has an orbital period of " + str(orbPeriods[3]) + " Earth years. \n")
        fileout.close()
    
    
    def updatexy(self, sPos, mePos, vPos, ePos, maPos, saPos):
        """
        Method to assing the full numpy position arrays to instance variables.
        """
        self.sunPos = sPos
        self.merPos = mePos   
        self.venPos = vPos 
        self.earPos = ePos 
        self.marPos = maPos 
        self.satPos = saPos
        
            
    def animate(self, i):
        """
        Method which when called will return the current positions of each planet, 
        allowing for them to be plotted.
        """
        self.patches[0].center = (self.sunPos[i])
        self.patches[1].center = (self.merPos[i])
        self.patches[2].center = (self.venPos[i])
        self.patches[3].center = (self.earPos[i])
        self.patches[4].center = (self.marPos[i])
        self.patches[5].center = (self.satPos[i])
        return self.patches
    
        
    def display(self, sCol, meCol, vCol, eCol, maCol, saCol):
        """
        Method to create the empty plot which will contain the animation. It then
        adds circles representing the celestial objects at their initial locations before
        looping through the animate method determining where to move them at each increment.
        At each incrememnt it moves the objects to their new position.
        """
        fig = plt.figure()
        ax = plt.axes()
        #Creating the place to plot
            
        self.patches = []
        self.patches.append(plt.Circle((self.sunPos[0]), 10**10, color = sCol, animated = True))
        self.patches.append(plt.Circle((self.merPos[0]), 9.5**10, color = meCol, animated = True))
        self.patches.append(plt.Circle((self.venPos[0]), 9.5**10, color = vCol, animated = True))
        self.patches.append(plt.Circle((self.earPos[0]), 9.5**10, color = eCol, animated = True))
        self.patches.append(plt.Circle((self.marPos[0]), 9.5**10, color = maCol, animated = True))
        self.patches.append(plt.Circle((self.satPos[0]), 9.5**10, color = saCol, animated = True))
        #adding the object information to the patch list
        
        for i in range(0, len(self.patches)):
            ax.add_patch(self.patches[i])
        #Creating and adding the patches to the plot
            
        ax.axis('scaled')
        ax.set_xlim(-3*10**11, 3*10**11)
        ax.set_ylim(-3*10**11, 3*10**11)
        #Here we choose the size of the axis, the 'scaled' ensures the planets are circulr
        
        anim = FuncAnimation(fig, self.animate, init_func = self.init, frames = self.inc, repeat = False, interval = 50, blit = True)
        plt.show()
        #These two lines creates and displays the animation on the created plot
    