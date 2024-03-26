# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 21:18:14 2021

@author: ajaj3
"""

import numpy as np
from numpy.linalg import norm

class Cbody(object):
    
    
    def __init__(self, filename, i):
        """
        Method to initialise the celestial class and read in relevent data from
        file.
        """
        filein = open(filename, "r")
        lines = filein.readlines()
        #The file containing the relevant body information is opened
        inc = i            
        #The increment is passed from the Solar class
        self.Vel = np.array([float(lines[1]), float(lines[2])]) 
        self.Pos = np.array([float(lines[3]), float(lines[4])])
        #Here we define our initial position and velocity vectors, note that the
        #velocities were calculated by hand and stored within the files
        self.posList = np.zeros(((inc+1), 2))
        self.posList[0] = self.Pos
        #Here a list is defined to store all of the positions at each increment, with
        #the initial position being defined.
        self.mass =  float(lines[0]) 
        self.colour = str(lines[5])
        #The additional object data is read in from here
        self.a = np.zeros((3,2))
        self.G = 6.67408*(10**(-11))
        #These variables are what allows for the acceleration to be calculated
        filein.close()
        
        
    def iPos(self):
        """ 
        Method to initialise the initial position.
        """
        self.cPos = self.Pos
            
            
    def iAcc(self, m, d):
        """ 
        Method to define the initial accelerations. This is done by calling the Calcacc method,
        with the appropriate mass and distance list. Here the assumption is made that initially
        a(0) ~ a(t-delta_t). 
        """
        self.a[0] = self.Calcacc(m, d)
        self.a[1] = self.a[0]   
            
          
    def Updatepos(self, tStep, k):
        """ 
        Method to calculate and return the position at each timestep. This is done
        using the Beeman method.
        """
        self.Pos = self.Pos + self.Vel*tStep + (1/6) * tStep**2 * (4*self.a[1] - self.a[0])
        #Beeman numerical integration calculation
        self.cPos = self.Pos
        #Sets the current position, allowing for a list of distances to be found
        self.posList[k] = self.Pos
        #The position at the kth increment is added to the position list
        return self.Pos
      
    
    def Calcacc(self, m, d):
        """ 
        Method to calculate and return the next acceleration for each timestep.
        """
        fList = [0]*len(m)
        for i in range(len(m)):
            #This loop creates a list of each force that the celestial body is experiencing
            r = self.cPos - d[i]     
            fList[i] = (self.G * self.mass * m[i])*r/(norm(r)**3)
        a = -sum(fList)/self.mass
        #The sum function is used to find the total force vector, and then acceleration is calculated
        return a
    
    
    def Updatevel(self, tStep, m, d):
        """ 
        Method to calculate the velocity at each timestep.
        """    
        self.a[2] = self.Calcacc(m, d)
        self.Vel = self.Vel + (1/6) * tStep * (2*self.a[2] + 5*self.a[1] - self.a[0])
         
          
    def Updateacc(self):
        """ 
        Method to update the accelerations for each timestep
        """
        self.a[0] = self.a[1]
        self.a[1] = self.a[2]
    
    def returnCpos(self):
        """ 
        Method to return the current position.
        """
        return self.cPos
    
    def OrbPeriod(self):
        """
        Method to calculate the orbital period
        """
        fullOrb = 0
        for k in range(int(1000)):
            if float(self.posList[k][1]) <=0 and float(self.posList[k+1][1]) >= 0:
                #This condition determines if the body has completed a full loop
                orbPeriod = k+1
                fullOrb += 1
        orbPeriod = (orbPeriod/fullOrb)
        #The orbital period is calculated by dividing the last increment number at which
        #the body completed a loop and divides it by the number of loops
        return orbPeriod
        
    def Kenergy(self):
        """
        A method to calculate the kinetic energy at different intervals. This is to check if energy is conserved.
        """
        k = 0.5*self.mass*(norm(self.Vel)**2)
        return k

    
    def Penergy(self, m, d):
        """
        A method to calculate the potential energy at different intervals. This is to check if energy is conserved
        """
        pList = [0] * len(m)
        for i in range(len(d)):
            r = self.cPos - d[i] 
            pList[i] = -0.5 * ((self.G * self.mass * m[i])/norm(r))   
        p = sum(pList)
        return p
        
        
        
        
        
        