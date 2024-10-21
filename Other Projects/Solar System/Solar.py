# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:34:49 2021

@author: ajaj3
"""

from Sim import Sim
from Cbody import Cbody


class Solar(object):
    
    def run(self):
        
        
        SolarSim = Sim("OrbitData.txt")    
        inc, tStep = SolarSim.Return()
        #Here we create and instance of the simulation class and receive how many increments
        #shall be looped over, alongside the values of the timestep    
        celest = ["Sun.txt", "Mercury.txt", "Venus.txt", "Earth.txt", "Mars.txt", "Satellite.txt"]
        #A list containing all of the text files we wish to read in
        celes_info = []            
        cMass = []
        cPos = []    
        #defining empty list to store the mass, positions and information of the objects
        TList = [0] * inc
        #defining empty list to store total energy values
        orbPeriods = []
        #defining an empty list to store orbital periods
        
        for i in range(len(celest)):
            #A loop to add the intial values to the three lists defined above
            celes_info.append(Cbody(celest[i], inc))
            cMass.append(celes_info[i].mass)
            cPos.append(celes_info[i].Pos)
            #Each planet has its initial information retrieved from the Cbody class
            #These two create mass and initial position lists within Solar
            celes_info[i].iPos()
            celes_info[i].iAcc((cMass[:i] + cMass[i+1:]), (cPos[:i] + cPos[i+1:]))
            #These two define initial position and acceleration within the instances of Cbody
            #The "list[:i] + list[i+1:]" allows for a list to be passed containing all list information
            #minus the ith data value.
        
        for k in range(inc):
            
            tCel = [0] * len(celes_info)
        
            for j in range(len(celes_info)):
                """
                This loop calculates the r vector, acceleration, velocity and
                position for each object, at each timestep.
                """     
                celes_info[j].Updatepos(tStep, k)
                #Updating the position
                celes_info[j].Updatevel(tStep, (cMass[:j] + cMass[j+1:]), (cPos[:j] + cPos[j+1:]))
                #Updating the velocity, again using the method where we pass a list without that specific object
                #mass and position
                celes_info[j].Updateacc()
                #Updating the acceleration list
                cPos[j] = celes_info[j].returnCpos() 
                #Updating the current position of each body              
                tCel[j] = celes_info[j].Kenergy() + celes_info[j].Penergy((cMass[:j] + cMass[j+1:]), (cPos[:j] + cPos[j+1:])) 
                #The total energy for each object is calculated at each timestep.  
                
            TList[k] = sum(tCel)
            #Calculating the total energy of the system at each increment by summing the energy of each planet.
        
        
        SolarSim.EnergyCon(TList)
        #This method is called to create a plot of the energy against time
        #NOTE: The graph is behind the solar system, move that graph to see it.
        
           
        for j in range(1 , (len(celes_info) - 1)):
            #This loop calculates the orbital periods for the planets, note that
            #the orbital periods of the sun and satellite are not calculated here.
            orbPeriods.append(celes_info[j].OrbPeriod())   
        eYear = orbPeriods[2]
        #Here we define the length of a year on Earth
        for j in range(len(orbPeriods)):
            #This defines the orbital periods in terms of Earth years
            orbPeriods[j]= orbPeriods[j]/eYear
        
        
        SolarSim.Periods(orbPeriods, eYear)
        #Method to write the orbital periods to a file
        
        
        SolarSim.updatexy(celes_info[0].posList, celes_info[1].posList, celes_info[2].posList, celes_info[3].posList, celes_info[4].posList, celes_info[5].posList)
        #This is done to combine the position data for both bodies into a single instance
        #allowing for them both to be plotted at once.
        SolarSim.display(celes_info[0].colour, celes_info[1].colour, celes_info[2].colour, celes_info[3].colour, celes_info[4].colour, celes_info[5].colour)
        #This will display the orbital motion  
             
def main():
    S = Solar() 
    S.run() 
    
main()