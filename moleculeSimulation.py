#Method that holds the code for the molecule simulations
# 
# Importing useful modules
import numpy as np 
import matplotlib.pyplot as plt
import helper
import scipy
import os 
import PIL 

#The class that contains our cell simulation
class molecule:
    '''
    Object that represents a bunch of molecules in a volume
    Params:
    numberOfmolecules; int; number of molecules that we will create
    diffusionCoefficient: Float: the diffusion coefficient in um^2/s of the molecules 
    ROI, 20: region in um per side that we are interested in
    noise, 0.5,Amplitude of the gaussian white noise
    periodic, False, If the molecules wrap around or not
    '''
    
    def __init__(self,numMolec = 30, diffusionCoefficient = 0.1, ROI = 20,imageResolution = 256, 
    noise =0.5,periodic = False):
        #Setting our coefficients
        self.numMolec = numMolec
        self.diffusionCoefficient = diffusionCoefficient
        self.ROI = ROI                              #Region of interest in um
        self.imageResolution = imageResolution      #Number of pixels in our image
        self.noiseAmp = noise
        self.periodic = periodic

        #Cytoskeleton
        self.jumpProb = 1           #Probability a walker can jump
        self.skeleton = False       #Is there a skeleton there
        self.xSkeleton = []         #X coordinates of the skeleton
        self.ySkeleton = []         #Y coordinates of the skeleton

        #Lipidrafts
        self.lipidRaft = False      #Do we have lipid rafts
        self.lipidRaftCenters = []
        self.lipidRaftRadius = []
        self.lipidRaftJumpProb = []


        #Now for cell positions
        self.xPositions = np.random.rand(numMolec)*ROI
        self.yPositions = np.random.rand(numMolec)*ROI 
        
        #History of cell positions
        self.xPosHist = []
        self.yPosHist =[] 

        #Other 
        self.umToPixel = self.imageResolution/self.ROI
        self.beamRadius = 0.4      #point source function in um, where values fall to 33 percentish, i.e 1 sigma
        


    #Cytoskeleteon confinement
    def cytoskeleteonConfinement(self, numSquares,jumpProb):
        ''' 
       cytoskeletalConfinement: creates a meshwork skeleton for the particles to diffuse within
       ------------------------
       numSquares, int: number of squares across the ROI, i.e. 3 squares would mean 9 total
       jumpProb, float, 0 < 1: the probability that if a walker tries to jump across the cytoskeleton
        it will succeed
        '''
        #Update our internal variables
        self.jumpProb = jumpProb
        self.skeleton = True
        #Now time to make the skeleton
        #Stretches for 2x roi in each direction
        lineLocations = np.linspace(0-2*self.ROI,self.ROI+2*self.ROI,numSquares*5+1)

        self.xSkeleton = lineLocations
        self.ySkeleton = lineLocations.copy()

        return

    #LipidDomain Confinement
    def lipidDomainConfinement(self, radius,location,jumpProbability = 0.1):
        ''' 
        Creates circular lipid domains for confinement with their own diffusion coefficients
        Params:
        -------
        radius, float: size of microdomain in um
        location, (float,float): where the microdomain is, in um
        jumpProbabilities, float, 0,1: the probability that a particle will jump over a membrane
        '''
        self.lipidRaftCenters.append(location)
        self.lipidRaftRadius.append(radius)
        self.lipidRaft = True 
        self.lipidRaftJumpProb.append(jumpProbability)
        return
    
        
    #Export our 'image'
    def imageExport(self):
        ''' 
        export an image of the molecules
        Params:
        -------
        Returns:
        --------
        '''
        #Bringing some stuff from the class instance
        imageResolution = self.imageResolution

        #Load the gaussian kernel 
        #========================
        x = np.linspace(0,self.imageResolution,self.imageResolution); y = x.copy()
        xx,yy = np.meshgrid(x,y)
        kernel = helper.gaussian(1,xx, yy, self.beamRadius,self.umToPixel)      #Gaussian Beam
    
        image = np.zeros((imageResolution,imageResolution))       #our pixels

        noise = np.random.randn(np.shape(image)[0],np.shape(image)[0])
        noise = abs((noise/np.max(noise))*self.noiseAmp)
        
       
        
        #Change it so that it just resolves 9t 
        #now we gotta place our molecule in the x,y field 
        xIntegerPositions = (np.round(self.xPositions*(float(self.imageResolution/self.ROI))))
        yIntegerPositions = (np.round(self.yPositions*(float(self.imageResolution/self.ROI))))
        
        for i,j in zip(xIntegerPositions,yIntegerPositions):
            #Make sure our molecule is actually in our image
            if i < 0 or i > imageResolution-1 or j < 0 or j > imageResolution - 1:
                continue 
            
            image[int(i)][int(j)] = 1      #i.e. we have a molecule there


        #Now we want to convolve it with a gaussian
        imageFT = np.fft.rfft2(image)
        gaussianFT= np.fft.rfft2(kernel)

        returnedImage = np.fft.irfft2(imageFT*gaussianFT) 
        return (returnedImage+noise)
    
    #Diffuse our molecules
    def diffuse(self,timeStepSize):
        ''' 
        diffuses the cells based on a random diffusion process based on fick diffusion
        Params:
        -------
        timeStepSize: how large of a time step is taken (s)
        ''' 
         
        def positionGenerator(diffusionCoefficient,timeStepSize,numSamples):
            '''
            Diffusing
            '''
            d = diffusionCoefficient
            t = timeStepSize
            return 2*np.sqrt(d*t)*scipy.special.erfinv(2*(np.random.uniform(size = numSamples)-0.5))

        #Generating our x jumps
        xOffset = (positionGenerator(self.diffusionCoefficient, timeStepSize, self.numMolec))
        xOffset = np.asarray(xOffset)
        yOffset = (positionGenerator(self.diffusionCoefficient, timeStepSize, self.numMolec))
        yOffset = np.asarray(yOffset)

        #Now time to see if we need to check
        xAccept = np.ones((self.numMolec))       #Intiially start by accepting all
        yAccept = xAccept.copy()

        #If we have cytoskeletal confinements
        if self.skeleton == True:
            #Go through particles- see if any jump over a line
            #If they do jump over a line- see if they can (probability)
            #if True- jump, if False- stay the same
            #Check the x 
            xAccept= xAccept*skeletonJumper(self.xPositions,xOffset,self.xSkeleton,self.jumpProb)
            yAccept = yAccept*skeletonJumper(self.yPositions,yOffset,self.ySkeleton,self.jumpProb)

            



        #If we have lipid rafts
        #For each pair of positions we see if it has entered any lipid domain
        if self.lipidRaft == True:
            for i in range(self.numMolec):
                xPos,yPos = self.xPositions[i],self.yPositions[i]
                xPro,yPro = xPos + xOffset[i], yPos + yOffset[i]
                #Over all our rafts
                for j in range(len(self.lipidRaftCenters)):
                    Accepted = lipidDomainCrosser(self.lipidRaftCenters[j], self.lipidRaftRadius[j], 
                    (xPos,yPos), (xPro,yPro), self.lipidRaftJumpProb[j])
                    xAccept[i] = xAccept[i]*Accepted
                    yAccept[i] = yAccept[i]*Accepted

            

        if self.periodic == True:
            self.xPositions += (xOffset*xAccept)
            self.yPositions += (yOffset*yAccept)
            self.xPositions = self.xPositions%self.ROI
            self.yPositions = self.yPositions%self.ROI
        else:
            self.xPositions += (xOffset*xAccept)
            self.yPositions += (yOffset*yAccept)
        return 


    
    #Return a bunch of diffusion
    def simulate(self,timeSteps,timeStepSize):
        '''
        Simulates our cells
        '''

        #Reset our position history
        self.xPosHist = []
        self.yPosHist = []

        returnedArray = []
        
        for i in range(timeSteps):
            returnedArray.append(self.imageExport())
            self.diffuse(timeStepSize)

            #After diffusion add our positions
            self.xPosHist.append(self.xPositions.copy())
            self.yPosHist.append(self.yPositions.copy())
            

        return returnedArray

#Helper functions
def cytoskeletonCrosser(cytoskeleton,curPos,proPos):
    '''
    Determines whether a given jump is crossing a cytoskeleton in one dimension
    Params:
    cytoskeleton, array[floats]: the positions of the cytoskeleton
    curPos, float: current position of a walker
    proPos, float:  proposed position of a walker
    Returns:
    True if walker has crossed a barrier- false if not
    '''
    preArray = cytoskeleton-curPos
    postArray = cytoskeleton-proPos 
    for i,j in zip(preArray,postArray):
        if i*j < 0:
            return True
    return False
    
def skeletonJumper(positions,offset,skeleton,jumpProb):
    '''
    Checks whether or not a molecule can jump over a cytoskeletal element:
    Params:
    positions, [float]: where the particles are pre jump
    offset, [float]: the proposed offsets of the particles
    skeleton, [float]: the location of the skeletal elements
    jumpProb, float[0,1]: the probability a molecule can jump over a cytoskeletal element
    Returns
    acceptArray: [bool]: the list of whether or not a jump has been accepted
    '''
    acceptArray = np.ones(len(positions))
    #Check the x 
    proposed = positions+offset
    counter = -1
    for i,j in zip(positions,proposed):
        counter += 1
        crossedBool = cytoskeletonCrosser(skeleton,i,j)
        
        if crossedBool == False:
            
            acceptArray[counter] = 1


        elif crossedBool and np.random.random() < jumpProb:
            acceptArray[counter] = 1
        else:
            acceptArray[counter] = 0
    
    return np.array(acceptArray)
#Determines whether or not a walker has crossed a lipid domain
def lipidDomainCrosser(location, radii,currentPosition,proposedPosition,probability):
    '''
    Checks to see if a walker has crossed a lipid domain, one dimensional, so one would \n
    check the x coordinate and the y coordinate seperately.
    Params:
    -------
    - locations: list:floats: a list of the centers of the lipid microdomains
    - radii: list:floats: a list of the radius' of the lipid microdomains
    - currentPosition: (float,float): where the walker currently is
    - proposedPosition: (float,float): where the walker wants to go
    - probability: float, 0<1: probaiblity that the walker will cross the membrane
    Returns:
    - boolean: Whether or not the jump was successful 
    '''
    #Checks to see if a point is in a circle
    def cirleChecker(position,location,radii):
        '''Small checker to see if a point is in a circle'''
        xCent,yCent = location 
        xPos, yPos = position
        if (xPos - xCent)**2 + (yPos - yCent)**2 <= radii**2:
            return True 
        else:
            return False 
    
    #If current and proposed positions are in the domain continue
    curPosBool = cirleChecker(currentPosition, location, radii)
    proPosBool = cirleChecker(proposedPosition, location, radii)
    if  curPosBool and proPosBool or (not curPosBool) and (not proPosBool):
        return True #i.e jump is accepted because of the check
    elif np.random.uniform() < probability:
        return True 
    else: 
        return False  
     



#our testing code 
if __name__ == '__main__' : 
    #Cytoskeleton crosser test
    cytoarray = np.array([1,2,3,4,5])
    prePos = 2.5
    postPos = 2.8 
    print(cytoskeletonCrosser(cytoarray,prePos,postPos))




        
        

    





