import __main__
from LSCSM import LSCSM
import numpy
import pylab
from scipy.optimize import fmin_tnc
from visualization import printCorrelationAnalysis
from STA import STA_LR

def fitLSCSM(training_inputs,training_set,validation_inputs,validation_set):
    num_pres,num_neurons = numpy.shape(training_set) 
    
    print "Creating LSCSM model"
    lscsm = LSCSM(training_inputs,training_set)
    print "Created LSCSM model"
    
    # create the theano loss function
    func = lscsm.func()

    print "Starting fitting"
    terr=[]
    verr=[]
    
    # set initial random values of the model parameter vector
    Ks = lscsm.create_random_parametrization(13)
    
    for i in xrange(0,__main__.__dict__.get('NumEpochs',100)):
        (Ks,success,c)=fmin_tnc(func ,Ks,fprime=lscsm.der(),bounds=lscsm.bounds,maxfun = __main__.__dict__.get('EpochSize',1000),messages=0)
        
        terr.append(func(numpy.array(Ks))/num_neurons/len(training_set))
        lscsm.X.set_value(validation_inputs)
        lscsm.Y.set_value(validation_set)
        verr.append(func(numpy.array(Ks))/num_neurons/len(validation_set))
        lscsm.X.set_value(training_inputs)
        lscsm.Y.set_value(training_set)
        
        print "Finnised epoch: ", i, "training error: ",  terr[-1], "validation error: ", verr[-1]
        pylab.plot(verr,'r')
        pylab.plot(terr,'b')
        pylab.draw()
    

    print 'Final training error: ', func(numpy.array(Ks))/num_neurons/len(training_set)
    lscsm.X.set_value(validation_inputs)
    lscsm.Y.set_value(validation_set)
    print 'Final validation error: ', func(numpy.array(Ks))/num_neurons/len(validation_set)
    
    lscsm.printParams(Ks)
    
    return [Ks,lscsm]   
  

def runLSCSM():
    import dataimport
    
    # load data
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,raw_validation_set) = dataimport.sortOutLoading()
    
    # fit LSCSM
    [K,lscsm]=  fitLSCSM(numpy.mat(training_inputs),numpy.mat(training_set),numpy.mat(validation_inputs),numpy.mat(validation_set))
    
    # fir STA with laplacian regularization to compare the data
    rpi = STA_LR(numpy.mat(training_inputs),numpy.mat(training_set),__main__.__dict__.get('RPILaplaceBias',0.0001))
    
    #compute the responses of fitted STA model to validation and training set
    rpi_pred_act = training_inputs * rpi
    rpi_pred_val_act = validation_inputs * rpi
    
    #compute the responses of fitted LSCSM model to validation and training set
    lscsm_pred_act = lscsm.response(training_inputs,K)
    lscsm_pred_val_act = lscsm.response(validation_inputs,K)
    
    
    # print correlations between predicted and measures responses for STA and LSCSM
    print 'STA'
    printCorrelationAnalysis(training_set,validation_set,rpi_pred_act,rpi_pred_val_act)
    
    print 'LSCSM'    
    printCorrelationAnalysis(training_set,validation_set,lscsm_pred_act,lscsm_pred_val_act)
