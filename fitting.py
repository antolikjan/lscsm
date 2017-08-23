import numpy as np
from scipy.optimize import fmin_tnc
from visualization import computeCorr
from traceback import format_exc
from os import rename
from checkpointing import saveResults
    
def fitLSCSM(lscsm,Ks,training_inputs,training_set,validation_inputs,validation_set,fit_params={},checkpointName=None,compCorr=False):
    """
    Runs the optimization process (the optimization function is fmin_tnc)
    """
    
    num_pres,num_neurons = np.shape(training_set) 
    func = lscsm.func()
    terr=[]
    verr=[]
    tcorr=[]
    vcorr=[]
    
    if not(fit_params.has_key('numEpochs')):
        fit_params['numEpochs']=100
    if not(fit_params.has_key('epochSize')):
        fit_params['epochSize']=10000
        
    # If the number of iterations reached during a previous run (leading to the current value of Ks) hasn't been indicated, set it to 0.
    if not(fit_params.has_key('n_rep')):# & checkpointName<>None:
        fit_params['n_rep']=0

    try:
        print "Starting fitting"  
        for i in xrange(0,fit_params['numEpochs']):
            
            if i == 0:
                terr.append(func(np.array(Ks))/num_neurons/len(training_set))
                lscsm.X.set_value(validation_inputs.astype(lscsm.X.dtype))
                lscsm.Y.set_value(validation_set.astype(lscsm.X.dtype))
                verr.append(func(np.array(Ks))/num_neurons/len(validation_set))
                lscsm.X.set_value(training_inputs.astype(lscsm.X.dtype))
                lscsm.Y.set_value(training_set.astype(lscsm.X.dtype))
                print "Before training: ", i, "train error: ",  terr[-1], "val error: ", verr[-1]
                

            # Do epochSize steps of optimization
            #(Ks,success,c)=fmin_tnc(func ,Ks,fprime=lscsm.der(),bounds=lscsm.bounds,maxfun=fit_params['epochSize'],messages=0) 
            (Ks,success,c)=fmin_tnc(func ,Ks,fprime=lscsm.der(),maxfun=fit_params['epochSize'],messages=0) 
            
            # Compute temporary training and validation errors
            terr.append(func(np.array(Ks))/num_neurons/len(training_set))
            lscsm.X.set_value(validation_inputs.astype(lscsm.X.dtype))
            lscsm.Y.set_value(validation_set.astype(lscsm.X.dtype))
            verr.append(func(np.array(Ks))/num_neurons/len(validation_set))
            lscsm.X.set_value(training_inputs.astype(lscsm.X.dtype))
            lscsm.Y.set_value(training_set.astype(lscsm.X.dtype))
            
                        
            if compCorr:
                # Compute temporary training and validation correlations
                vcorr.append(computeCorr(lscsm.response(validation_inputs,Ks),validation_set).mean())
                tcorr.append(computeCorr(lscsm.response(training_inputs,Ks),training_set).mean())              
            
            if checkpointName<>None:
                # Save temporary results into files
                fit_params['n_rep']+=fit_params['epochSize']
                saveResults(lscsm,fit_params,K=Ks,errors=[terr,verr],corr=[tcorr,vcorr],prefix=checkpointName)
             
            # Display temporary error and correlation values 
            if compCorr:
                print "Finished epoch: ", i, "train error: ",  terr[-1], "val error: ", verr[-1], "train corr:", tcorr[-1], "val corr:", vcorr[-1]
            else:
                print "Finished epoch: ", i, "train error: ",  terr[-1], "val error: ", verr[-1]

    except:
        # If there was some error, print its description
        print format_exc()
            
    finally:
        # Even if there was an error, return results:            
        return [Ks,terr,verr,tcorr,vcorr]