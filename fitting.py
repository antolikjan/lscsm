import numpy as np
from scipy.optimize import fmin_tnc
from scipy.optimize.tnc import RCSTRINGS
from visualization import computeCorr
from traceback import format_exc
from checkpointing import saveResults, loadErrCorr
from sys import stdout    

def fitLSCSM(lscsm,Ks,training_inputs,training_set,validation_inputs,validation_set,fit_params={},checkpointName=None,compCorr=False):
    """
    Runs the optimization process (the optimization function is fmin_tnc)
    """
    
    num_pres,num_neurons = np.shape(training_set) 
    func = lscsm.func()
    der = lscsm.der()
    
    if not(fit_params.has_key('numEpochs')):
        fit_params['numEpochs']=100
    if not(fit_params.has_key('epochSize')):
        fit_params['epochSize']=1000   
        
    # If the number of iterations reached during a previous run (leading to the current value of Ks) hasn't been indicated, set it to 0.
    if not(fit_params.has_key('n_rep')):
        fit_params['n_rep']=0   

    # Compute initial training and validation errors    
    lscsm.X.set_value(validation_inputs.astype(lscsm.X.dtype))
    lscsm.Y.set_value(validation_set.astype(lscsm.Y.dtype))
    verr=[func(np.array(Ks))/num_neurons/len(validation_set)]
    lscsm.X.set_value(training_inputs.astype(lscsm.X.dtype))
    lscsm.Y.set_value(training_set.astype(lscsm.Y.dtype))
    terr=[func(np.array(Ks))/num_neurons/len(training_set)]     
                        
    if compCorr:
        # Compute initial training and validation correlations
        vcorr=[computeCorr(lscsm.response(validation_inputs,Ks),validation_set).mean()]
        tcorr=[computeCorr(lscsm.response(training_inputs,Ks),training_set).mean()]
    else:
        tcorr=vcorr=None
    
    if fit_params['n_rep']>0:
        try:
            terr,verr=loadErrCorr(checkpointName+'_error')
            terr=list(terr)
            verr=list(verr)
            if compCorr:
                tcorr,vcorr=loadErrCorr(checkpointName+'_corr')
                tcorr=list(tcorr)
                vcorr=list(vcorr)
        except:
            pass
            
        
    try:
        print "Starting fitting"  
        for i in xrange(0,fit_params['numEpochs']):
            # Do epochSize steps of optimization
            (Ks,success,c)=fmin_tnc(func ,Ks,fprime=der,bounds=lscsm.bounds,maxfun=fit_params['epochSize'],messages=0) 
            if c!=3:
                print RCSTRINGS[c],
                print 'Number of function evaluations:', str(success)
            # Compute temporary training and validation errors
            terr.append(func(np.array(Ks))/num_neurons/len(training_set))
            lscsm.X.set_value(validation_inputs.astype(lscsm.X.dtype))
            lscsm.Y.set_value(validation_set.astype(lscsm.Y.dtype))
            verr.append(func(np.array(Ks))/num_neurons/len(validation_set))
            lscsm.X.set_value(training_inputs.astype(lscsm.X.dtype))
            lscsm.Y.set_value(training_set.astype(lscsm.Y.dtype))
                                  
            if compCorr:
                # Compute temporary training and validation correlations
                vcorr.append(computeCorr(lscsm.response(validation_inputs,Ks),validation_set).mean())
                tcorr.append(computeCorr(lscsm.response(training_inputs,Ks),training_set).mean())              
            
            if checkpointName<>None:
                # Save temporary results into files
                fit_params['n_rep']+=fit_params['epochSize']
                saveResults(lscsm,fit_params,K=Ks,errors=[terr,verr],corr=[tcorr,vcorr],prefix=checkpointName)
                if verr[-1]<min(verr[:-1]):
                    saveResults(lscsm,fit_params,K=Ks,errors=[terr,verr],corr=[tcorr,vcorr],prefix=checkpointName+'_BEST')
             
            # Display temporary error and correlation values 
            if compCorr:
                print "Finished epoch: ", i, "train error: ",  terr[-1], "val error: ", verr[-1], "train corr:", tcorr[-1], "val corr:", vcorr[-1]
            else:
                print "Finished epoch: ", i, "train error: ",  terr[-1], "val error: ", verr[-1]
            stdout.flush()    

    except:
        # If there was some error, print its description
        print format_exc()
            
    finally:
        # Even if there was an error, return results:            
        return [Ks,terr,verr,tcorr,vcorr]
