import numpy as np
from scipy.optimize import fmin_tnc
from visualization import computeCorr
from traceback import format_exc
from os import rename
from checkpointing import saveResults
    

def fitLSCSM(lscsm,Ks,training_inputs,training_set,validation_inputs,validation_set,fit_params={'numEpochs': 100, 'epochSize':1000},checkpointName=None,savecorr=False,savepath=''):
    """
    Runs the optimization process (the optimization function is fmin_tnc)
    """
    
    num_pres,num_neurons = np.shape(training_set) 
    func = lscsm.func()
    terr=[]
    verr=[]
    tcorr=[]
    vcorr=[]        
    filenames=[]

    try:
        print "Starting fitting"  
        for i in xrange(0,fit_params['numEpochs']):
            # Do epochSize steps of optimization
            (Ks,success,c)=fmin_tnc(func ,Ks,fprime=lscsm.der(),bounds=lscsm.bounds,maxfun=fit_params['epochSize'],messages=0) 
            
            # Compute temporary training and validation errors
            terr.append(func(np.array(Ks))/num_neurons/len(training_set))
            lscsm.X.set_value(validation_inputs)
            lscsm.Y.set_value(validation_set)
            verr.append(func(np.array(Ks))/num_neurons/len(validation_set))
            lscsm.X.set_value(training_inputs)
            lscsm.Y.set_value(training_set)
            
            fit_params['n_rep'] = (i+1)*fit_params['epochSize']
            
            if savecorr:
                # Compute temporary training and validation correlations
                vcorr.append(computeCorr(lscsm.response(validation_inputs,Ks),validation_set).mean())
                tcorr.append(computeCorr(lscsm.response(training_inputs,Ks),training_set).mean())              
            
            if checkpointName<>None:
                # Save temporary results into files
                filenames=saveResults(lscsm,fit_params,K=Ks,errors=[terr,verr],corr=[tcorr,vcorr],prefix=checkpointName,suffix='_temp',outpath=savepath)
             
            # Display temporary error and correlation values 
            if savecorr:
                print "Finished epoch: ", i, "train error: ",  terr[-1], "val error: ", verr[-1], "train corr:", tcorr[-1], "val corr:", vcorr[-1]
            else:
                print "Finished epoch: ", i, "train error: ",  terr[-1], "val error: ", verr[-1]
        
        # Remove "_temp" tag from final files        
        for fn in filenames:
             rename(fn,fn[:-5])

    except:
        # If there was some error, print its description
        print format_exc()
            
    finally:
        # Even if there was an error, return results:            
        return [Ks,terr,verr,tcorr,vcorr]