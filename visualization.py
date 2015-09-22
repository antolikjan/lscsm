from scipy.stats import pearsonr
import numpy as np


def computeCorr(pred_act,responses):
    """
    Compute correlation between predicted and recorded activity for each cell
    """

    num_pres,num_neurons = np.shape(responses)
    corr=np.zeros(num_neurons)
    
    for i in xrange(0,num_neurons):
        if np.all(pred_act[:,i]==0) & np.all(responses[:,i]==0):
            corr[i]=1.
        elif not(np.all(pred_act[:,i]==0) | np.all(responses[:,i]==0)):
            # /!\ To prevent errors due to very low values during computation of correlation
            if abs(pred_act[:,i]).max()<1:
                pred_act[:,i]=pred_act[:,i]/abs(pred_act[:,i]).max()
            if abs(responses[:,i]).max()<1:
                responses[:,i]=responses[:,i]/abs(responses[:,i]).max()    
            corr[i]=pearsonr(np.array(responses)[:,i].flatten(),np.array(pred_act)[:,i].flatten())[0]
            
    return corr


def printCorrelationAnalysis(act,val_act,pred_act,pred_val_act):
    """
    This function simply calculates the correlation between the predicted and 
    and measured responses for the training and validation set and prints them out.
    """
    train_c=computeCorr(pred_act,act)
    val_c=computeCorr(pred_val_act,val_act)
    
    print 'Correlation Coefficients (training/validation): ' + str(np.mean(train_c)) + '/' + str(np.mean(val_c))
    return (train_c,val_c)

