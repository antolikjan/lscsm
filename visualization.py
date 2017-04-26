from scipy.stats import pearsonr
import numpy as np


def computeCorr(resp1,resp2):
    """
    Compute correlation between predicted and recorded activity for each cell
    """
    r1=np.array(resp1).copy()
    r2=np.array(resp2).copy()
    num_pres,num_neurons = np.shape(r1)
    corr=np.zeros(num_neurons)
    
    for i in xrange(0,num_neurons):
        if np.all(r1[:,i]==0) & np.all(r2[:,i]==0):
            corr[i]=1.
        elif not(np.all(r1[:,i]==0) | np.all(r2[:,i]==0)):
            # /!\ To prevent errors due to very low values during computation of correlation
            if abs(r1[:,i]).max()<1:
                r1[:,i]=r1[:,i]/abs(r1[:,i]).max()
            if abs(r2[:,i]).max()<1:
                r2[:,i]=r2[:,i]/abs(r2[:,i]).max()    
            corr[i]=pearsonr(np.array(r1)[:,i].flatten(),np.array(r2)[:,i].flatten())[0]
            
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