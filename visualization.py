import scipy.stats
import numpy

def printCorrelationAnalysis(act,val_act,pred_act,pred_val_act):
    """
    This function simply calculates the correlation between the predicted and 
    and measured responses for the training and validation set and prints them out.
    """
    num_pres,num_neurons = numpy.shape(act)
    import scipy.stats
    train_c=[]
    val_c=[]
    
    for i in xrange(0,num_neurons):
        train_c.append(scipy.stats.pearsonr(numpy.array(act)[:,i].flatten(),numpy.array(pred_act)[:,i].flatten())[0])
        val_c.append(scipy.stats.pearsonr(numpy.array(val_act)[:,i].flatten(),numpy.array(pred_val_act)[:,i].flatten())[0])
    
    print 'Correlation Coefficients (training/validation): ' + str(numpy.mean(train_c)) + '/' + str(numpy.mean(val_c))
    return (train_c,val_c)

