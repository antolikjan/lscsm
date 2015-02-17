import numpy
import numpy.linalg

def STA_LR(training_inputs,training_set, laplace_bias):
    """
    This function implements the spike triggered averaging with laplacian regularization.
    It takes the training inputs and responses, and returns the estimated kernels.
    """
    kernel_size =  numpy.shape(training_inputs)[1]
    laplace = laplaceBias(int(numpy.sqrt(kernel_size)),int(numpy.sqrt(kernel_size)))
    return numpy.linalg.pinv(training_inputs.T*training_inputs + laplace_bias*laplace) * training_inputs.T * training_set
    

def laplaceBias(sizex,sizey):
	S = numpy.zeros((sizex*sizey,sizex*sizey))
	for x in xrange(0,sizex):
		for y in xrange(0,sizey):
			norm = numpy.mat(numpy.zeros((sizex,sizey)))
			norm[x,y]=4
			if x > 0:
				norm[x-1,y]=-1
			if x < sizex-1:
				norm[x+1,y]=-1   
			if y > 0:
				norm[x,y-1]=-1
			if y < sizey-1:
				norm[x,y+1]=-1
			S[x*sizex+y,:] = norm.flatten()
	S=numpy.mat(S)
        return S*S.T
