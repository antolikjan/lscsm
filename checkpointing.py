import numpy as np
from datetime import datetime
from pprint import pformat
from struct import pack, unpack
from LSCSM import LSCSM   


def saveMat(mat,filename,filepath=''):
    """    
    Saves an array/matrix in binary format
    """    
    mat=np.reshape(np.array(mat),-1)
    with open(filepath+filename,'wb') as f:
        f.write(pack('<%dd'%len(mat),*mat))
        

def saveResults(lscsm,suppl_params,K=None,errors=None,corr=None,prefix='lscsmfit',suffix='',outpath=''): 
    """
    Saves lscsm fitting results and meta-parameters
    """
    
    filenames=[]
    if len(suffix)>0:
        if suffix[0]!='_':
            suffix = '_'+suffix
   
    # Save K (lscsm parameters)   
    if not(K in [None,[]]):
        saveMat(K,prefix+'_K'+suffix,filepath=outpath)
        filenames.append(outpath+prefix+'_K'+suffix)
    
    # Save meta-parameters
    meta_params = dict(lscsm.get_param_values())
    suppl_params['stimsize']=lscsm.size
    suppl_params['date']=str(datetime.now().date())+'-'+str(datetime.now().time())[:8]
    with open(outpath+prefix+'_metaparams'+suffix,'w') as f:
        f.write('meta_params = {\n'+pformat(meta_params)[1:]+'\n\n')
        f.write('suppl_params = {\n'+pformat(suppl_params)[1:]+'\n\n')
    filenames.append(outpath+prefix+'_metaparams'+suffix)
    
    # Save successive values of training and validation error
    if not(errors in [None,[],[None,None],[[],[]]]):
        assert len(errors)==2
        saveMat(errors,prefix+'_error'+suffix,filepath=outpath)
        filenames.append(outpath+prefix+'_error'+suffix)
    
    # Save successive values of training and validation correlation    
    if not(corr in [None,[],[None,None],[[],[]]]):
        assert len(corr)==2
        saveMat(corr,prefix+'_corr'+suffix,filepath=outpath)
        filenames.append(outpath+prefix+'_corr'+suffix)
        
    return filenames         


def loadVec(filename,filepath=''):
    """
    Load an array/matrix stored in binary format
    """     
    with open(filepath+filename,'rb') as f:
        binary=f.read()
    assert len(binary)%8==0
    return np.array(unpack('<%dd'%(len(binary)/8),binary))
    
    
def loadParams(filename,filepath=''):
    """
    Load meta-parameters
    """    
    with open(filepath+filename,'r') as f:
        exec(f.read())
    return meta_params,suppl_params
    

def loadErrCorr(filename,filepath=''):
    """
    Load successive values of training and validation error/correlation
    """
    vec=loadVec(filename,filepath=filepath)
    assert len(vec)%2==0
    return vec[:len(vec)/2], vec[len(vec)/2:]
    
    
def restoreLSCSM(training_inputs,training_set,paramfile,Kfile=None,resultspath='',parampath=''):
    """
    Uses a meta-parameter file and stim+response data to recreate a lcscm object identical to the one used for a previous fit
    (with or whithout setting initial parameters K to the final parameter values obtained for this previous fit, depending on whether a parameter file (Kfile) is provided or not)   
    Returns the lscsm object, the initial parameter values and the data (stimuli and responses)
    """
    
    meta_params, suppl_params = loadParams(paramfile,filepath=parampath)
    lscsm=LSCSM(training_inputs,training_set,**meta_params) 
    if Kfile==None :
        if suppl_params.has_key('seed'):
            K = lscsm.create_random_parametrization(suppl_params['seed'])
        else:
            print "Seed value hasn't been stored in suppl_params object. Using seed=0"
            K = lscsm.create_random_parametrization(0)
    else:
        K=loadVec(Kfile,filepath=resultspath)
    
    return lscsm, K, suppl_params