import __main__
import numpy
import pylab

def sortOutLoading(data_set_name):

    params={}   
    params["normalize_inputs"] = __main__.__dict__.get('NormalizeInputs',False)
    params["normalize_activities"] = __main__.__dict__.get('NormalizeActivities',False)
    params["validation_set_fraction"] = __main__.__dict__.get('ValidationSetFraction',50)               
    params["cut_out"] = __main__.__dict__.get('CutOut',True)        
    params["spiking"] = __main__.__dict__.get('Spiking', True)
    params["2photon"] = __main__.__dict__.get('2photon', True)
    params["density"] = __main__.__dict__.get('density', 0.15)

    custom_index=None
    single_file_input = False



    if data_set_name == '2009_11_04_region3_new':
        #dataset_loc = "/home/jan/projects/lscsm/lscsm_analyis/josh_v/region3_resp3-7_sp.dat"
        #val_dataset_loc = "/home/jan/projects/lscsm/lscsm_analyis/josh_v/region3_val_resp3-7_sp.dat"
        
        dataset_loc = "/home/antolikjan/DATA/CalciumImaging/Flogl/josh_v/region3_resp2-6_sp.dat"
        val_dataset_loc = "/home/antolikjan/DATA/CalciumImaging/Flogl/josh_v/region3_val_resp2-6_sp.dat"
                
        
        params["density"]=0.3
        cut_out_x=0.3
        cut_out_y=0.0
        cut_out_size=1.0

        num_cells = 103    
        sepparate_validation_set = True
        num_rep=1
        num_frames=1
        transpose=False
        average_frames_from=0
        average_frames_to=1
        num_stim=1800
        inputs_offset=1001
        inputs_directory = "/home/antolikjan/DATA/CalciumImaging/Flogl/DataOct2009/20090925_image_list_used/"
        input_match_string = "image_%04d.tif"
        val_inputs_directory = "/home/antolikjan/DATA/CalciumImaging/Flogl/2009_11_04/"
        val_input_match_string = "/20091104_50stimsequence/50stim%04d.tif"
        val_reps = 10

    if data_set_name == '2009_11_04_region3':
        if params["spiking"]:
            dataset_loc = "/home/antolikjan/DATA/CalciumImaging/Flogl/2009_11_04/Raw/region3/spiking_3-7.dat"
            val_dataset_loc = "/home/antolikjan/DATA/CalciumImaging/Flogl/2009_11_04/Raw/region3/val/spiking_3-7.dat"
        else:
            dataset_loc = "/home/antolikjan/DATA/CalciumImaging/Flogl/2009_11_04/Raw/region3/nospiking_3-7.dat"
            val_dataset_loc = "/home/antolikjan/DATA/CalciumImaging/Flogl/2009_11_04/Raw/region3/val/nospiking_3-7.dat"

        params["density"]=0.15
        cut_out_x=0.3
        cut_out_y=0.0
        cut_out_size=1.0

        num_cells = 103    
        sepparate_validation_set = True
        num_rep=1
        num_frames=1
        transpose=False
        average_frames_from=0
        average_frames_to=1
        num_stim=1800
        inputs_offset=1001
        inputs_directory = "/home/antolikjan/DATA/CalciumImaging/Flogl/DataOct2009/20090925_image_list_used/"
        input_match_string = "image_%04d.tif"
        val_inputs_directory = "/home/antolikjan/DATA/CalciumImaging/Flogl/2009_11_04/"
        val_input_match_string = "/20091104_50stimsequence/50stim%04d.tif"
        val_reps = 10

    if data_set_name == '2009_11_04_region5':
        dataset_loc = "/home/jan/projects/lscsm/data/Mice/2009_11_04/Raw/region5/spiking_3-7.dat"
        val_dataset_loc = "/home/jan/projects/lscsm/data/Mice/2009_11_04/Raw/region5/val/spiking_3-7.dat"

        cut_out_x=0.3
        cut_out_y=0.0
        cut_out_size=1.0

        num_cells = 55    
        sepparate_validation_set = True
        num_rep=1
        num_frames=1
        transpose=False
        average_frames_from=0
        average_frames_to=1
        num_stim=1260
        inputs_offset=1001
        inputs_directory = "/home/jan/projects/lscsm/data/Flogl/DataOct2009/20090925_image_list_used/"
        input_match_string = "image_%04d.tif"
        val_inputs_directory = "/home/jan/projects/lscsm/data/Mice/2009_11_04/"
        val_input_match_string = "/20091104_50stimsequence/50stim%04d.tif"
        val_reps = 8

    if data_set_name == '2010_04_22':
        if params["spiking"]:
            dataset_loc = "/home/jan/projects/lscsm/data/Mice/2010_04_22/spiking_3-7.dat"
            val_dataset_loc = "/home/jan/projects/lscsm/data/Mice/2010_04_22/val/spiking_3-7.dat"
        else:
            print 'ERROR: no no-spiking data'
        num_cells = 102    
        sepparate_validation_set = True
        num_rep=1
        num_frames=1
        transpose=False
        average_frames_from=0
        average_frames_to=1
        num_stim=1800
        inputs_offset=0
        cut_out_x=0.1
        cut_out_y=0.0
        cut_out_size=1.0
        inputs_directory = "/home/jan/projects/lscsm/data/Mice/Stimuli/NG/1800/depackaged/"
        input_match_string = "frame%05d.tif"
        val_inputs_directory = "/home/jan/projects/lscsm/data/Mice/Stimuli/NG/1800/depackaged_val/"
        val_input_match_string = "frame%05d.tif"
        val_reps = 12



    if data_set_name == 'DataLee':
       dataset_loc = "/home/jan/projects/lscsm/DataLee/responses/responses.dat" 
       num_cells = 279
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=True
       inputs_offset=0
       params["2photon"]=False
       num_stim=1800
       single_file_input=True
       inputs_directory = "/home/jan/projects/lscsm/DataLee/stimuli/"
       input_match_string = "stimuli.dat"       


    dataset = loadSimpleDataSet(dataset_loc,num_stim,num_cells,num_rep=num_rep,num_frames=num_frames,offset=0,transpose=transpose)
    
    if custom_index != None:
       (index,data) = dataset
       dataset = (custom_index, data)    
    
    dataset = averageRepetitions(dataset)
    
    if not sepparate_validation_set:    
        (validation_data_set,dataset) = splitDataset(dataset,params["validation_set_fraction"])
        validation_set = generateTrainingSet(validation_data_set)
        if single_file_input:
                validation_inputs=generateInputsFromBinaryFile(validation_data_set,inputs_directory,input_match_string,params["density"])
        else:
                validation_inputs=generateInputs(validation_data_set,inputs_directory,input_match_string,params["density"],offset=inputs_offset)

    else:
        valdataset = loadSimpleDataSet(val_dataset_loc,50,num_cells,val_reps)
        (valdataset,trash) = splitDataset(valdataset,params["validation_set_fraction"])
        #get rid of frames and make it into a 3D stack where first dimension is repetitions
        from copy import deepcopy
        (index,raw_val_set) = valdataset
        rr=[]
        for i in xrange(0,val_reps):
            rr.append(generateTrainingSet(averageRepetitions((index,deepcopy(raw_val_set)),reps=[i])))
        raw_val_set = rr
        valdataset = averageRangeFrames(valdataset,0,1)
        valdataset = averageRepetitions(valdataset)
        validation_set = generateTrainingSet(valdataset)
        if single_file_input:
                validation_inputs=generateInputsFromBinaryFil(valdataset,val_inputs_directory,val_input_match_string,params["density"])
        else:
                validation_inputs=generateInputs(valdataset,val_inputs_directory,val_input_match_string,params["density"],offset=0)

    

    training_set = generateTrainingSet(dataset)
    if single_file_input:
        training_inputs=generateInputsFromBinaryFile(dataset,inputs_directory,input_match_string,params["density"])
    else:
        training_inputs=generateInputs(dataset,inputs_directory,input_match_string,params["density"],offset=inputs_offset)

    if params["normalize_inputs"]:
       training_inputs = ((numpy.array(training_inputs) - 128.0)/128.0) * numpy.sqrt(2) 
       validation_inputs = ((numpy.array(validation_inputs) - 128.0)/128.0) * numpy.sqrt(2)

    training_inputs=numpy.array(training_inputs)/1000000.0
    validation_inputs=numpy.array(validation_inputs)/1000000.0

    if params["spiking"] and params["2photon"]:
        training_set = (training_set)/0.028
        validation_set = (validation_set)/0.028

    
    if params["normalize_activities"]:
        m = numpy.mean(training_set)
        means = numpy.mean(training_set,axis=0)
        training_set = numpy.divide(training_set,numpy.reshape(means,(1,num_cells)))*m
        validation_set = numpy.divide(validation_set,numpy.reshape(means,(1,num_cells)))*m
        if sepparate_validation_set:
                for i in xrange(0,val_reps):
                      raw_val_set[i] = numpy.divide(raw_val_set[i],numpy.reshape(means,(1,num_cells)))*m
        

    if params["cut_out"]:
        (x,y)= numpy.shape(training_inputs[0])
        training_inputs = cut_out_images_set(training_inputs,int(x*cut_out_size),(int(x*cut_out_y),int(y*cut_out_x)))
        validation_inputs = cut_out_images_set(validation_inputs,int(x*cut_out_size),(int(x*cut_out_y),int(y*cut_out_x)))

    
    (sizex,sizey) = numpy.shape(training_inputs[0])
    
    training_inputs = generate_raw_training_set(training_inputs)
    validation_inputs = generate_raw_training_set(validation_inputs)
    
    if not sepparate_validation_set:
       raw_val_set = [validation_set]
    
    print "Training set size:"
    print numpy.shape(training_set)
    return (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,raw_val_set)
    
    
def loadSimpleDataSet(filename,num_stimuli,n_cells,num_rep=1,num_frames=1,offset=0,transpose=False):
    f = file(filename, "r") 
    data = [line.split() for line in f]
    if transpose:
       data = numpy.array(data).transpose()         
    
    f.close()
    print "Dataset shape:", numpy.shape(data)

    dataset = [([[] for i in xrange(0,num_stimuli)]) for j in xrange(0,n_cells)]
    
    
    for k in xrange(0,n_cells):
        for i in xrange(0,num_stimuli):
            for rep in xrange(0,num_rep):
                f = []
                for fr in xrange(0,num_frames):
                       f.append(float(data[rep*num_stimuli+i*num_frames+fr+offset][k]))
                dataset[k][i].append(f)
    return (numpy.arange(0,num_stimuli),dataset)
            


 
def averageRepetitions(dataset,reps=None):
    (index,data) = dataset
    (num_cells,num_stim,num_rep,num_frames) = numpy.shape(data)
    if reps==None:
       reps = numpy.arange(0,num_rep,1)

    for cell in data:
        for stimulus in xrange(0,num_stim):
            r = [0 for i in range(0,num_frames)]
            for rep in reps:
                for f in xrange(0,num_frames):
                    r[f]+=cell[stimulus][rep][f]/(len(reps)*1.0)
            
            cell[stimulus]=[r]
    return (index,data)

def splitDataset(dataset,ratio):
    (index,data) = dataset
    (num_cells,num_stim,trash1,trash2) = numpy.shape(data)

    dataset1=[]
    dataset2=[]
    index1=[]
    index2=[]

    if ratio<=1.0:
        tresh = num_stim*ratio
    else:
        tresh = ratio

    for i in xrange(0,num_stim):
        #if i >= num_stim - numpy.ceil(tresh):
        if i <= numpy.floor(tresh):
            index1.append(index[i])
        else:    
            index2.append(index[i])
    
    for cell in data:
        d1=[]
        d2=[]
        for i in xrange(0,num_stim):
            #if i >= num_stim - numpy.ceil(tresh):
            if i <= numpy.floor(tresh):
               d1.append(cell[i])
            else:    
               d2.append(cell[i])
        dataset1.append(d1)
        dataset2.append(d2)
    return ((index1,dataset1),(index2,dataset2))

def generateTrainingSet(dataset):
    (index,data) = dataset

    training_set=[]
    for cell in data:
        cell_set=[]
        for stimuli in cell:
           for rep in stimuli:
                for frame in rep:
                    cell_set.append(frame)
        training_set.append(cell_set)
    return numpy.array(numpy.matrix(training_set).T)

def generateInputsFromBinaryFile(dataset,directory,image_matching_string):
    (index,data) = dataset      
    
    f = file(directory + image_matching_string, "r") 
    data = [numpy.array(line.split()) for line in f]
    f.close()
    ins = []
    for j in index:
        b = data[j]
        z = []
        for i in xrange(0,len(b)):
            z.append(float(b[i]))
        s=numpy.sqrt(len(b))        
        ins.append(numpy.reshape(numpy.array(z),(s,s))) 
    return ins

def generateInputs(dataset,directory,image_matching_string,density,offset):
    (index,data) = dataset
    from PIL import Image
    # ALERT ALERT ALERT We do not handle repetitions yet!!!!!
    image_filenames=[directory+image_matching_string %(i+offset) for i in index]
    ins = []
    for j in xrange(0,len(index)):
        #inputs[j].pattern_sampler.whole_pattern_output_fns=[]
        image = Image.open(image_filenames[j])
        (width,height) = image.size
        inp = image.resize((int(width*density), int(height*density)), Image.ANTIALIAS)
        ins.append(numpy.array(inp.getdata()).reshape( int(height*density),int(width*density)))

    return ins

def generate_raw_training_set(inputs):
    out = []
    for i in inputs:
        out.append(i.flatten())
    return numpy.array(out)

def averageRangeFrames(dataset,min,max):
    (index,data) = dataset

    for cell in data:
        for stimulus in cell:
            for r in xrange(0,len(stimulus)):
                stimulus[r]=[numpy.average(stimulus[r][min:max])]

    return (index,data)
    
def cut_out_images_set(inputs,size,pos):
    (sizex,sizey) = numpy.shape(inputs[0])

    (x,y) = pos
    inp = []
    if (x+size <= sizex) and (y+size <= sizey):
        for i in inputs:
                inp.append(i[x:x+size,y:y+size])
    else:
        print "cut_out_images_set: out of bounds"
    return inp
    
