"""
This file contains the implementation of the HSM model.
"""

import numpy
import theano
import param

from theano import tensor as T
from TheanoVisionModel import TheanoVisionModel
            
class HSM(TheanoVisionModel):
      
      num_lgn = param.Integer(default=30,bounds=(0,10000),doc="""Number of lgn units""")
      num_hidden = param.Integer(default=100,bounds=(0,10000),doc="""Number of hidden units""")
      v1of = param.String(default='LogisticLoss',doc="""Transfer function of 'V1' neurons""")
      lgnof = param.String(default='Linear',doc="""Transfer function of 'LGN' neurons""")
      balanced_LGN = param.Boolean(default=False,doc="""Will all LGN cells have balanced strength of center and surround?""")
      LGN_treshold = param.Boolean(default=False,doc="""Will all LGN cells have threshold?""")
      second_layer = param.Boolean(default=True,doc="""Will there be a second layer of neurons?""")
      negative_lgn = param.Boolean(default=True,doc="""Whether LGN neurons can send negative weights to L4""")
      maximum_weight_l1 = param.Number(default=20000,bounds=(0,100000000000000),doc="""Maximum weights in the hidden layer""")
      maximum_weight_l2 = param.Number(default=4,bounds=(0,100000000000000),doc="""Maximum weights in the output layer""")
      lgn_pos_bound = param.Number(default=2,bounds=(0,100),doc="""Minimum distance from LGN RF center to image edge""")
      lgn_size_bounds = param.NumericTuple(default=(1,25),length=2,doc="""LGN RF size bounds""")
      lgn_gain_bounds = param.NumericTuple(default=(0.0,10.0),length=2,doc="""LGN center and surround amplitude bounds""")
      threshold_bounds = param.NumericTuple(default=(-20,20),length=2,doc="""Spiking threshold bounds for all layers""")      
      
      def construct_free_params(self):
          
            # LGN 
            self.lgn_x = self.add_free_param("x_pos",self.num_lgn,(self.lgn_pos_bound,self.size-self.lgn_pos_bound))
            self.lgn_y = self.add_free_param("y_pos",self.num_lgn,(self.lgn_pos_bound,self.size-self.lgn_pos_bound))
            self.lgn_sc = self.add_free_param("size_center",self.num_lgn,self.lgn_size_bounds)
            self.lgn_ss = self.add_free_param("size_surround",self.num_lgn,self.lgn_size_bounds)
            
            if not self.balanced_LGN:
               self.lgn_rc = self.add_free_param("center_weight",self.num_lgn,self.lgn_gain_bounds)
               self.lgn_rs = self.add_free_param("surround_weight",self.num_lgn,self.lgn_gain_bounds)

            if self.LGN_treshold:
                self.lgn_t = self.add_free_param("lgn_threshold",self.num_lgn,self.threshold_bounds)
                
            # V1
            if self.negative_lgn:
               minw = -self.maximum_weight_l1
            else:
               minw = 0

            if self.second_layer:
               self.hidden_w = self.add_free_param("hidden_weights",(self.num_lgn,self.num_hidden),(minw,self.maximum_weight_l1))
               self.hl_tresh = self.add_free_param("hidden_layer_threshold",self.num_hidden,self.threshold_bounds)
               self.output_w = self.add_free_param("output_weights",(self.num_hidden,self.num_neurons),(-self.maximum_weight_l2,self.maximum_weight_l2))
               self.ol_tresh = self.add_free_param("output_layer_threshold",self.num_neurons,self.threshold_bounds)
            else:
               self.output_w = self.add_free_param("output_weights",(self.num_lgn,self.num_neurons),(minw,self.maximum_weight_l1))
               self.ol_tresh = self.add_free_param("output_layer_threshold",self.num_neurons,self.threshold_bounds)


      def construct_model(self):
            # construct the 'retinal' x and y coordinates matrices
            # = list of x-coordinates and list of y-coordinates for each position in the grid
            xx = theano.shared(numpy.repeat([numpy.arange(0,self.size,1,dtype=theano.config.floatX)],self.size,axis=0).T.flatten())   
            yy = theano.shared(numpy.repeat([numpy.arange(0,self.size,1,dtype=theano.config.floatX)],self.size,axis=0).flatten())
            
            if self.balanced_LGN:
                lgn_kernel = lambda i,x,y,sc,ss: T.dot(self.X,(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi)) - (T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/ss[i]).T/ (2*ss[i]*numpy.pi)))
                lgn_output,updates = theano.scan(lgn_kernel , sequences= T.arange(self.num_lgn), non_sequences=[self.lgn_x,self.lgn_y,self.lgn_sc,self.lgn_ss])
            else:
                # NOT WE MADE SURROND TO BE SC+SS
                lgn_kernel = lambda i,x,y,sc,ss,rc,rs: T.dot(self.X,rc[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi)) - rs[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/(sc[i]+ss[i])).T/ (2*(sc[i]+ss[i])*numpy.pi)))
                lgn_output,updates = theano.scan(lgn_kernel,sequences=T.arange(self.num_lgn),non_sequences=[self.lgn_x,self.lgn_y,self.lgn_sc,self.lgn_ss,self.lgn_rc,self.lgn_rs])
            
            lgn_output = lgn_output.T
            
            if self.LGN_treshold:
               lgn_output = lgn_output - self.lgn_t.T
               
            lgn_output = self.construct_of(lgn_output,self.lgnof)
            
            if self.second_layer:
                  output = T.dot(lgn_output,self.hidden_w)
                  
                  model_output = self.construct_of(output-self.hl_tresh,self.v1of)
                  model_output = self.construct_of(T.dot(model_output , self.output_w) - self.ol_tresh,self.v1of)
            else:
                  output = T.dot(lgn_output,self.output_w)
                  model_output = self.construct_of(output-self.ol_tresh,self.v1of)
            
            self.model_output = model_output
            
            return model_output
            
          
          

        


class InverseHSM(TheanoVisionModel):
      
      num_lgn = param.Integer(default=30,bounds=(0,10000),doc="""Number of lgn units""")
      num_hidden = param.Integer(default=100,bounds=(0,10000),doc="""Number of hidden units""")
      v1of = param.String(default='LogisticLoss',doc="""Transfer function of 'V1' neurons""")
      negative_lgn = param.Boolean(default=True,doc="""Whether LGN neurons can send negative weights to L4""")
      maximum_weight_l1 = param.Number(default=200,bounds=(0,100000000000000),doc="""Maximum weights in the hidden layer""")
      maximum_weight_l2 = param.Number(default=4,bounds=(0,100000000000000),doc="""Maximum weights in the output layer""")
      lgn_pos_bound = param.Number(default=5,bounds=(0,100),doc="""Minimum distance from LGN RF center to image edge""")
      lgn_size_bounds = param.NumericTuple(default=(1,7),length=2,doc="""LGN RF size bounds""")
      lgn_gain_bounds = param.NumericTuple(default=(0.0,10.0),length=2,doc="""LGN center and surround amplitude bounds""")
      threshold_bounds = param.NumericTuple(default=(-20,20),length=2,doc="""Spiking threshold bounds for all layers""")      
      
      def construct_free_params(self):
          
            # LGN 
            self.lgn_x = self.add_free_param("x_pos",self.num_lgn,(self.lgn_pos_bound,self.size-self.lgn_pos_bound))
            self.lgn_y = self.add_free_param("y_pos",self.num_lgn,(self.lgn_pos_bound,self.size-self.lgn_pos_bound))
            self.lgn_sc = self.add_free_param("size_center",self.num_lgn,self.lgn_size_bounds)
            #self.lgn_ss = self.add_free_param("size_surround",self.num_lgn,self.lgn_size_bounds)
            
            self.lgn_rc = self.add_free_param("center_weight",self.num_lgn,self.lgn_gain_bounds)
            #self.lgn_rs = self.add_free_param("surround_weight",self.num_lgn,self.lgn_gain_bounds)
                
            # V1
            if self.negative_lgn:
               minw = -self.maximum_weight_l1
            else:
               minw = 0

            #self.first_layer_wieghts = self.add_free_param("hidden_weights",(self.kernel_size,self.num_lgn),(minw,self.maximum_weight_l1))
            #self.first_layer_thresh = self.add_free_param("hidden_layer_threshold",self.num_lgn,self.threshold_bounds)

            self.first_layer_wieghts = self.add_free_param("hidden_weights",(self.kernel_size,self.num_hidden),(minw,self.maximum_weight_l1))
            self.first_layer_thresh = self.add_free_param("hidden_layer_threshold",self.num_hidden,self.threshold_bounds)
            self.second_layer_wieghts = self.add_free_param("output_weights",(self.num_hidden,self.num_lgn),(-self.maximum_weight_l2,self.maximum_weight_l2))
            self.second_layer_thresh = self.add_free_param("output_layer_threshold",self.num_lgn,self.threshold_bounds)
            
            #self.second_layer_wieghts = self.add_free_param("output_weights",(self.num_hidden,self.num_neurons))#,(-self.maximum_weight_l2,self.maximum_weight_l2))
            #self.second_layer_thresh = self.add_free_param("output_layer_threshold",self.num_neurons)#,self.threshold_bounds)


      def construct_model(self):
            # construct the 'retinal' x and y coordinates matrices
            # list of x-coordinates and list of y-coordinates for each position in the grid
            
            layer1_output = T.dot(self.X,self.first_layer_wieghts)   
            layer1_output = self.construct_of(layer1_output - self.first_layer_thresh,self.v1of)

            layer2_output = T.dot(layer1_output,self.second_layer_wieghts)   
            layer2_output = self.construct_of(layer2_output - self.second_layer_thresh,self.v1of)

            xx = theano.shared(numpy.repeat([numpy.arange(0,numpy.sqrt(self.num_neurons),1,dtype=theano.config.floatX)],numpy.sqrt(self.num_neurons),axis=0).T.flatten())   
            yy = theano.shared(numpy.repeat([numpy.arange(0,numpy.sqrt(self.num_neurons),1,dtype=theano.config.floatX)],numpy.sqrt(self.num_neurons),axis=0).flatten())

            #lgn_kernel = lambda i,x,y,sc,ss,rc,rs,layer_output: T.outer(layer_output[:,i],rc[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i])/ (2*sc[i]*numpy.pi)) - rs[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/(sc[i]+ss[i]))/ (2*(sc[i]+ss[i])*numpy.pi)))
            #lgn_output,updates = theano.scan(lgn_kernel,sequences=T.arange(self.num_lgn),non_sequences=[self.lgn_x,self.lgn_y,self.lgn_sc,self.lgn_ss,self.lgn_rc,self.lgn_rs,layer2_output])            
            
            lgn_kernel = lambda i,x,y,sc,rc,layer_output: T.outer(layer_output[:,i],rc[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i])/ (2*sc[i]*numpy.pi)))
            lgn_output,updates = theano.scan(lgn_kernel,sequences=T.arange(self.num_lgn),non_sequences=[self.lgn_x,self.lgn_y,self.lgn_sc,self.lgn_rc,layer2_output])            

            model_output = T.sum(lgn_output,axis=0)

            self.model_output = model_output
            return self.model_output
            
          
          

        
