"""
This file contains the implementation of the LSCSM model.
"""

import numpy
import theano
import param

from theano import tensor as T
from TheanoVisionModel import TheanoVisionModel
            
class LSCSM(TheanoVisionModel):
      
      num_lgn = param.Integer(default=9,bounds=(0,10000),doc="""Number of lgn units""")
      hlsr = param.Number(default=0.2,bounds=(0,1.0),doc="""The hidden layer size ratio""")
      v1of = param.String(default='LogisticLoss',doc="""Transfer function of 'V1' neurons""")
      lgnof = param.String(default='Linear',doc="""Transfer function of 'LGN' neurons""")
      balanced_LGN = param.Boolean(default=False,doc="""Will all LGN cells have balanced strength of center and surround?""")
      LGN_treshold = param.Boolean(default=False,doc="""Will all LGN cells have threshold?""")
      second_layer = param.Boolean(default=True,doc="""Will there be a second layer of neurons?""")
      LL = param.Boolean(default=True,doc="""Whether to use Log-Likelyhood. False will use MSE.""")
      negative_lgn = param.Boolean(default=True,doc="""Whether LGN neurons can send negative weights to L4""")
      maximum_weight_l1 = param.Number(default=20000,bounds=(0,100000000000000),doc="""Maximum weights in the hidden layer""")
      maximum_weight_l2 = param.Number(default=4,bounds=(0,100000000000000),doc="""Maximum weights in the output layer""")
      
      def construct_free_params(self):
          
            # LGN 
            self.lgn_x = self.add_free_param("x_pos",self.num_lgn,(6,self.size-6))
            self.lgn_y = self.add_free_param("y_pos",self.num_lgn,(6,self.size-6))
            self.lgn_sc = self.add_free_param("size_center",self.num_lgn,(1,25))
            self.lgn_ss = self.add_free_param("size_surround",self.num_lgn,(1,25))
            
            if not self.balanced_LGN:
               self.lgn_rc = self.add_free_param("center_weight",self.num_lgn,(0.0,10.0))
               self.lgn_rs = self.add_free_param("surround_weight",self.num_lgn,(0.0,10.0))

            if self.LGN_treshold:
                self.lgn_t = self.add_free_param("lgn_threshold",self.num_lgn,(0,20))
                
            # V1
            if self.negative_lgn:
               minw = -self.maximum_weight_l1
            else:
               minw = 0

            if self.second_layer:
               self.hidden_w = self.add_free_param("hidden_weights",(self.num_lgn,int(self.num_neurons*self.hlsr)),(minw,self.maximum_weight_l1))
               self.hl_tresh = self.add_free_param("hidden_layer_threshold",int(self.num_neurons*self.hlsr),(0,20))
               self.output_w = self.add_free_param("output_weights",(int(self.num_neurons*self.hlsr),self.num_neurons),(-self.maximum_weight_l2,self.maximum_weight_l2))
               self.ol_tresh = self.add_free_param("output_layer_threshold",int(self.num_neurons),(0,20))
            else:
               self.output_w = self.add_free_param("output_weights",(self.num_lgn,self.num_neurons),(minw,self.maximum_weight_l1))
               self.ol_tresh = self.add_free_param("output_layer_threshold",self.num_neurons,(0,20))


      def construct_model(self):
            # construct the 'retinal' x and y coordinates matrices
            xx = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten())   
            yy = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten())
            
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
            
          
          

        
