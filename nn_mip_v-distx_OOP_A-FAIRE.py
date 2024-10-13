"""nn_mip_v-distx.py
In SCIP 
MIP problems to represent a neural network L layers . 

Training is realized via minimzing distance between x inputs and x outputs of the previous layer. 
for a subset of layers. 

"""

from pyscipopt import Model
from pyscipopt.scip import quicksum

# we assume have function that return the dataset
# TODO: a pytorch-style dataloader 
dataset = [()] # each data point is a tuple (x_data[d], label[d])


class NN_MIP(Model):

    def __init__(self, L, Nl= [1, 1], l_subset = [], **kwargs):
        super().__init__(**kwargs)
        
        # Set hyperparameters of the NN 
        self.L = L # umber of layer l = 0..L-1, output is x[L], r[L]
        self.Nl = Nl # list
        self.l_subset=l_subset
        self.Lw = {'W':0, 'x':0 }                   # Lower bounds of x variables
        self.Up = {'W':2* max(Nl), 'x':2* max(Nl)}  # Upper bounds of W variables
        self.D = 1
        self.dataset = list() # initialize to  dummy values
        
        
    def _construct_NN(self, dataset):    
        self.dataset = dataset 
        self.D = max(1, len(dataset)) # size of the dataset (or of the batch)
        self._def_variables()
        self._def_constraints(dataset)    # Dataset incorporated at the NN construction
        self._def_objective()

    def _def_variables(self):
        D = self.D
        L, Nl = self.L, self.Nl
        Uw, Ux = self.Up['W'],  self.Up['x']
        Lw, Lx = self.Lw['W'],  self.Lw['x']
        l_subset = self.l_subset

        # Define the variables 
        self.x = [[[self.addVar(lb=Lx, lu=Ux, vtype="C") for i in range(Nl) ] 
                for l in range(L) if l not in l_subset]
                for d in range(D)]

        #For l in l_subset: we define distx and define (x , x_out) instead of x
        # where x_out is the relu output.
        self.distx = [{sign: [[self.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ]
                        for l in l_subset] 
                        for sign in ['+', '-'] }
                        for d in range(D)]
        self.x_in = [[[self.addVar(lb=Lx, lu=Ux, vtype="C") for i in range(Nl) ] 
                for l in l_subset ]
                for d in range(D)]
        self.x_out = [[[self.addVar(lb=Lx, lu=Ux, vtype="C") for i in range(Nl) ] 
                for l in l_subset ]
                for d in range(D)]


        self.khi = [[[self.addVar(lb=Lx, lu=Ux, vtype="C") for i in range(Nl) ] 
                for l in range(L)]
                for d in range(D)]

        self.W = { sign: [[[self.addVar(lb=Lw, lu=Uw, vtype="C") for j in range(Nl[l+1]) ] 
                for i in range(Nl[l]) ]
                for l in range(L)]
                for sign in ['+', '-'] 
            }


        self.b = {sign: [[self.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ]
                        for l in range(L)] 
                        for sign in ['+', '-'] 
            }

        self.omega = { sign: [[[self.addVar(lb=Lw, lu=Uw, vtype="C") for j in range(Nl[l+1]) ] 
                for i in range(Nl[l]) ]
                for l in range(L)]
                for sign in ['+', '-'] 
            }

        self.z = [{ sign: [[[self.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ] 
                for i in range(Nl[l]) ]
                for l in range(L)]
                for sign in ['+', '-'] }
                for d in range(D)]
            

        self.y = [{sign: [[self.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ]
                for l in range(L)] 
                for sign in ['+', '-'] }
                for d in range(D)]


        # Binary var for relu activation:  (also called 'n')
        self.r = [[[self.addVar( vtype="B") for j in range(Nl[l+1]) ]
                for l in range(L)]
                for d in range(D)]


        # Binary variables for quadratic (bilinear) term activation (also called m) 
        self.q = [{ sign: [[[self.addVar( vtype="B") for j in range(Nl[l+1]) ] 
                        for i in range(Nl[l]) ]
                        for l in range(L)       ]
                        for sign in ['+', '-'] }
                        for d in range(D)]



    def _get_variables(self, s):
        y = self.y
        z = self.z
        q = self.q 
        W = self.W
        b = self.b
        x = self.x
        x_in = self.x_in
        x_out = self.x_out
        r =self.r
        omega = self.omega 
        khi = self.khi
        distx = self.distx

        if   s == 'quad':        return z, q, W, x, omega, khi, x_in
        elif s == 'linQuadSumm': return y, z             
        elif s == 'relu':        return r, x, x_out, y, b
        elif s == 'output':      return r
        elif s == 'distx':       return x_in, x_out, distx

    # ==================================================================
    # Define the constraints
    # ==================================================================

    # ---------------------------------------------------------------------------
    # Constraints for the quadratic term linearization
    # ---------------------------------------------------------------------------
    def _add_McCornick(self, z, w, x):
        """ Add Constraints for the McCornick Envelope of bilinear term z = w*x """
        Lw, Up  = self.Lw, self.Up
        x_ ={'U':Up['x'], 'L':Lw['x'] }
        w_ ={'U':Up['W'], 'L':Lw['W'] }

        for a in ['U', 'L']:
            for b in ['U', 'L']:
                sig = 1 if a==b else -1
                exp = x_[a] * (w - w_[b] /2) + w_[b] * (x - x_[a] /2) - z  
                self.addCons( sig*exp >= 0 ) 


    def _add_quad_constraint(self, l,i,j, withMcCornick=False, if_x_in =False):
        # The quadratic term z = w*x is equivalent to 2 linear systems: one for x, one for w
        # I specify x because for l in l_subset: x is called x_in

        # variables:
        z, q, W, x, omega, khi, x_in = self._get_variable('quad')
        if if_x_in: x = x_in

        #Hyperparams:
        D  = self.D
        Lw, Up  = self.Lw, self.Up

        
        # ----------------------------------------------------------------------------
        #           WARNING       ACHTUNG    ATTENCION  
        # Strategy 1 provided by GPT4o:  to be verified
        # Exact reformulation of the quadratic constraint z = w*x as MILP feasability:
        # A_quad * v_quad <= b_row
        # ----------------------------------------------------------------------------
        A_quad, b_quad = dict(), dict()
        for sy in ['W','x']: 
            A_quad ={sy:[ [0, -Up[sy],       1,      0],
                          [1,  Lw[sy], -Up[sy],      0],
                          [-1,      0,       0, Lw[sy]],
                          [0,       0,      -1,      1],
                          [0, -Up[sy],       0,      1],
                          [0,  Up[sy],       1,     -1],
                        ]
                    }
            b_quad = {sy:[0, Lw[sy], 0, 0, 0, Up[sy]] }

        
        for d in range(D):
            for sign in ['+', '-']:   
                for sy in ['W','x']:  
                    v_quad = [  z[d][sign][l][i][j], 
                                q[d][sign][l][i][j],
                                (W[sign][l][i][j]     if sy=='W' else   x[d][l][i] ), 
                                (omega[sign][l][i][j] if sy=='W' else khi[d][l][i] )
                            ]
        
                for A_row, b_row in zip(A_quad, b_quad):
                        expr = quicksum([A_row[id] * var for id,var in enumerate(v_quad)]) - b_row 
                        self.addCons( expr <= 0 )
        
                if withMcCornick:
                    self._add_McCornick(z[d][sign][l][i][j], W[sign][l][i][j] , x[d][l][i] ) 
                    
                    


    # ---------------------------------------------------------------------------
    # Constraints for the summation of the linearized quadratic terms
    # ---------------------------------------------------------------------------
    """
    0=y_{l}^{j+} \sum_{i=1}^{N_{l}}} z_{l}^{ij+}

    0=y_{l}^{j-} \sum_{i=1}^{N_{l}}} z_{l}^{ij-}

    """
    def _add_linQuadSumm_constraint(self,  l,j):
 
        y, z = self._get_variable('linQuadSumm')
        Nl = self.Nl
        
        a_linQ = [1] + [-1 for i in range(Nl[l]) ]
        for d in range(self.D):  
            for sign in ['+', '-']:
                v_linQ = [ y[d][sign][l][j]] +  [  z[d][sign][l][i][j] for i in range(Nl[l]) ]           
                expr = quicksum([a_linQ[id] * var for id,var in enumerate(v_linQ)]) 
                self.addCons( expr = 0 )

    # ---------------------------------------------------------------------------
    # Constraints for Relu activation:  A_relu * v_relu <= b_relu
    # --------------------------------------------------------------------------- 

    def _add_relu_constraint(self, l,j, if_x_out = False ):
        # for l in l_subset: we take x = x_out

        r, x, x_out, y, b = self._get_variable('relu')
        if if_x_out: x = x_out
        Nl = self.Nl

        #M constant for Big-M formulation of relu  : M_r > y - b
        M_r = 2*Nl

        A_relu =   [[0,  -1, 1, -1, 1, -1],
                    [M_r, 1, -1, 1, -1, 1],
                    [-M_r, 1, 0, 0,  0, 0]] 
        
        b_relu = [0, M_r, 0 ]
    
        for d in range(self.D):  
                v_relu =      [ r[d][l][j], 
                                x[d][l+1][j],
                                y[d]['+'][l][j], 
                                y[d]['-'][l][j],
                                b['+'][l][j],
                                b['-'][l][j] 
                                ]

                for A_row, b_row in zip(A_relu, b_relu):
                    expr = quicksum([A_row[id] * var for id,var in enumerate(v_relu)]) - b_row 
                    self.addCons( expr <= 0 )


    # ---------------------------------------------------------------------------
    # Hard constraints on the output binary relu activation and data labels
    # ---------------------------------------------------------------------------
    def _add_output_constraint(self, labels ):
        """ Assuming we have r[d][l] = label[d] , d = 0...D-1,  
            We constrain the binary relu variable for layer l, (default l=L ) to be the labels"""
        
        r = self._get_variable('output')
        L, Nl = self.L, self.Nl

        for d, label_d in enumerate(labels):
            for j in range(Nl[L+1]):
                # output of layer L
                self.addCons( r[d][L][j] - label_d[j] == 0)


    # ---------------------------------------------------------------------------
    # Constraints on distance distx between x_in and x_out for l in domL
    # Alternatively:  minimize with def_layer_objective(l) for l in domL
    # ---------------------------------------------------------------------------
    def _add_distx_constraint(self, l,j ):
        """If we distinguidh x_in from x_out instead of posing x_in = x_out = x as above 
        """
        x_in, x_out, distx = self._get_variables('distx') 
        for d in range(self.D):
            # output of layer l for l in  l_subset
            self.addCons( x_in[d][l][j] - x_out[d][l][j] - distx[d]['+'][l][j] + distx[d]['-'][l][j] == 0 )

    #----------------------------------------------------------------------
    def _layer_objective(self, l ):
        assert l in self.l_subset
        distx = self.distx

        # Define the objective function for layer l: sum over j and d indices 
        return sum( [ sum(distx[d]['+'][l]) + sum(distx[d]['-'][l])  
                    for d in range(self.D)]                              
                )
            
    def def_objective(self):
        objective = sum([ self.layer_objective(l) for l in self.l_subset])
        self.setObjective(objective, sense="minimize")

    # ==========================================================================
    # Define the constraints for layer l

    def _add_layer(self, l):    
        
        x = self.x # no x_in, no x_out
        Nl =self.Nl
        assert l not in self.l_subset + [self.L]
        for j in range(Nl[l+1]):
            for i in range(Nl[l]):
                self.add_quad_constraint( l,i,j)   # z[i,j] = w[i,j] * x[i] 
            self.add_linQuadSumm_constraint( l,j)  # y[j] = sum_{i}( z )
            self.add_relu_constraint(l,j )         # x[l+1][j] = relu( y[j] - b[j] )    
            
    
    def _add_distx_layer(self, l):    
        # layer where x_in =/= x_out and distx = abs(x_in[l+1] - x_out[l] )   
        assert l in self.l_subset
        Nl =self.Nl
        has_x_in  = True
        has_x_out = True
        for j in range(Nl[l+1]):
            for i in range(Nl[l]): 
                self.add_quad_constraint( l,i,j, has_x_in)  # z[i,j] = w[i,j] * x_in[i] 
            self.add_linQuadSumm_constraint( l,j)           # y[j] = sum_{i}( z )
            self.add_relu_constraint(l,j, has_x_out )       # x_out[l+1][j] = relu( y[j] - b[j] )  
            self.add_distx_constraint( l,j )                # distx = abs( x_out[l] - x_in[l] ) 
            # distx will be added to the objective function to minimize

    #==============================================================


    def def_constraints(self):

        x_data, labels = self.dataset 
        for l in range(self.L):
            if l==0:                     self._add_layer(l, x_data)
            if l not in self.l_subset:   self._add_layer(l)
            else:                        self._add_distx_layer(l)
        
        #Output from the last layer (i.e. l=L-1) is x[L], r[L]
        self._add_output_constraint(labels )


    def load_data(self):
        # TODO
        self.dataset = [(), (),()]


    # ==========================================================================
    #           Solving the optimization or feasability problem 
    # ==========================================================================
    def set_solver(self):
        # Set the LP solver to use the simplex method
        # How can we adapt the strategy ??
        self.setIntParam("lp/solvertype", 0) # 1: interior point for LP
        # But this is a MILP ? 


    def train_NN(self):
        # Training: Optimize the problem
        self.optimize()


    def validate(self):
        # dont know how to do that !!??
        NotImplemented

    def validate(self):
        NotImplemented


    def print_solution(self):
        # Print the optimal solution
        print("Optimal solution:")
        print(f"W = {self.getVal(self.W)}")
        print(f"b = {self.getVal(self.b)}")  
        print(f"Objective value = {self.getObjVal()}")

if __name__ == '__main__':
 
 
    # ========================================================================
    #   Construct and train the NN model
    # ========================================================================
    
    nn2 = NN_MIP()
    nn2.load_data()
    nn2.set_solver()
    nn2.train_NN()

    nn2.print_solution()