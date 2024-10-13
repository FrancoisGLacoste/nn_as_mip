"""nn_mip_v-distx.py
In SCIP 
MIP problems to represent a neural network L layers . 

Training is realized via minimzing distance between x inputs and x outputs of the previous layer. 
for a subset of layers. 

"""

from pyscipopt import Model
from pyscipopt.scip import quicksum

# we assume have function that return the dataset
dataset = [()] # each data point is a tuple (x_data[d], label[d])

# Create a SCIP model
model = Model("NN_MIP")

# Hyperparameters
L = 2
Nl = [1, 1] # i.e. Nl[l]

D = len(dataset)

# Define the constant bounds on variables s x and W_p, W-m
Lw = 0
Uw = 2* max(Nl)

Lx = 0
Ux = 2* max(Nl) 

l_subset = [] 


# ============================================================================

# Define the variables 

x = [[[model.addVar(lb=Lx, lu=Ux, vtype="C") for i in range(Nl) ] 
        for l in range(L) if l not in l_subset]
        for d in range(D)]

#For l in l_subset: we define distx and define (x , x_out) instead of x
# where x_out is the relu output.
distx = [{sign: [[model.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ]
                for l in l_subset] 
                for sign in ['+', '-'] }
                for d in range(D)]
x_in = [[[model.addVar(lb=Lx, lu=Ux, vtype="C") for i in range(Nl) ] 
        for l in l_subset ]
        for d in range(D)]
x_out = [[[model.addVar(lb=Lx, lu=Ux, vtype="C") for i in range(Nl) ] 
        for l in l_subset ]
        for d in range(D)]


khi = [[[model.addVar(lb=Lx, lu=Ux, vtype="C") for i in range(Nl) ] 
        for l in range(L)]
        for d in range(D)]

W = { sign: [[[model.addVar(lb=Lw, lu=Uw, vtype="C") for j in range(Nl[l+1]) ] 
        for i in range(Nl[l]) ]
        for l in range(L)]
        for sign in ['+', '-'] 
    }


b = {sign: [[model.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ]
                for l in range(L)] 
                for sign in ['+', '-'] 
    }

omega = { sign: [[[model.addVar(lb=Lw, lu=Uw, vtype="C") for j in range(Nl[l+1]) ] 
        for i in range(Nl[l]) ]
        for l in range(L)]
        for sign in ['+', '-'] 
    }

z = [{ sign: [[[model.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ] 
        for i in range(Nl[l]) ]
        for l in range(L)]
        for sign in ['+', '-'] }
        for d in range(D)]
    

y = [{sign: [[model.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ]
        for l in range(L)] 
        for sign in ['+', '-'] }
        for d in range(D)]


# Binary var for relu activation:  (also called 'n')
r = [[[model.addVar( vtype="B") for j in range(Nl[l+1]) ]
        for l in range(L)]
        for d in range(D)]

#M constant for Big-M formulation of relu  : M_r > y - b
M_r = 2*Nl

# Binary variables for quadratic (bilinear) term activation (also called m) 
q = [{ sign: [[[model.addVar( vtype="B") for j in range(Nl[l+1]) ] 
                for i in range(Nl[l]) ]
                for l in range(L)       ]
                for sign in ['+', '-'] }
                for d in range(D)]




# ===========================================================================
# Define the constraints
# ===========================================================================

# ---------------------------------------------------------------------------
# Constraints for the quadratic term linearization
# ---------------------------------------------------------------------------
"""


"""


def add_quad_constraint( l,i,j, x = x):
    # The quadratic term z = w*x is equivalent to 2 linear systems: one for x, one for w
    # I specify x because for l in l_subset: x is called x_in

    assert Lw == Lx     
    L = Lw
    assert Uw == Ux     
    U = Uw

    # A_quad * v_quad <= b_row
    A_quad =  [ [0, -U, 1, 0],
                [1, L, -U, 0],
                [-1, 0, 0, L],
                [0, 0, -1, 1],
                [0, -U, 0, 1],
                [0, U, 1, -1],
                ]
    b_quad = [0, L, 0, 0, 0, U]

    
    for d in range(D):
        for sign in ['+', '-']:   
            for sy in ['W','x']:  
                v_quad = [  z[d][sign][l][i][j], 
                                    q[d][sign][l][i][j],
                                    (W[sign][l][i][j]     if sy=='W' else   x[l][i] ), 
                                    (omega[sign][l][i][j] if sy=='W' else khi[l][i] )
                                    ]
            
            for A_row, b_row in zip(A_quad, b_quad):
                    expr = quicksum([A_row[id] * var for id,var in enumerate(v_quad)]) - b_row 
                    model.addCons( expr <= 0 )
    

# ---------------------------------------------------------------------------
# Constraints for the summation of the linearized quadratic terms
# ---------------------------------------------------------------------------
"""
0=y_{l}^{j+} \sum_{i=1}^{N_{l}}} z_{l}^{ij+}

0=y_{l}^{j-} \sum_{i=1}^{N_{l}}} z_{l}^{ij-}

"""
def add_linQuadSumm_constraint( l,j):
    a_linQ = [1] + [-1 for i in range(Nl[l]) ]
    for d in range(D):  
        for sign in ['+', '-']:
            v_linQ = [ y[d][sign][l][j]] +  [  z[d][sign][l][i][j] for i in range(Nl[l]) ]           
            expr = quicksum([a_linQ[id] * var for id,var in enumerate(v_linQ)]) 
            model.addCons( expr = 0 )




# ---------------------------------------------------------------------------
# Constraints for Relu activation:  A_relu * v_relu <= b_relu
# ---------------------------------------------------------------------------
# CONSTANTS:
A_relu = [[0, -1, 1, -1, 1, -1],
            [M_r, 1, -1, 1, -1,  1],
            [-M_r, 1, 0, 0,  0,  0]] 

b_relu = [0, M_r, 0 ]


def add_relu_constraint(l,j, x = x):
   # for l in l_subset: we take x = x_out
    
   for d in range(D):  
        v_relu =      [ r[d][l][j], 
                        x[d][l+1][j],
                        y[d]['+'][l][j], 
                        y[d]['-'][l][j],
                        b['+'][l][j],
                        b['-'][l][j] 
                        ]

        for A_row, b_row in zip(A_relu, b_relu):
            expr = quicksum([A_row[id] * var for id,var in enumerate(v_relu)]) - b_row 
            model.addCons( expr <= 0 )



# ---------------------------------------------------------------------------
# Hard constraints on the output binary relu activation and data labels
# ---------------------------------------------------------------------------
def add_output_constraint(labels , l=L):
    """ Assuming we have r[d][l] = label[d] , d = 0...D-1,  
        We constrain the binary relu variable for layer l, (default l=L ) to be the labels"""
    
    for d in range(D):
        for j in range(Nl[l+1]):
            # output of layer L
            model.addCons( r[d][L][j] - labels[d][j] == 0)


# ---------------------------------------------------------------------------
# Constraints on distance distx between x_in and x_out for l in domL
# Alternatively:  minimize with def_layer_objective(l) for l in domL
# ---------------------------------------------------------------------------
def add_distx_constraint( l,j ):
     """If we distinguidh x_in from x_out instead of posing x_in = x_out = x as above 
     
    
     """
     for d in range(D):
        # output of layer l for l in  l_subset
        model.addCons( x_in[d][l][j] - x_out[d][l][j] - distx[d]['+'][l][j] + distx[d]['-'][l][j] == 0 )

#----------------------------------------------------------------------
def layer_objective(l):
    assert l in l_subset
    # Define the objective function for layer l: sum over j and d indices 
    return sum( [ sum(distx[d]['+'][l]) + sum(distx[d]['-'][l])  
                  for d in range(D)]                              
              )
         
def def_objective(l_subset):
    objective = sum([ layer_objective(l) for l in l_subset])
    model.setObjective(objective, sense="minimize")



# ==========================================================================
# Define the constraints for layer l

def add_layer(l,x=x):    
    # layer where x_in = x_out as by default    
    assert l not in l_subset + [L]
    for j in range(Nl[l+1]):
        for i in range(Nl[l]):
            add_quad_constraint( l,i,j,x) # z[i,j] = w[i,j] * x[i] 
        add_linQuadSumm_constraint( l,j)  # y[j] = sum_{i}( z )
        add_relu_constraint(l,j,x )       # x[l+1][j] = relu( y[j] - b[j] )    
        
   
def add_distx_layer(l):    
    # layer where x_in =/= x_out and distx = abs(x_in[l+1] - x_out[l] )   
    assert l in l_subset

    for j in range(Nl[l+1]):
        for i in range(Nl[l]): 
            add_quad_constraint( l,i,j, x_in)  # z[i,j] = w[i,j] * x_in[i] 
        add_linQuadSumm_constraint( l,j)       # y[j] = sum_{i}( z )
        add_relu_constraint(l,j, x_out )       # x_out[l+1][j] = relu( y[j] - b[j] )  
        add_distx_constraint( l,j )            # distx = abs( x_out[l] - x_in[l] ) 
        # distx will be added to the objective function to minimize



#==============================================================


def def_constraints(dataset):

    x_data, labels = dataset 
    for l in range(L):
        if l==0:                add_layer(l, x_data)
        if l not in l_subset:   add_layer(l)
        else:                   add_distx_layer(l)
    
    #Output from the last layer (i.e. l=L-1) is x[L], r[L]
    add_output_constraint(labels , L)

def construct_NN(dataset ):
    
    # Assuming here that the hyperparameters and the variables are already defined as global
    #set_hyperparam()            # hyperparameters of the NN architecture
    #def_variables()
    def_constraints(dataset)    # Dataset incorporated at the NN construction
    def_objective()



# ==========================================================================
#           Solving the optimization or feasability problem 
# ==========================================================================
def set_solver():
    # Set the LP solver to use the simplex method
    # How can we adapt the strategy ??
    model.setIntParam("lp/solvertype", 0) # 1: interior point for LP
    # But this is a MILP ? 


def train_NN():
    # Training: Optimize the problem
    model.optimize()


def validate():
    # dont know how to do that !!??
    NotImplemented

def validate():
    NotImplemented

# ========================================================================
#   Construct and train the NN model
# ========================================================================
construct_NN(dataset )
set_solver()
train_NN()


# Print the optimal solution
print("Optimal solution:")
print(f"x = {model.getVal(x)}")
print(f"y = {model.getVal(y)}")
print(f"Objective value = {model.getObjVal()}")
