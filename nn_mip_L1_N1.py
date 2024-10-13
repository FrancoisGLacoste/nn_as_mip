"""nn_mip_L1_N1.py
In SCIP 
MIP problems to represent a neuron (neural network with a single neuron in a single layer 

"""

from pyscipopt import Model
from pyscipopt.scip import quicksum

# we assumew have function that return the dataset
dataset = [()] # each data point is a tuple (x0[d], nL[d])

# Create a SCIP model
model = Model("NN_MIP_L1_N1")

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
        for l in range(L) ]
        for d in range(D)]

#For l in l_subset: we define distx and define (x , x_out) instead of x
# where x_out is the relu output.
distx = [{sign: [[model.addVar(lb=0, vtype="C") for j in range(Nl[l+1]) ]
                for l in l_subset] 
                for sign in ['+', '-'] }
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


def add_quad_constraint( l,i,j):
    # The quadratic term z = w*x is equivalent to 2 linear systems: one for x, one for w

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


def add_relu_constraint(l,j):
    
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
def add_output_constraint(label , l=L):
    """ Assuming we have r[d][l] = label[d] , d = 0...D-1,  
        We constrain the binary relu variable for layer l, (default l=L ) to be the labels"""
    
    for d in range(D):
        for j in range(Nl[l+1]):
            # output of layer L
            model.addCons( r[d][L][j] - label[d][j] == 0)


# ---------------------------------------------------------------------------
# Constraints on distance distx between x_in and x_out for l in domL
# Alternatively:  minimize with def_layer_objective(l) for l in domL
# ---------------------------------------------------------------------------
def add_dist_constraint( l):
     """If we distinguidh x_in from x_out instead of posing x_in = x_out = x as above 
     
     DO NOT USE WHEN x_in = x_out = x
     """
     x_in = x, x_out = x  # NOTE: just so that it run  
     for d in range(D):
        for i in range(Nl[l]):
            # output of layer L
            model.addCons( x_in[l][i] - x_out[l][i] - distx['+'][l][i] + distx['-'][l][i] == 0 )

#----------------------------------------------------------------------
def layer_objective(l):
    # Define the objective function for layer l
    return sum(distx['+'][l]) + sum(distx['-'][l]) # summation over j indices 

def def_objective():
    # As in usual NN: only fit the output data to optimize the entire NN at once (not layer-wise)
    objective = layer_objective(L)
    model.setObjective(objective, sense="minimize")

def def_layer_objective(l):
    # NOT like in USUAL NN: in case we want to try to 
    # fit x layer-wise by sequentially optimizing each layer subproblem 
    objective = layer_objective(l)
    model.setObjective(objective, sense="minimize")

# ==========================================================================
# Define the constraints for a single layer l 

def add_layer(l):
    for j in range(Nl[l+1]):
        for i in range(Nl[l]):
            add_quad_constraint( l,i,j)
        add_linQuadSumm_constraint( l,j)
        add_relu_constraint(l,j)
        
   




#==============================================================


# ==========================================================================
def build_NN():
    for l in range(L):
        add_layer(l)




# Set the LP solver to use the simplex method
model.setIntParam("lp/solvertype", 0) # 1: interior point for LP


# Optimize the problem
model.optimize()

# Print the optimal solution
print("Optimal solution:")
print(f"x = {model.getVal(x)}")
print(f"y = {model.getVal(y)}")
print(f"Objective value = {model.getObjVal()}")
