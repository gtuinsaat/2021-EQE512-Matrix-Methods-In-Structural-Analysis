"""
________________________________________________________________
| Gebze Technical University | Department of Civil Engineering | 
________________________________________________________________
|       EQEQ512 Matrix Methods in Structural Analysis          | 
________________________________________________________________
|         Dr. Ahmet Anıl Dindar (adindar@gtu.edu.tr)           |
________________________________________________________________

This Python file consists of member stiffness, system stiffness matrix functions
________________________________________________________________
"""

#----------------------------------------------------------------------------
# Importing modules
import numpy as np


#----------------------------------------------------------------------------
# Truss Member Stiffness Matrix
def truss_stiffness_creator(member_props) :
    """
    E: modulus of elasticiy
    A: Area 
    Joint I
    Joint J
    Return 
    k_member    
    """
    
    delta_1 = member_props["jointJ"][0] - member_props["jointI"][0] 
    delta_2 = member_props["jointJ"][1] - member_props["jointI"][1] 

    length = (delta_1**2 + delta_2**2)**.5

    c1 = delta_1 / length
    c2 = delta_2 / length

    a = c1**2
    b = c1*c2
    c = c2**2
    
    K = member_props["E"]*member_props["A"]/length

    K2 = np.array([[ a , b , -a , -b] , 
             [ b , c , -b , -c],
             [-a , -b , a , b ],
             [-b , -c , b , c ]])

    K_mem = K * K2
    return( K_mem )

#----------------------------------------------------------------------------
# Beam Member Stiffness Matrix

def beam_stiffness_matrix( member_props ):
    """
    member_props
    
    Example: 
    member_1 = {"E":30_000 ,  "I" : 26E-4 * 1.E12 ,  
                "jointI": joint_nodes[1] , 
                "jointJ": joint_nodes[2]}
                
    Returns    
    k: Member Stiffness matrix
    """
    
    import numpy as np
    E = member_props["E"]
    I = member_props["I"]
    
    # Length of the member
    delta_1 = member_props["jointJ"][0] - member_props["jointI"][0] 
    delta_2 = member_props["jointJ"][1] - member_props["jointI"][1] 
    L = (delta_1**2 + delta_2**2)**.5
    print( L )
    
    # Let's denote some terms
    twelveEIL3 = 12*E*I/(L**3)
    sixEIL2 = 6 * E * I / (L**2)
    fourEIL = 4 * E * I / (L)
    twoEIL =  2* E * I / (L)
    
    # Finally, the member stiffness matrix
    K = np.array( [[ twelveEIL3 , sixEIL2 , -1* twelveEIL3 , sixEIL2],
                  [sixEIL2 , fourEIL , -1* sixEIL2 , twoEIL],
                  [-1* twelveEIL3 , -1 * sixEIL2 , twelveEIL3, -1*sixEIL2],
                  [sixEIL2 , twoEIL , -1*sixEIL2 , fourEIL]])
                  
    return( K )

#----------------------------------------------------------------------------

# System Stiffness Matrix

def system_stiffness_matrix( k_members , joint_nodes , member_nodes , DOF = 2):
    """
    Inputs
    
    k_members :
    joint_nodes :
    member_nodes :
    DOF : 2
    
    Returns
    K : system stiffness Matrix
    
    """
    K = np.zeros( (len(joint_nodes) * DOF , len(joint_nodes) * DOF) , dtype=int)
    for beam_no , beam_node in member_nodes.items() :     
        k_temp = k_members[ beam_no]
        for counter_i , i in enumerate( beam_node ): 
            for counter_j , j in enumerate(beam_node) :
                add_k = k_temp[ counter_i * DOF  : counter_i * DOF + DOF  , counter_j * DOF  : counter_j * DOF + DOF  ] 
                if i == 1 and j == 1 : 
                    K[ 0  : DOF  , 0  : DOF] = K[ 0  : DOF  , 0  : DOF ] + add_k
                else:
                    K[ i * DOF - DOF  : i * DOF   , j * DOF - DOF : j * DOF  ] = K[ i * DOF - DOF  : i * DOF  , j * DOF - DOF : j * DOF  ] + add_k                             

    return( K ) 


##########################################################################

# Applying Boundary Conditions

def apply_boundary_conditions( K , U_factor , P_factor ) :
    K_bc = K
    for count_i , i in enumerate( U_factor ):
        for count_j , j in enumerate( P_factor):
            if i == 0 or j == 0:
                K_bc[count_i][count_j] = 0
                if count_i == count_j : 
                    K_bc[count_i][count_j] = 1

            else: 
                K_bc[count_i][count_j] = K_bc[count_i][count_j]

    return( K_bc )