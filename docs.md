



# Class FermiNet


# x (n_e, 3) sig (3, n_e * n_det) exp (n_e, n_e*n_det) ({e_0}^{n_e*n_det}, {e_1}}^{n_e*n_det}, ...)

# there should be a way to do this with init
  # do we have a baseline method? Always push to use pyfig? 
  # Can we use ** operator - if no, preinit everything, then options
  # Should indicate with * that everything is kw 
  # is the single electron feature graph invariant to sth
  """
    x (b, n_e, 3) e coord
    xu (b, n_u, 3) spin up e coord
    xd (b, n_d, 3) spin down e coord
    x_d :   : l1-norm e pos
    ee_disp :       : e-e displacement
    ee_d :       : e-e dist
    x_s:            : single stream variable
    x_p:    
    x_s_res:        : single stream residual 
    x_p_res: 

"""