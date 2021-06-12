import numpy as np

cimport numpy as np
cimport cython


DTYPE = np.int
ctypedef np.int_t DTYPE_INT_t
DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

# # regularization functions used to deal with numerical demons of seepage
# cdef _regularize_G(np.ndarray[DTYPE_FLOAT_t, ndim=1] u, DTYPE_FLOAT_t reg_factor):
#     """Smooths transition of step function with an exponential. 0<=u<=1."""
#     return np.exp(-(1 - u) / reg_factor)
#
#
# cdef _regularize_R(np.ndarray[DTYPE_FLOAT_t, ndim=1] u):
#     """ramp function on u."""
#     return u * np.greater_equal(u, 0.)
#
#
# def fun_dhdt(float t,
#             np.ndarray[DTYPE_FLOAT_t, ndim=1] h,
#             np.ndarray[DTYPE_FLOAT_t, ndim=1] f,
#             np.ndarray[DTYPE_FLOAT_t, ndim=1] dqdx,
#             np.ndarray[DTYPE_FLOAT_t, ndim=1] reg_thickness,
#             np.ndarray[DTYPE_FLOAT_t, ndim=1] n,
#             float reg_factor):
#     """generate a function to pass to solve_ivp"""
#     cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] out = (
#         1
#         / n
#         * (
#             f
#             - dqdx
#             - _regularize_G(h / reg_thickness, reg_factor) * _regularize_R(f - dqdx)
#         )
#     return out
#
# def _calc_grad_at_link(np.ndarray[DTYPE_FLOAT_t, ndim=1] values,
#           np.ndarray[DTYPE_INT_t, ndim=1] node_head,
#           np.ndarray[DTYPE_INT_t, ndim=1] node_tail,
#           np.ndarray[DTYPE_FLOAT_t, ndim=1] link_lengths,
#           ):
#
# 	cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] out = np.empty(len(node_head), dtype=float)
# 	return np.divide(
# 		values[node_head] - values[node_tail],
# 		link_lengths,
# 		out=out,
# 	)


def _calc_flux_div_at_node(np.ndarray[DTYPE_FLOAT_t, ndim=1] unit_flux,
          np.ndarray[DTYPE_INT_t, ndim=1] node_at_cell,
          np.ndarray[DTYPE_INT_t, ndim=1] link_at_face,
          np.ndarray[DTYPE_INT_t, ndim=2] faces_at_cell,
          np.ndarray[DTYPE_FLOAT_t, ndim=1] area_of_cell,
          np.ndarray[DTYPE_FLOAT_t, ndim=1] length_of_face,
          np.ndarray[signed char, ndim=2] link_dirs_at_node,
          ):

  cdef int number_of_dirs = link_dirs_at_node.shape[1]
  cdef int number_of_nodes = link_dirs_at_node.shape[0]
  cdef int c
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] buf = np.empty_like(area_of_cell)
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] out = np.empty(number_of_nodes, dtype=float)
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] total_flux = unit_flux[link_at_face] * length_of_face

  for c in range(number_of_dirs):
    buf -= total_flux[faces_at_cell[:, c]] * link_dirs_at_node[node_at_cell, c]
  out[node_at_cell] = buf / area_of_cell
  return out

def _map_value_at_max_node_to_link(np.ndarray[DTYPE_FLOAT_t, ndim=1] controls,
          np.ndarray[DTYPE_FLOAT_t, ndim=1] values,
          np.ndarray[DTYPE_INT_t, ndim=1] node_at_link_head,
          np.ndarray[DTYPE_INT_t, ndim=1] node_at_link_tail,
          ):


  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] head_control = controls[node_at_link_head]
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] tail_control = controls[node_at_link_tail]
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] head_vals = values[node_at_link_head]
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] tail_vals = values[node_at_link_tail]
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] out = np.empty(len(node_at_link_head), dtype=float)

  out[:] = np.where(tail_control > head_control, tail_vals, head_vals)
  return out
