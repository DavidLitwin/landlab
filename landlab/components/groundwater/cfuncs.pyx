import numpy as np

cimport numpy as np
cimport cython


DTYPE = np.int
ctypedef np.int_t DTYPE_INT_t
DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

# @cython.boundscheck(False)
# @cython.wraparound(False)
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


# @cython.boundscheck(False)
# @cython.wraparound(False)
def _calc_flux_div_at_node(np.ndarray[DTYPE_FLOAT_t, ndim=1] unit_flux,
          np.ndarray[DTYPE_INT_t, ndim=1] node_at_cell,
          np.ndarray[DTYPE_INT_t, ndim=1] link_at_face,
          np.ndarray[DTYPE_INT_t, ndim=2] faces_at_cell,
          np.ndarray[DTYPE_FLOAT_t, ndim=1] area_of_cell,
          np.ndarray[DTYPE_FLOAT_t, ndim=1] length_of_face,
          np.ndarray[DTYPE_INT_t, ndim=2] link_dirs_at_node,
          int number_of_cells,
          int number_of_nodes,
          ):

  cdef int number_of_dirs = link_dirs_at_node.shape[1]
  cdef int c
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] buf = np.empty(number_of_cells, dtype=float)
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] out = np.empty(number_of_nodes, dtype=float)
  cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] total_flux = unit_flux[link_at_face] * length_of_face

  for c in range(number_of_dirs):
    buf -= total_flux[faces_at_cell[:, c]] * link_dirs_at_node[node_at_cell, c]
  out[node_at_cell] = buf / area_of_cell
  return out

# _calc_net_face_flux_at_cell
# if out is None:
#     out = grid.empty(at="cell")
# total_flux = unit_flux_at_faces * grid.length_of_face
# out = np.zeros(grid.number_of_cells)
# fac = grid.faces_at_cell
# for c in range(grid.link_dirs_at_node.shape[1]):
#     out -= total_flux[fac[:, c]] * grid.link_dirs_at_node[grid.node_at_cell, c]
# return out
#
# out[grid.node_at_cell] = (
#     _calc_net_face_flux_at_cell(grid, unit_flux[grid.link_at_face])
#     / grid.area_of_cell
# )
