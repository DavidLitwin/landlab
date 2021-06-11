import numpy as np

cimport numpy as np
cimport cython


DTYPE = np.int
ctypedef np.int_t DTYPE_INT_t
DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

# @cython.boundscheck(False)
# @cython.wraparound(False)
def _calc_grad_at_link(np.ndarray[DTYPE_FLOAT_t, ndim=1] values,
          np.ndarray[DTYPE_INT_t, ndim=1] node_head,
          np.ndarray[DTYPE_INT_t, ndim=1] node_tail,
          np.ndarray[DTYPE_FLOAT_t, ndim=1] link_lengths,
          ):

	cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] out = np.empty(len(node_head), dtype=float)
	return np.divide(
		values[node_head] - values[node_tail],
		link_lengths,
		out=out,
	)


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
