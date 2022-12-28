from propagation cimport Instantgnn

cdef class InstantGNN:
	cdef Instantgnn c_instantgnn

	def __cinit__(self):
		self.c_instantgnn=Instantgnn()

	def initial_operation(self,path,dataset,unsigned int m,unsigned int n,rmax,alpha,np.ndarray array3):
		return self.c_instantgnn.initial_operation(path.encode(),dataset.encode(),m,n,rmax,alpha,Map[MatrixXd](array3))

	def snapshot_operation(self, upfile, rmax,alpha, np.ndarray array3):
		return self.c_instantgnn.snapshot_operation(upfile.encode(), rmax, alpha, Map[MatrixXd](array3))

	def overall_operation(self, rmax,alpha, np.ndarray array3):
		return self.c_instantgnn.overall_operation(rmax, alpha, Map[MatrixXd](array3))

	def snapshot_operation_rate_Z(self, upfile, begin, rmax,alpha, threshold, np.ndarray array3, np.ndarray array4):
		return self.c_instantgnn.snapshot_operation_rate_Z(upfile.encode(), begin, rmax, alpha, threshold, Map[MatrixXd](array3), Map[MatrixXd](array4))

	def linenum_operation(self, upfile, begin, end, rmax,alpha, np.ndarray array3):
		return self.c_instantgnn.linenum_operation(upfile.encode(), begin, end, rmax, alpha, Map[MatrixXd](array3))