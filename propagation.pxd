from eigency.core cimport *
from libcpp.string cimport string

ctypedef unsigned int uint

cdef extern from "instantAlg.cpp":
#cdef extern from "instantAlg_arxiv.cpp":
	pass

cdef extern from "instantAlg.h" namespace "propagation":
	cdef cppclass Instantgnn:
		Instantgnn() except+
		double initial_operation(string,string,uint,uint,double,double,Map[MatrixXd] &) except +
		void snapshot_operation(string, double, double, Map[MatrixXd] &) except +
		void overall_operation(double,double, Map[MatrixXd] &) except +
		void linenum_operation(string, int,int,double,double, Map[MatrixXd] &) except +
		int snapshot_operation_rate_Z(string, int, double, double, double, Map[MatrixXd] &, Map[MatrixXd] &)
