For Matlab:
	/src includes necessary algorithms for TGCR/NLTGCR. nltgcr_base is the original algorithm, nltgcr_opt
	uses Newton iteration as a baseline to enforce the resiude check. nltgcr_nc uses the negative curvature to 
	escape from the non local minimal points.
	/scripts includes files to run experiments including linear equation, minimax and softmax regression. Run 
           test_nonlinear_opt to test the nonconvex problems. Run grad_plot to see the landscape of the two toy
	problems rosenbrock and myfun3.
	/problem includes the two examples we examined on Monday. rosenbrockwithgrad is the one causing
	stagnation problem. myfun3 is the one with origin as local maximum.
	
	

For Python:
	mnist_nltgcr.py is the file we use for testing NLTGCR for a deep learning application on MNIST.
	test_linear_regression.py and test_nltgcr.py verify the correctness of our implementation. 