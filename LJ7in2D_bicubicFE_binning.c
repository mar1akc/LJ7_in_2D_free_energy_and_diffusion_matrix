// This code does two things.
//
// (1) Reads well-tempered metadynamics data (Gaussian bumps) and approximates 
// the biasing potential with a bicubic spline. The derivatives are evaluated 
// by finite differences for smoothing purposes.
//
// (2) Runs a long trajectory in the biased force field and find the free energy 
// using the binning approach.

// Compile command:  gcc LJ7in2D_bicubicFE_binning.c -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BETA 10.0
#define GAMMA 1.0 // the artificial temperature for the discount factor
#define SIGMA 0.02 // width parameter for Gaussian bumps
#define NBUMPS_MAX 50000 // 50000 10000
#define HEIGHT 2.0e-2

#define N1 129 // the number of grid points along mu2-axis
#define N2 129 // the number of grid points along mu3-axis

#define NSTEPS 1e9  // the length of the stochastic trajectory that we bin

#define PI 3.141592653589793
#define PI2 6.283185307179586 // 2*PI
#define RSTAR 1.122462048309373 // 2^{1/6}

#define TAU 5.0e-5;
#define NATOMS 7 // the number of atoms
#define KAPPA 100.0 // spring constant for the restraining potential that turn on 
// if an atom is at distance more than 2 from the center of mass
#define mabs(a) ((a) >= 0 ? (a) : -(a))
#define sgn(a) ((a) == 0 ? 0 : ((a) > 0  ? 1 : -1 ))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define INFTY 1.0e6

struct vec2 {
	double x;
	double y;
};

//----- bicubic interpolation
void Gpot_and_ders_on_grid(int Nbumps,double *bump_CV1,double *bump_CV2,double *height,
	double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *pot,double *der1,double *der2,double *der12);
void compute_bicubic_coeff_matrix(double *pot,double *der1,double *der2,double *der12,
	double *Amatr,int ind);
double wsum0(double a,double b,double c,double d);	
double wsum1(double a,double b,double c,double d);	
double wsum2(double a,double b,double c,double d);	
double wsum3(double a,double b,double c,double d);	
void evaluate_Gpot_and_ders(double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *Amatr,double cv1,double cv2,double *FEval,double *FEder1,double *FEder2);
//----- running long trajectory in the biased potential and binning it
void init_conf(double *conf,int Natoms);
double LJpot(double *conf,int Natoms);
void LJpot_and_grad(double *conf,double *pot,double *grad,int Natoms);
char MALAstep(double *conf0,double *conf1,int Natoms,double dt, 
			double *Vpot0,double *Vpot1,double *Vgrad1,double *w);
struct vec2	box_mueller(void); // generates a pair of Gaussian random variables N(0,1)
// aligns configuration by solving Wahba's problem
// https://en.wikipedia.org/wiki/Wahba%27s_problem
void align( double *conf0, double *conf1, int Natoms ); 
// collective variables mu2 and mu3
struct vec2 CVs(double *conf,int Natoms);
void CVgrad(double *conf,double *mu2,double *mu3,
			double *mu2grad,double *mu3grad,int Natoms);
void restraining_pot_and_grad(double *conf,double *pot,double *grad,int Natoms);
void binning_trajectory(long *bins,double *grid_CV1,double *grid_CV2,
	int Nsteps,int Natoms,
	double *conf0,double dt,double h1,double h2,double *bicubic_matrix);
void FEders_on_grid(double *pot,double *der1,double *der2,double *der12);

//----- main	
int main(void);

//-------------------------------------------------------------


void Gpot_and_ders_on_grid(int Nbumps,double *bump_CV1,double *bump_CV2,double *height,
	double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *pot,double *der1,double *der2,double *der12) {
	int i,j,n,ind;
	double sig2 = SIGMA*SIGMA;
	double aux1,aux2,aux_exp,aux_der1;
	double fac = GAMMA/(GAMMA + 1.0/BETA);
	int n1m1 = N1-1, n2m1 = N2-1;
	
	// the derivatives are computed by finite differences for the purpose of smoothing 
	// while they are available analytically
	// the derivatives are with respect to parameters (x1,x2) that rescale the 
	// cell in the CV space only a unit square:
	// x1 = (CV1- CV10)/h1, x2 = (CV2-CV20)/h2; CV1 = x1*h1 + CV10, x2 = h2*CV2 + CV20
	// d(pot)/d(x1) = d(pot)/d(x1) d(x1)/d(CV1) = h1*d(pot)/d(x1)
	// likewise for d(pot)d(x2)
	
	for( i = 0; i < N1; i++ ) {
		for( j = 0; j < N2; j++ ) {
			ind = i + j*N1;	
			pot[ind] = 0.0;
			if( i == 0 || j == 0 || i == n1m1 || j == n2m1 ) {
				der1[ind] = 0.0;
				der2[ind] = 0.0;
				der12[ind] = 0.0;
			}
			for( n = 0; n < Nbumps; n++ ) {
				aux1 = grid_CV1[i] - bump_CV1[n];
				aux2 = grid_CV2[j] - bump_CV2[n];
				aux_exp = height[n]*exp(-0.5*(aux1*aux1 + aux2*aux2)/sig2);
				pot[ind] += aux_exp;
				if( i == 0 || j == 0 || i == n1m1 || j == n2m1 ) {
					der1[ind] -= aux1*aux_exp/sig2;
					der2[ind] -= aux2*aux_exp/sig2;
					der12[ind] += (aux1/sig2)*(aux2/sig2)*aux_exp;
				}
				
			}
			pot[ind] *= fac;
			if( i == 0 || j == 0 || i == n1m1 || j == n2m1 ) {
				der1[ind] *= fac*h1;
				der2[ind] *= fac*h2;
				der12[ind] *= fac*h1*h2;
			}		
		}
	}
	for( i = 1; i < n1m1; i++ ) {
		for( j = 1; j < n2m1; j++ ) {
			ind = i + j*N1;	
			der1[ind] = 0.5*(pot[ind+1]-pot[ind-1]);
			der2[ind] = 0.5*(pot[ind+N1]-pot[ind-N1]);

		}
	}
	for( i = 1; i < n1m1; i++ ) {
		for( j = 1; j < n2m1; j++ ) {
			ind = i + j*N1;	
			der12[ind]  = 0.5*(der1[ind+N1] - der1[ind-N1]);
		}
	}
}

//----------------------
void compute_bicubic_coeff_matrix(double *pot,double *der1,double *der2,double *der12,
	double *Amatr,int ind) {
	int i,j,ind0 = ind,ind1 = ind + 1,ind2 = ind + N1,ind3 = ind + N1 + 1;
	double F[16]; // matrix F defined row-wise
	double B[16];
	// row 1
	F[0] = pot[ind0];
	F[1] = pot[ind2];
	F[2] = der2[ind0];
	F[3] = der2[ind2];
	// row 2
	F[4] = pot[ind1];
	F[5] = pot[ind3];
	F[6] = der2[ind1];
	F[7] = der2[ind3];
	// row 3
	F[8] = der1[ind0];
	F[9] = der1[ind2];
	F[10] = der12[ind0];
	F[11] = der12[ind2];
	// row 4
	F[12] = der1[ind1];
	F[13] = der1[ind3];
	F[14] = der12[ind1];
	F[15] = der12[ind3];
	// Computes A : = M F M^\top where
	// F = [f(0,0),f(0,1),der2(0,0),der2(0,1);
	//		f(1,0),f(1,1),der2(1,0),der2(1,1);
	//		der1(0,0),der1(0,1),der12(0,0),der12(0,1);
	//		der1(1,0),der1(1,1),der12(1,0),der12(1,1)]
	// (0,0) corresponds to ind0 = ind, 
	// (1,0) corresponds to ind1 = ind + 1, 
	// (0,1) corresponds to ind2 = ind + N1,
	// (1,1) corresponds to ind3 = ind + 1 + N1
	
	// B = FM^\top
	for( i = 0; i < 4; i++ ) {
		j = i*4;
		B[j] = wsum0(F[j],F[j+1],F[j+2],F[j+3]);
		B[j+1] = wsum1(F[j],F[j+1],F[j+2],F[j+3]);
		B[j+2] = wsum2(F[j],F[j+1],F[j+2],F[j+3]);
		B[j+3] = wsum3(F[j],F[j+1],F[j+2],F[j+3]);	
	}
	// A = M*B 
	for( i = 0; i < 4; i++ ) {
		Amatr[ind*16 + i] = wsum0(B[i],B[i+4],B[i+8],B[i+12]);
		Amatr[ind*16 + i + 4] = wsum1(B[i],B[i+4],B[i+8],B[i+12]);
		Amatr[ind*16 + i + 8] = wsum2(B[i],B[i+4],B[i+8],B[i+12]);
		Amatr[ind*16 + i + 12] = wsum3(B[i],B[i+4],B[i+8],B[i+12]);
	}	
}

// these functions perform multiplication of the matrix
// M = [1,0,0,0;
//      0,0,1,0;
//     -3,3,-2,1;
//      2,-2,1,1;]
// by vector [a,b,c,d]^\top

double wsum0(double a,double b,double c,double d) {
	return a;
}
double wsum1(double a,double b,double c,double d) {
	return c;
}
double wsum2(double a,double b,double c,double d) {
	return 3.0*(b-a) - 2.0*c - d;
}
double wsum3(double a,double b,double c,double d) {
	return 2.0*(a-b) + c + d;
}

//-------------------
// evaluate the free energy and its gradient at a query point
void evaluate_Gpot_and_ders(double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *Amatr,double cv1,double cv2,double *FEval,double *FEder1,double *FEder2) {
	int i,j,ind,ishift;
	double x,y;
	// FEval(x,y) = \sum_{i,j=0}^3 a(i,j)x^i y^j
	// FEder1(x,y) = \sum_{i=1}^3\sum{j=0}^3 a(i,j)ix^{i-1} y^j
	// FEder2(x,y) = \sum_{i=0}^3\sum{j=1}^3 a(i,j)jx^i y^{j-1}
	
	// find the cell
	i = min(max(0,(int)floor((cv1 - grid_CV1[0])/h1)),N1-2);
	j = min(max(0,(int)floor((cv2 - grid_CV2[0])/h2)),N2-2);
	x = (cv1 - grid_CV1[0] - h1*i)/h1;
	y = (cv2 - grid_CV2[0] - h2*j)/h2;
	
	ind = i + N1*j;
	ishift = ind*16;
	*FEval = 0.0;
	*FEder1 = 0.0;
	*FEder2 = 0.0;
	for( i=0; i<4; i++ ) {
		for( j=0; j<4; j++ ) {
			*FEval += Amatr[ishift + i*4 + j]*pow(x,i)*pow(y,j);			
		}
	}
	for( i=1; i<4; i++ ) {
		for( j=0; j<4; j++ ) {
			*FEder1 += Amatr[ishift + i*4 + j]*i*pow(x,i-1)*pow(y,j);			
		}
	}
	for( i=0; i<4; i++ ) {
		for( j=1; j<4; j++ ) {
			*FEder2 += Amatr[ishift + i*4 + j]*j*pow(x,i)*pow(y,j-1);			
		}
	}
}




//-------------------------------------------------------------


//-------------------------------------------------------------


void init_conf(double *conf,int Natoms) {
	// hexagon
	double a = PI/3.0;
	int k;
	conf[0] = 0.0;
	conf[Natoms] = 0.0;
	for( k=1; k<7; k++ ) {
		conf[k] = RSTAR*cos((k-1)*a);
		conf[k+Natoms] = RSTAR*sin((k-1)*a);
	}
}


struct vec2	box_mueller(){
	double x1, y1, p, q;
	struct vec2 g;
	
	do{
			p=random();
			x1 = p/RAND_MAX;
			p=random();
			y1 = p/RAND_MAX;
	}
	while( x1 == 0.0 );
		/* Box-Muller transform */
		p=PI2*y1;
		q=2.0*log(x1);
		g.x=cos(p)*sqrt(-q);
		g.y=sin(p)*sqrt(-q);
		return g;
}

//------------------------------------------------------------

double LJpot(double *conf,int Natoms) {
	double dist_squared,rm6,dx,dy,pot = 0.0;
	int j,k;
	// pot = 4*sum_{j < k}(r_{jk}^{-12} - r_{jk}^{-6})
	for( k = 1; k < Natoms; k++ ) {
		for( j = 0; j < k; j++ ) {
			dx = conf[k] - conf[j];
			dy = conf[k+Natoms] - conf[j+Natoms];
			dist_squared = dx*dx + dy*dy;
			rm6 = 1.0/(dist_squared*dist_squared*dist_squared);
			pot += rm6*(rm6 - 1.0);
		}
	}
	pot *= 4.0;
	return pot;
}

void LJpot_and_grad(double *conf,double *pot,double *grad,int Natoms) {
	double aux,rm6,rm8,rm14,dx,dy,dist_squared;
	int j,k;
	// grad[k] = 4*sum_{j \neq k}(-12*r_{jk}^{-14} + r_{jk}^{-8})*(conf[j]-conf[k])
	for( k = 0; k < Natoms; k++ ) {
		grad[k] = 0.0;
		grad[k+Natoms] = 0.0;
	}
	*pot = 0.0;
	for( k = 1; k < Natoms; k++ ) {
		for( j = 0; j < k; j++ ) {
			dx = conf[k] - conf[j];
			dy = conf[k+Natoms] - conf[j+Natoms];
			dist_squared = dx*dx + dy*dy;
			rm6 = 1.0/(dist_squared*dist_squared*dist_squared);
			*pot += rm6*(rm6 - 1.0);
			rm8 = rm6/dist_squared;
			aux = (-12.0*(rm6*rm8) + 6.0*rm8)*dx;
			grad[k] += aux;
			grad[j] -= aux;
			aux = (-12.0*(rm6*rm8) + 6.0*rm8)*dy;
			grad[k+Natoms] += aux;
			grad[j+Natoms] -= aux;			
		}
	}
	*pot *= 4.0;
	for( k = 0; k < Natoms; k++ ) {
		grad[k] *= 4.0;
		grad[k+Natoms] *= 4.0;
	}
}

//------------------------------------------------------------
char MALAstep(double *conf0,double *conf1,int Natoms,double dt, 
			double *Vpot0,double *Vpot1,double *Vgrad1,double *w){
	int k;
	double aux,Q01 = 0.0,Q10 = 0.0; // transition probabilities between configurations 0 and 1
	double alpha,eta;
	char ch;
	// evaluate the transition probabilities Q01 and Q10
	Q01 = 0.0;
	Q10 = 0.0;
	for( k=0; k < Natoms; k++ ) {
		Q01 += w[k]*w[k] + w[k+Natoms]*w[k+Natoms];
		aux = conf0[k]-conf1[k] + dt*Vgrad1[k];
		Q10 += aux*aux;
		aux = conf0[k+Natoms]-conf1[k+Natoms] + dt*Vgrad1[k+Natoms];
		Q10 += aux*aux;
	}
	alpha = exp(-BETA*((*Vpot1) - (*Vpot0) +(Q10-Q01)*0.25/dt));
	if( alpha >= 1.0 ) { // accept move
		ch = 1;		
	}
	else { // accept move with probability alpha
		eta = (double)random();
		eta /= RAND_MAX; // uniform random variable on (0,1)
		ch = ( eta < alpha ) ? 1 : 0; 
	}
	return ch;	
}

//------------------------------------------------------------

void align( double *conf0, double *conf1, int Natoms ) {
	double B11 = 0.0, B12 = 0.0, B21 = 0.0, B22 = 0.0;
	double xc = 0.0, yc = 0.0;
	double alpha = 0.0, c, s, x, y; // the angle
	int k;
	// format: conf = [x0, x1, ..., x_{Natoms}, y0, y1, ..., y_{Natoms}]
	// conf0 must be centered so that its center of mass is at the origin
	// B = Sum(u_i v_i^\top)
	
	// center conf1
	for( k = 0; k < Natoms; k++ ){
		xc += conf1[k];
		yc += conf1[k+Natoms];
	}
	xc /= Natoms;
	yc /= Natoms;
	for( k = 0; k < Natoms; k++ ){
		conf1[k] -= xc;
		conf1[k+Natoms] -= yc;
	}
	
	// 	// B = [conf0_x;conf0_y][conf1_x;conf1_y]^\top
	for( k = 0; k < Natoms; k++ ) {
		B11 += conf0[k]*conf1[k];
		B12 += conf0[k]*conf1[k+Natoms];
		B21 += conf0[k+Natoms]*conf1[k];
		B22 += conf0[k+Natoms]*conf1[k+Natoms];
	} 

	// solve 1D optimization problem
	// f(x) = (B11 - cos(x))^2 + (B12 + sin(x))^2 +
	// (B21 - sin(x))^2 + (B22 - cos(x))^2 --> min
	// f(x) = -(B11+B22)*cos(x) - (B21-B12)*sin(x)
	// f'(x) = (B11+B22)*sin(x) - (B21-B12)*cos(x)
	// alpha = atan((B21-B12)/(B11+B22));
	// R = [cos(x),-sin(x); sin(x), cos(x)]
	alpha = atan((B21-B12)/(B11+B22));
	c = cos(alpha);
	s = sin(alpha);
	for( k = 0; k < Natoms; k++ ){
		x = conf1[k];
		y = conf1[k+Natoms];
		conf1[k] = x*c - y*s;
		conf1[k+Natoms] = x*s + y*c;
	}
}
//------------------------------------------------------------
// Computes CVs mu2 and mu3 given atomic coordinates
// c_i(x) = sum_{j\neq i} (1 - (r_{ij}/1.5)^8) / (1 - (r_{ij}/1.5)^{16})
// mean(c_i) = [sum_{i=1}^{Natoms} c_i] / Natoms
// \mu_2(x) = [\sum_{i} (c_i - mean(c_i))^2] / Natoms
// \mu_3(x) = [\sum_{i} (c_i - mean(c_i))^3] / Natoms

struct vec2 CVs(double *conf,int Natoms) {
	double coord_num[Natoms],mean_coord_num = 0.0;
	double aux_x, aux_y,r2,r8,r16,aux;
	int j,k;
	struct vec2 colvars;
	
	// compute coordination numbers
	for( k=0; k<Natoms; k++ ) {
		coord_num[k] = 0.0;
	}
	for( k=1; k<Natoms; k++ ) {	
		for( j=0; j<k; j++ ) {
			aux_x = conf[k] - conf[j];
			aux_y = conf[k+Natoms] - conf[j+Natoms];
			r2 = (aux_x*aux_x + aux_y*aux_y)/2.25;
			r8 = pow(r2,4);
			r16 = r8*r8;
			aux = (1.0 - r8)/(1.0 - r16);
			coord_num[k] += aux;
			coord_num[j] += aux;
		}		
	}
	// compute mean coordination number
	for( k=0; k<Natoms; k++ ) mean_coord_num += coord_num[k];
	mean_coord_num /= Natoms;

	// compute mu2 and mu3
	for( k=0; k<Natoms; k++ ) coord_num[k] -= mean_coord_num;
	colvars.x = 0.0;
	colvars.y = 0.0;
	for( k=0; k<Natoms; k++ ) {
		aux = coord_num[k]*coord_num[k];
		colvars.x += aux;
		colvars.y += aux*coord_num[k];
	}
	colvars.x /= Natoms;
	colvars.y /= Natoms;
	
	return colvars;
}

//-----------------------------------------------------------


//-----------------------------------------------------------
// Compute the CVs mu2 and mu3 and their gradients 
void CVgrad(double *conf,double *mu2,double *mu3,
			double *mu2grad,double *mu3grad,int Natoms) {
	double coord_num[Natoms],mean_coord_num = 0.0;
	double aux_x,aux_y,r2,r8,r16,aux,iden,fac;
	int j,k;
	double grad_coord_num[Natoms][2*Natoms],mean_grad[2*Natoms];
	int dim = 2*Natoms;
	const double sigma2 = 2.25; // 1.5^2

	// initialization	
	for( k=0; k<Natoms; k++ ) {
		coord_num[k] = 0.0;		
		for( j=0; j<dim; j++ ) {
			grad_coord_num[k][j] = 0.0;	
		}
	}
	for( j=0; j<dim; j++ ) {
		mean_grad[j] = 0.0;
		mu2grad[j] = 0.0;
		mu3grad[j] = 0.0;
	}	
	
	// compute coordination numbers
	for( k=1; k<Natoms; k++ ) {	
		for( j=0; j<k; j++ ) {
			aux_x = conf[k] - conf[j];
			aux_y = conf[k+Natoms] - conf[j+Natoms];
			r2 = (aux_x*aux_x + aux_y*aux_y)/sigma2;
			r8 = pow(r2,4);
			r16 = r8*r8;
			aux = (1.0 - r8)/(1.0 - r16);
			coord_num[k] += aux;
			coord_num[j] += aux;
			iden = 1.0/(1.0 - r16);
			fac = -4.0*pow(r2,3)*iden + aux*8.0*pow(r2,7)*iden;
			aux = fac*2.0*aux_x/sigma2;
			grad_coord_num[k][k] += aux;
			grad_coord_num[k][j] -= aux;
			grad_coord_num[j][k] += aux;
			grad_coord_num[j][j] -= aux;
			aux = fac*2.0*aux_y/sigma2;
			grad_coord_num[k][k+Natoms] += aux;
			grad_coord_num[k][j+Natoms] -= aux;
			grad_coord_num[j][k+Natoms] += aux;
			grad_coord_num[j][j+Natoms] -= aux;
		}		
	}
	
		// compute mean coordination number and its grad
	for( k=0; k<Natoms; k++ ) mean_coord_num += coord_num[k];
	mean_coord_num /= Natoms;
	for( j=0; j<dim; j++ ) {
		for( k=0; k<Natoms; k++ ) mean_grad[j] += grad_coord_num[k][j];
		mean_grad[j] /= Natoms;
	}

	// compute mu2 and mu3 and their gradients
	for( k=0; k<Natoms; k++ ) coord_num[k] -= mean_coord_num;
	*mu2 = 0.0;
	*mu3 = 0.0;
	for( k=0; k<Natoms; k++ ) {
		aux = coord_num[k]*coord_num[k];
		*mu2 += aux;
		*mu3 += aux*coord_num[k];
		for( j=0; j<dim; j++ ) {
			aux = coord_num[k]*(grad_coord_num[k][j] - mean_grad[j]);
			mu2grad[j] += 2.0*aux;
			mu3grad[j] += 3.0*coord_num[k]*aux;
		}
	}
	*mu2 /= Natoms;
	*mu3 /= Natoms;
	for( j=0; j<dim; j++ ) {
		mu2grad[j] /= Natoms;
		mu3grad[j] /= Natoms;
	}

}

//------------------------------------------------------------
// Restraining pot and grad
void restraining_pot_and_grad(double *conf,double *pot,double *grad,int Natoms) {
	double xc = 0.0, yc = 0.0, dist2, aux_x, aux_y,spring_pot;
	int k;
	
	// center conf
	for( k = 0; k < Natoms; k++ ){
		xc += conf[k];
		yc += conf[k+Natoms];
	}
	xc /= Natoms;
	yc /= Natoms;	
	for( k = 0; k < Natoms; k++ ){
		aux_x = conf[k] - xc;
		aux_y = conf[k+Natoms] - yc;
		dist2 = aux_x*aux_x + aux_y*aux_y - 4.0;
		if( dist2 > 0.0 ) {
			*pot += KAPPA*dist2*0.5;
			grad[k] -= KAPPA*aux_x;
			grad[k+Natoms] -= KAPPA*aux_y;		
		}
	}
}

//------------------------------------------------------------
// Evaluate the total potential energy and its gradient

void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad,
	double *CV1,double *CV2,double *CV1grad,double *CV2grad,
	double *grid_CV1,double *grid_CV2,double h1,double h2,
	double *bicubic_matrix,double *FEpot,double *FEder1,double *FEder2) {
	
	int j,dim = Natoms*2;

	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
	CVgrad(conf,CV1,CV2,CV1grad,CV2grad,Natoms);			
	evaluate_Gpot_and_ders(grid_CV1,grid_CV2,h1,h2,bicubic_matrix,*CV1,*CV2,
				FEpot,FEder1,FEder2);
	// need to divide by h as FEder is the derivative w.r.t a parameter in a\in(0,1)
	// d(FE)/dCV = d(FE)/da * da/dCV, a(CV) = (CV - CV0)/h, da/dCV = 1/h	
	for( j=0; j<dim; j++ ) {
		grad[j] += *FEder1*CV1grad[j]/h1 + *FEder2*CV2grad[j]/h2;
	}	
	*pot += *FEpot;	
	
// 	printf("CV1 = %.4e, CV2 = %.4e\n",*CV1,*CV2);
// 	printf("FEpot = %.4e,FEder1 = %.4e,FEder2 = %.4e\n",*FEpot,*FEder1,*FEder2);
		
}





//------------------------------------------------------------
void binning_trajectory(long *bins,double *grid_CV1,double *grid_CV2,
	int Nsteps,int Natoms,
	double *conf0,double dt,double h1,double h2,double *bicubic_matrix) {

	int j,n,k,dim = Natoms*2,j1,j2;
	struct vec2 gauss01;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *CV1,*CV2,*CV1grad,*CV2grad,*FEpot,*FEder1,*FEder2;
	double sig2 = SIGMA*SIGMA;
	struct vec2 cv;
	char ch;
	
	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	CV1 = (double *)malloc(sizeof(double));
	CV2 = (double *)malloc(sizeof(double));
	CV1grad = (double *)malloc(dim*sizeof(double));
	CV2grad = (double *)malloc(dim*sizeof(double));
	FEpot = (double *)malloc(sizeof(double));
	FEder1 = (double *)malloc(sizeof(double));
	FEder2 = (double *)malloc(sizeof(double));
	
	total_pot_and_grad(conf0,Natoms,Vpot0,Vgrad0,CV1,CV2,CV1grad,CV2grad,
		grid_CV1,grid_CV2,h1,h2,bicubic_matrix,FEpot,FEder1,FEder2);
				
	for( j = 0; j < Nsteps; j++ ) {		
		// generate array of random vars N(0,std) of size 2*Natoms
		for( k=0; k<Natoms; k++ ) {
			gauss01 = box_mueller();
			w[k] = std*gauss01.x;
			w[k+Natoms] = std*gauss01.y;			
		}
		// propose move
		for( k = 0; k < Natoms; k++ ) {
			conf1[k] = conf0[k] - dt*Vgrad0[k] + w[k];
			conf1[k+Natoms] = conf0[k+Natoms] - dt*Vgrad0[k+Natoms] + w[k+Natoms];
		}
		// evaluate the potential and the gradient at the proposed point
		total_pot_and_grad(conf1,Natoms,Vpot1,Vgrad1,CV1,CV2,CV1grad,CV2grad,
			grid_CV1,grid_CV2,h1,h2,bicubic_matrix,FEpot,FEder1,FEder2);
		ch = MALAstep(conf0,conf1,Natoms,dt,Vpot0,Vpot1,Vgrad1,w);
		if( ch == 1 ) { // step was accepted
			// align configurations
			align(conf0,conf1,Natoms);
			for( k=0; k<dim; k++ ) {
				conf0[k] = conf1[k];
				Vgrad0[k] = Vgrad1[k];
			}
			*Vpot0 = *Vpot1;
		}
		// bin the current position 
		// the grid points are the centers of the bins
		
		j1 = min(max(0,(int)floor((*CV1 - grid_CV1[0])/h1+0.5)),N1-1);
		j2 = min(max(0,(int)floor((*CV2 - grid_CV2[0])/h2 +0.5)),N2-1);
		
// 		printf("%i, %i\n",j1,j2);
// 		printf("%.4f, %.4e\n",(*CV1 - grid_CV1[0])/h1+0.5,(*CV2 - grid_CV2[0])/h2 +0.5);
// 		exit(1);
		
		
		bins[j1 + j2*N1]++;
		
		if( j%100000 == 0 ) printf("binning: %i steps\n",j);
	}
}

//-----------------------------------------------------------

void FEders_on_grid(double *pot,double *der1,double *der2,double *der12) {

	int n1m1 = N1-1, n2m1 = N2-1;
	int i,j,ind;

	// interior grid points
	for( i = 1; i < n1m1; i++ ) {
		for( j = 1; j < n2m1; j++ ) {
			ind = i + j*N1;	
			der1[ind] = 0.5*(pot[ind+1]-pot[ind-1]);
			der2[ind] = 0.5*(pot[ind+N1]-pot[ind-N1]);

		}
	}
	for( i = 1; i < n1m1; i++ ) {
		for( j = 1; j < n2m1; j++ ) {
			ind = i + j*N1;	
			der12[ind]  = 0.5*(der1[ind+N1] - der1[ind-N1]);
		}
	}
	// borders i = 0 and i = n1m1
	for( j = 1; j < n2m1; j++ ) {
		// i = 0
		ind = j*N1;
		der1[ind] = pot[ind+1] - pot[ind];
		der2[ind] = 0.5*(pot[ind+N1]-pot[ind-N1]);
		// i = n1m1
		ind = j*N1+n1m1;
		der1[ind] = pot[ind] - pot[ind-1];
		der2[ind] = 0.5*(pot[ind+N1]-pot[ind-N1]);
	}
	for( j = 1; j < n2m1; j++ ) {
		// i = 0
		ind = N1*j;
		der12[ind] = 0.5*(der1[ind+N1] - der1[ind-N1]);
		// i = n1m1
		ind = j*N1+n1m1;
		der12[ind] = 0.5*(der1[ind+N1] - der1[ind-N1]);		
	}
	// borders j = 0 and j = n2m1
	for( i = 1; i < n1m1; i++ ) {
		// j = 0
		ind = i;
		der1[ind] = 0.5*(pot[ind+1]-pot[ind-1]);
		der2[ind] = pot[ind+N1]-pot[ind];
		// j = n2m1
		ind = n2m1*N1+i;
		der1[ind] = 0.5*(pot[ind+1]-pot[ind-1]);
		der2[ind] = pot[ind]-pot[ind-N1];
	}
	for( i = 1; i < n1m1; i++ ) {
		// j = 0
		ind = i;
		der12[ind] = der1[ind+N1] - der1[ind];
		// j = n2m1
		ind = n2m1*N1+i;
		der12[ind] = der1[ind]-der1[ind-N1];
	}
	// corners
	// i = 0; j = 0;
	der1[0] = pot[1] - pot[0];
	der2[1] = pot[N1] - pot[0];
	der12[0] = der1[N1] - der1[0];
	// i = n1m1; j = 0;
	der1[n1m1] = pot[n1m1] - pot[n1m1-1];
	der2[n1m1] = pot[N1+n1m1] - pot[n1m1];
	der12[n1m1] = der1[N1+n1m1] - der1[n1m1];
	// i = 0; j = n2m1;
	ind = N1*n2m1;
	der1[ind] = pot[ind+1] - pot[ind];
	der2[ind] = pot[ind] - pot[ind-N1];
	der12[ind] = der1[ind] - der1[ind-N1];
	// i = n1m1; j = n2m1;
	ind =  n1m1 + N1*n2m1;
	der1[ind] = pot[ind] - pot[ind-1];
	der2[ind] = pot[ind] - pot[ind-N1];
	der12[ind] = der1[ind] - der1[ind-N1];
}






//------------------------------------------------------------
int main(void){
	int Nbumps, Ngrid = N1*N2, Nsteps = NSTEPS, Natoms = NATOMS;
	int i,j,ind,n,dim=NATOMS*2;
	double h1,h2;
	double *height,*val1,*val2;
	double *val1_min,*val1_max,*val2_min,*val2_max;
	double *grid_CV1,*grid_CV2,*grid_pot,*grid_der1,*grid_der2,*grid_der12;
	double *bicubic_matrix;
	FILE *fpot,*fder1,*fder2,*fder12,*fpar,*fbicubic;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    char fpot_name[] = "GBumpsData/GaussianBumps_beta5.txt";
    double dt = TAU;

	val1 = (double *)malloc(NBUMPS_MAX*sizeof(double));
	val2 = (double *)malloc(NBUMPS_MAX*sizeof(double));
	height = (double *)malloc(NBUMPS_MAX*sizeof(double));
	val1_min = (double *)malloc(sizeof(double));
	val1_max = (double *)malloc(sizeof(double));
	val2_min = (double *)malloc(sizeof(double));
	val2_max = (double *)malloc(sizeof(double));
	
// read the metadynamics data for the Gauusian bumps
	fpot = fopen(fpot_name,"r");
	
	n = 0;
	*val1_min = INFTY;
	*val1_max = -INFTY;
	*val2_min = INFTY;
	*val2_max = -INFTY;
	while( !feof(fpot) && n < NBUMPS_MAX ) {
		fscanf( fpot,"%le\t%le\t%le\n",height+n,val1+n,val2+n);
		*val1_min = min(*val1_min,val1[n]);
		*val1_max = max(*val1_max,val1[n]);
		*val2_min = min(*val2_min,val2[n]);
		*val2_max = max(*val2_max,val2[n]);
		n++;
	}
	fclose(fpot);
	printf("The total number of bumps is n = %i\n",n);
	printf("val1_min = %.4e\n",*val1_min);
	printf("val1_max = %.4e\n",*val1_max);
	printf("val2_min = %.4e\n",*val2_min);
	printf("val2_max = %.4e\n",*val2_max);
	Nbumps = n;
	
// Compute data for constructing the bicubic spline
	grid_CV1 = (double *)malloc(N1*sizeof(double));
	grid_CV2 = (double *)malloc(N2*sizeof(double));
	h1 = (*val1_max-*val1_min)/(N1-1);
	h2 = (*val2_max-*val2_min)/(N2-1);
	
	// extend the domain
	*val1_min -= 5.0*h1;
	*val1_max += 5.0*h1;
	*val2_min -= 5.0*h2;
	*val2_max += 5.0*h2;
	
	h1 = (*val1_max-*val1_min)/(N1-1);
	h2 = (*val2_max-*val2_min)/(N2-1);
	
	for( i=0; i<N1; i++ ) {
		grid_CV1[i] = *val1_min + h1*i;		
	}
	for( i=0; i<N2; i++ ) {
		grid_CV2[i] = *val2_min + h2*i;		
	}
	
	printf("Computing FE data on the grid");
	grid_pot = (double *)malloc(Ngrid*sizeof(double));
	grid_der1 = (double *)malloc(Ngrid*sizeof(double));
	grid_der2 = (double *)malloc(Ngrid*sizeof(double));
	grid_der12 = (double *)malloc(Ngrid*sizeof(double));

	Gpot_and_ders_on_grid(Nbumps,val1,val2,height,
		grid_CV1,grid_CV2,h1,h2,grid_pot,grid_der1,grid_der2,grid_der12);
	printf(" ... done!\n");
	

// Compute coefficient matrices for the bicubic spline	
	printf("Computing the bicubic matrix on the grid");

	bicubic_matrix = (double *)malloc(Ngrid*16*sizeof(double));	
	
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(grid_pot,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind);
		}
	}
	printf(" ... done!\n");
	
// // Evaluate the bicubic spline on a finer grid
// 	int n1fine = (N1-1)*4 + 1,n2fine = (N2-1)*4 + 1;
// 	double x,y,h1fine,h2fine;
// 	double *FEpot,*FEder1,*FEder2;
// 	
// 	h1fine = (*val1_max - *val1_min)/n1fine;
// 	h2fine = (*val2_max - *val2_min)/n2fine;
// 
// 	FEpot = (double *)malloc(sizeof(double));
// 	FEder1 = (double *)malloc(sizeof(double));
// 	FEder2 = (double *)malloc(sizeof(double));
// 	
// 	fpot = fopen("pot_fine.txt","w");
// 	fder1 = fopen("der1_fine.txt","w");
// 	fder2 = fopen("der2_fine.txt","w");
// 	
// 	for( j = 0; j < n2fine; j++ ) {
// 		y = *val2_min + h2fine*j;
// 		for( i = 0; i < n1fine; i++ ) {
// 			x = *val1_min + h1fine*i;
// 			evaluate_Gpot_and_ders(grid_CV1,grid_CV2,h1,h2,bicubic_matrix,x,y,
// 				FEpot,FEder1,FEder2);
// 			fprintf(fpot,"%.10e\t",*FEpot);
// 			fprintf(fder1,"%.10e\t",*FEder1);
// 			fprintf(fder2,"%.10e\t",*FEder2);	
// 		}
// 		fprintf(fpot,"\n");
// 		fprintf(fder1,"\n");
// 		fprintf(fder2,"\n");
// 	}	
// 	fclose(fpot);
// 	fclose(fder1);
// 	fclose(fder2);
	
	// save the parameters and the bicubic matrix 
	fpar = fopen("Data/bicubic_params.txt","w");
	fprintf(fpar,"%i\n",N1);
	fprintf(fpar,"%i\n",N2);
	fprintf(fpar,"%.10e\n",h1);
	fprintf(fpar,"%.10e\n",h2);
	fprintf(fpar,"%.10e\n",*val1_min);
	fprintf(fpar,"%.10e\n",*val1_max);
	fprintf(fpar,"%.10e\n",*val2_min);
	fprintf(fpar,"%.10e\n",*val2_max);
	fclose(fpar);
	
	// run a long trajectory in the biased potential and bin it
	long *bins;
	bins = (long *)malloc(Ngrid*sizeof(long));
	for( j=0; j<Ngrid; j++ ) bins[j] = 0;
	
	double *conf0;
	conf0 = (double *)malloc(dim*sizeof(double));
	init_conf(conf0,Natoms); 
	
 	CPUbegin=clock(); // start time measurement
 	binning_trajectory(bins,grid_CV1,grid_CV2,Nsteps,Natoms,
		conf0,dt,h1,h2,bicubic_matrix);
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("CPU time = %g\n",cpu);
	
	// restore the invariant measure
	// FE = free energy, BP = biasing potential
	// bins[i,j] \propto exp( - beta*(FE[i,j] + BP[i,j]))
	// p[i,j] \propto exp( -beta*FE[i,j] ) \propto bins[i,j]*exp( beta BP[i,j])
	
	
	// record the free energy
	char fname[100];	
	sprintf(fname,"Data/LJ7bins_beta%.0f.txt",BETA);
	fpot = fopen(fname,"w");
	for( i=0; i<N1; i++ ) {
		for( j=0; j<N2; j++ ) {
			ind = i + j*N1;
			fprintf(fpot,"%li\t",bins[ind]);
		}
		fprintf(fpot,"\n");
	}
	fclose(fpot);
	
	
	double *FE,FEmin,FEmax;
	int *empty_bins_ind,empty_bins_count = 0;;
	
	FE = (double *)malloc(Ngrid*sizeof(double));
	empty_bins_ind = (int *)malloc(Ngrid*sizeof(int));
	
	FEmin = 1.0e12;
	FEmax = -1.0e12;
	for( j=0; j<Ngrid; j++ ) {
		if( bins[j] > 0 ) {
			FE[j] = -log((double)bins[j])/BETA - grid_pot[j];
			FEmin = min(FEmin,FE[j]);
			FEmax = max(FEmax,FE[j]);
		}
		else {
			empty_bins_ind[empty_bins_count] = j;
			empty_bins_count++;
		}
	}
	for( j=0; j<Ngrid; j++ ) FE[j] -= FEmin;
	FEmax -= FEmin;
	for( j=0; j < empty_bins_count; j++ ) {
		FE[empty_bins_ind[j]] = FEmax;
	}
	
	sprintf(fname,"Data/LJ7free_energy_beta%.0f.txt",BETA);
	fpot = fopen(fname,"w");
	for( j=0; j<N2; j++ ) {
		for( i=0; i<N1; i++ ) {
			ind = i + j*N1;
			fprintf(fpot,"%.4e\t",FE[ind]);
		}
		fprintf(fpot,"\n");
	}
	fclose(fpot);
	
	// smooth free energy
	sprintf(fname,"Data/LJ7free_energy_bicubic_matrix_beta%.0f.txt",BETA);
	fpot = fopen(fname,"w");
	FEders_on_grid(FE,grid_der1,grid_der2,grid_der12);
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(FE,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind);
			for( n=0; n < 16; n++ ) {
				fprintf(fpot,"%.4e\t",bicubic_matrix[ind*16 + n]);
			}	
			fprintf(fpot,"\n");
		}
	}
	fclose(fpot);
	
	return 0;
}


