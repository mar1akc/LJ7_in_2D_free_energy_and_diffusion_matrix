// This code does the following three things.
//
// (1) Reads parameter file and binning data for computing free energy.
//
// (2) Computes the diffusion matrix at the centers of all nonempty bins 
// using harmonic biasing potential U = U0 + 0.5*Kdiffmatr(||CV - CV(cell_center)||^2)
// and the formula 
// M_{i,j} = (1/Nsteps)sum_steps sum_k=1^{dim} (dCV_i(step)/dx_k)(dCV_j(step)/dx_k).
//
// (3) Builds a bicubic interpolant for M and its derivatives

// Compile command:  gcc LJ7in2D_diffusion_matrix.c -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BETA 10.0

#define NSTEPS 1e6  // the length of the stochastic trajectory that we bin

#define PI 3.141592653589793
#define PI2 6.283185307179586 // 2*PI
#define RSTAR 1.122462048309373 // 2^{1/6}

#define TAU 1.0e-5;
#define NATOMS 7 // the number of atoms
#define KAPPA 100.0 // spring constant for the restraining potential that turn on 
// if an atom is at distance more than 2 from the center of mass
#define BKAPPA 500.0 // the spring constant for the biasing potential 
// attaching the configuration to particular values of CVs
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
void compute_bicubic_coeff_matrix(double *pot,double *der1,double *der2,double *der12,
	double *Amatr,int ind,int N1);
double wsum0(double a,double b,double c,double d);	
double wsum1(double a,double b,double c,double d);	
double wsum2(double a,double b,double c,double d);	
double wsum3(double a,double b,double c,double d);	
void evaluate_Gpot_and_ders(double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *Amatr,double cv1,double cv2,double *FEval,double *FEder1,double *FEder2,
	int N1,int N2);
void derivatives_on_grid(double *fun,double *der1,double *der2,double *der12,int N1,int N2);
//----- computing the diffusion matrix
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
void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad,
	double *CV1,double *CV2,double *CV1grad,double *CV2grad,
	double CV1star,double CV2star,double Bkappa);
void diffusion_matrix(int Natoms,double *conf0,double dt,int Nsteps,
	double CV1val,double CV2val,int ind,double *M11,double *M12,double *M22);
void prepare_conf(int Natoms,double *conf0,double CV1val,double CV2val,
		double dt,int Nsteps);

//----- main	
int main(void);

//-------------------------------------------------------------

//----------------------
void compute_bicubic_coeff_matrix(double *pot,double *der1,double *der2,double *der12,
	double *Amatr,int ind,int N1) {
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
	double *Amatr,double cv1,double cv2,double *FEval,double *FEder1,double *FEder2,
	int N1,int N2) {
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
	double CV1star,double CV2star,double Bkappa) {
	
	int j,dim = Natoms*2;
	double CV1diff,CV2diff;

	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
	CVgrad(conf,CV1,CV2,CV1grad,CV2grad,Natoms);			
	// add the effect of the biasing potential
	// B(x) = 0.5*Bkappa*[(CV1(x) - CV1val)^2 + (CV2(x) - CV2val)^2 )
	CV1diff = *CV1 - CV1star;
	CV2diff = *CV2 - CV2star;
	for( j=0; j<dim; j++ ) {
		grad[j] += Bkappa*(CV1diff*CV1grad[j] + CV2diff*CV2grad[j]);
	}	
	*pot += 0.5*Bkappa*(CV1diff*CV1diff + CV2diff*CV2diff);	
// 	printf("CV1 = %.4e, CV2 = %.4e\n",*CV1,*CV2);
// 	printf("FEpot = %.4e,FEder1 = %.4e,FEder2 = %.4e\n",*FEpot,*FEder1,*FEder2);
		
}
//------------------------------------------------------------
// 	diffusion_matrix(Natoms,conf,dt,Nsteps,grid_CV1[i],grid_CV2[j],ind,M11,M12,M22)

void diffusion_matrix(int Natoms,double *conf0,double dt,int Nsteps,
	double CV1val,double CV2val,int ind,double *M11,double *M12,double *M22) {

	int j,n,k,dim = Natoms*2,j1,j2;
	struct vec2 gauss01;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *CV1,*CV2,*CV1grad,*CV2grad;
	double M11update,M12update,M22update;
	struct vec2 cv;
	char ch;
	double Bkappa = BKAPPA;
	
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
	
	total_pot_and_grad(conf0,Natoms,Vpot0,Vgrad0,CV1,CV2,CV1grad,CV2grad,
		CV1val,CV2val,Bkappa);
				
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
			CV1val,CV2val,Bkappa);
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
		// update the values of the diffusion matrix
		M11update = 0.0;
		M12update = 0.0;
		M22update = 0.0;
		for( k = 0; k < dim; k++ ) {
			M11update += CV1grad[k]*CV1grad[k];
			M12update += CV1grad[k]*CV2grad[k];
			M22update += CV2grad[k]*CV2grad[k];
		}
		M11[ind] = (M11[ind]*j + M11update)/(j+1);
		M12[ind] = (M12[ind]*j + M12update)/(j+1);
		M22[ind] = (M22[ind]*j + M22update)/(j+1);
		if( j%100000 == 0 ) {
			printf("M11 = %.4e, M12 = %.4e, M22 = %.4e\n",M11[ind],M12[ind],M22[ind]);
		}
	}
}

//-----------------------------------------------------------

void derivatives_on_grid(double *fun,double *der1,double *der2,double *der12,int N1,int N2) {

	int n1m1 = N1-1, n2m1 = N2-1;
	int i,j,ind;

	// interior grid points
	for( i = 1; i < n1m1; i++ ) {
		for( j = 1; j < n2m1; j++ ) {
			ind = i + j*N1;	
			der1[ind] = 0.5*(fun[ind+1]-fun[ind-1]);
			der2[ind] = 0.5*(fun[ind+N1]-fun[ind-N1]);

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
		der1[ind] = fun[ind+1] - fun[ind];
		der2[ind] = 0.5*(fun[ind+N1]-fun[ind-N1]);
		// i = n1m1
		ind = j*N1+n1m1;
		der1[ind] = fun[ind] - fun[ind-1];
		der2[ind] = 0.5*(fun[ind+N1]-fun[ind-N1]);
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
		der1[ind] = 0.5*(fun[ind+1]-fun[ind-1]);
		der2[ind] = fun[ind+N1]-fun[ind];
		// j = n2m1
		ind = n2m1*N1+i;
		der1[ind] = 0.5*(fun[ind+1]-fun[ind-1]);
		der2[ind] = fun[ind]-fun[ind-N1];
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
	der1[0] = fun[1] - fun[0];
	der2[1] = fun[N1] - fun[0];
	der12[0] = der1[N1] - der1[0];
	// i = n1m1; j = 0;
	der1[n1m1] = fun[n1m1] - fun[n1m1-1];
	der2[n1m1] = fun[N1+n1m1] - fun[n1m1];
	der12[n1m1] = der1[N1+n1m1] - der1[n1m1];
	// i = 0; j = n2m1;
	ind = N1*n2m1;
	der1[ind] = fun[ind+1] - fun[ind];
	der2[ind] = fun[ind] - fun[ind-N1];
	der12[ind] = der1[ind] - der1[ind-N1];
	// i = n1m1; j = n2m1;
	ind =  n1m1 + N1*n2m1;
	der1[ind] = fun[ind] - fun[ind-1];
	der2[ind] = fun[ind] - fun[ind-N1];
	der12[ind] = der1[ind] - der1[ind-N1];
}






//------------------------------------------------------------
int main(void){
	int N1, N2, Nsteps = NSTEPS, Natoms = NATOMS, Ngrid;
	int i,j,k,ind,n,dim=NATOMS*2;
	double h1,h2;
	double *val1_min,*val1_max,*val2_min,*val2_max;
	double *grid_CV1,*grid_CV2;
	double *conf0,*conf;
	double *bicubic_matrix;
	double *M11,*M12,*M22;
	double *grid_der1,*grid_der2,*grid_der12;
	FILE *fM11,*fM12,*fM22,*fder1,*fder2,*fder12,*fpar,*fconf;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    double dt = TAU;
	char *bins;
	char fname[100];

	val1_min = (double *)malloc(sizeof(double));
	val1_max = (double *)malloc(sizeof(double));
	val2_min = (double *)malloc(sizeof(double));
	val2_max = (double *)malloc(sizeof(double));

	// save the parameters and the bicubic matrix 
	fpar = fopen("Data/bicubic_params.txt","r");
	fscanf(fpar,"%i\n",&N1);
	fscanf(fpar,"%i\n",&N2);
	fscanf(fpar,"%le\n",&h1);
	fscanf(fpar,"%le\n",&h2);
	fscanf(fpar,"%le\n",val1_min);
	fscanf(fpar,"%le\n",val1_max);
	fscanf(fpar,"%le\n",val2_min);
	fscanf(fpar,"%le\n",val2_max);
	fclose(fpar);
	// print read values
	printf("val1_min = %.4e\n",*val1_min);
	printf("val1_max = %.4e\n",*val1_max);
	printf("val2_min = %.4e\n",*val2_min);
	printf("val2_max = %.4e\n",*val2_max);
	printf("h1 = %.4e\n",h1);
	printf("h2 = %.4e\n",h2);
	printf("N1 = %i\n",N1);
	printf("N2 = %i\n",N2);

	grid_CV1 = (double *)malloc(N1*sizeof(double));
	grid_CV2 = (double *)malloc(N2*sizeof(double));

	for( i=0; i<N1; i++ ) {
		grid_CV1[i] = *val1_min + h1*i;		
	}
	for( i=0; i<N2; i++ ) {
		grid_CV2[i] = *val2_min + h2*i;		
	}
	
	conf0 = (double *)malloc(dim*sizeof(double));
	
	Ngrid = N1*N2;
	M11 = (double *)malloc(Ngrid*sizeof(double));
	M12 = (double *)malloc(Ngrid*sizeof(double));
	M22 = (double *)malloc(Ngrid*sizeof(double));
	
	// Compute the diffusion matrix at all nonempty bin centers
 	CPUbegin=clock(); // start time measurement
 	double confrad = 0.0,TOL = 1.0e-10;
 	fconf = fopen("Data/LJ7bins_confs.txt","r");
 	bins = (char *)malloc(Ngrid*sizeof(char)); 	 	
	for( i=0; i<N1; i++ ) {
		for( j=0; j<N2; j++ ) {
			ind = i + j*N1;
			confrad = 0.0;
			bins[ind] = 0;
			for( k = 0; k < dim; k++ ) {
				fscanf(fconf,"%le\t",conf0+k);
				confrad += conf0[k]*conf0[k];				
			}
			fscanf(fconf,"\n");
			if( confrad > TOL ) {
				bins[ind] = 1;
				printf("bin (%i,%i)\n",i,j);
				// Compute the diffusion matrix
				diffusion_matrix(Natoms,conf0,dt,Nsteps,grid_CV1[i],grid_CV2[j],
					ind,M11,M12,M22);
			}
		}
	}
	fclose(fconf);
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("Computation of the diffusion matrix: CPU time = %g\n",cpu);

	// Do nearest-neighbor interpolation
 	CPUbegin=clock(); // start time measurement
	double aux1,aux2,d0,d1;
	d0 = INFTY;
	d1 = min(h1,h2);
	int ind1;
	
	for( i=0; i<N1; i++ ) {
		for( j=0; j<N2; j++ ) {
			ind = i + j*N1;
			if( bins[ind] == 0 ) {				
				d0 = INFTY;
				d1 = min(h1,h2);
				for( k = 0; k < N1; k++ ) {
					for( n = 0; n < N2; n++ ) {
						ind1 = k+n*N1;
						if( bins[ind1] > 0 ) {
							aux1 = h1*(i-k);
							aux2 = h2*(j-n);
							d1 = sqrt(aux1*aux1+aux2*aux2);
							if( d1 < d0 ) {
								d0 = d1;
								M11[ind] = M11[ind1];
								M12[ind] = M12[ind1];
								M22[ind] = M22[ind1];
							}						
						}
					}
				}			
			}
		}
	}
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("Nearest neighbor interpolation: CPU time = %g\n",cpu);
	
	// Save the diffusion matrix
	sprintf(fname,"Data/LJ7_M11_beta%.0f.txt",BETA);
	fM11 = fopen(fname,"w");
	sprintf(fname,"Data/LJ7_M12_beta%.0f.txt",BETA);
	fM12 = fopen(fname,"w");
	sprintf(fname,"Data/LJ7_M22_beta%.0f.txt",BETA);
	fM22 = fopen(fname,"w");
		
	for( j= 0; j<N2; j++ ){
		for( i=0; i<N1; i++ ) {
			ind = i + N1*j;
			fprintf(fM11,"%.4e\t",M11[ind]);
			fprintf(fM12,"%.4e\t",M12[ind]);
			fprintf(fM22,"%.4e\t",M22[ind]);
		}
		fprintf(fM11,"\n");
		fprintf(fM12,"\n");
		fprintf(fM22,"\n");
	}	
	fclose(fM11);
	fclose(fM12);
	fclose(fM22);
		
	// Compute the bicubic matrices for the components of the diffusion matrix
	sprintf(fname,"Data/LJ7_M11_bicubic_matrix_beta%.0f.txt",BETA);
	fM11 = fopen(fname,"w");
	sprintf(fname,"Data/LJ7_M12_bicubic_matrix_beta%.0f.txt",BETA);
	fM12 = fopen(fname,"w");
	sprintf(fname,"Data/LJ7_M22_bicubic_matrix_beta%.0f.txt",BETA);
	fM22 = fopen(fname,"w");

	grid_der1 = (double *)malloc(Ngrid*sizeof(double));
	grid_der2 = (double *)malloc(Ngrid*sizeof(double));
	grid_der12 = (double *)malloc(Ngrid*sizeof(double));
	bicubic_matrix = (double *)malloc(Ngrid*dim*sizeof(double));

	derivatives_on_grid(M11,grid_der1,grid_der2,grid_der12,N1,N2);
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(M11,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind,N1);
			for( n=0; n < 16; n++ ) {
				fprintf(fM11,"%.4e\t",bicubic_matrix[ind*16 + n]);
			}	
			fprintf(fM11,"\n");
		}
	}
	fclose(fM11);

	derivatives_on_grid(M12,grid_der1,grid_der2,grid_der12,N1,N2);
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(M12,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind,N1);
			for( n=0; n < 16; n++ ) {
				fprintf(fM12,"%.4e\t",bicubic_matrix[ind*16 + n]);
			}	
			fprintf(fM12,"\n");
		}
	}
	fclose(fM12);

	derivatives_on_grid(M22,grid_der1,grid_der2,grid_der12,N1,N2);
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(M22,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind,N1);
			for( n=0; n < 16; n++ ) {
				fprintf(fM22,"%.4e\t",bicubic_matrix[ind*16 + n]);
			}	
			fprintf(fM22,"\n");
		}
	}
	fclose(fM22);
	
	return 0;
}

