// Generates sample positions for LJ7 in 2D using metadynamics.
// Biasing with respect to (mu2, mu3), the 2nd and 3rd moments of coordination numbers.
// MALA algorithm is used.
// The phase space is 14D.
// The positions and the heights of the bumps are saved.
// The samples are recorded at every NSKIP step.
// A total of NSAVE samples are recorded.

// Compile command:  gcc LJ7in2D_WTMetad_mu2mu3.c -lm -O3

// Rules-of-thumb for well-tempered metadynamics
// --> gamma should be approximately equal to the max 
// depth that needs to be filled with Gaussian bumps
// --> sigma should >= the size of features that we want to resolve
// --> height should be such that height*Nbumps*(2*pi*sigma^2)^{dim/2} \approx Volume to be filled

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BETA 5.0
#define GAMMA 1.0 // the artificial temperature for the discount factor
#define NSKIP 500  // save data every 10^6 steps
#define NSAVE 100000 // 10^5 time stes
#define PI 3.141592653589793
#define PI2 6.283185307179586 // 2*PI
#define RSTAR 1.122462048309373 // 2^{1/6}
#define TAU 5.0e-5
#define NATOMS 7 // the number of atoms
#define SIGMA 0.02 // width parameter for Gaussian bumps
#define HEIGHT 0.01 // height of Gaussian bump
#define NSTEPS_BETWEEN_DEPOSITS 500
#define NBUMPS_MAX 50000 // 10000 the number of Gaussian bumps to be deposited
#define KAPPA 100.0 // spring constant for the restraining potential that turns on 
// if an atom is at a distance more than 2 from the center of mass
#define mabs(a) ((a) >= 0 ? (a) : -(a))
#define sgn(a) ((a) == 0 ? 0 : ((a) > 0  ? 1 : -1 ))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))

struct vec2 {
	double x;
	double y;
};

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
void GaussBumps_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CV1,double *CV2,double *CV1grad,double *CV2grad,double *height,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot);
void total_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CV1,double *CV2,double *CV1grad,double *CV2grad,double *hight,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot);
void WTMetadynamics(int Nsteps_between_deposits,int Nbumps_max,int Natoms,
	double *conf0,double dt,double *val1,double *val2,double *height);
void generate_trajectory(FILE *fid,int Nskip,int Nsave,int Natoms,
	double *conf0,double dt,int Nbumps,double *val1,double *val2,double *height);	
void restraining_pot_and_grad(double *conf,double *pot,double *grad,int Natoms);
int main(void);

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
// 	test gradient of coordination numbers
// 	double coord_num_test[2][Natoms],conf_test[2*Natoms];
// 	int i,kk,jj;
// 	double dd = 1.0e-6,fd;
// 	
// 	for( k = 0; k < Natoms; k++ ) {
// 		printf("\nk = %i, coord_num[%i] = %.10e\n",k,k,coord_num[k]);
// 		for( j=0;j<dim;j++ ) {
// 			compute gradient of coordination numbers by finite difference
// 			for( kk=0; kk<Natoms; kk++ ) {
// 				coord_num_test[0][kk] = 0.0;
// 				coord_num_test[1][kk] = 0.0;	
// 				conf_test[kk] = conf[kk];
// 				conf_test[kk+Natoms] = conf[kk+Natoms]; 	
// 			}
// 			for( i = 0; i < 2; i++ ) {
// 				conf_test[j] = (i==0) ? conf[j] + dd : conf[j] - dd;
// 				for( kk=1; kk<Natoms; kk++ ) {	
// 					for( jj=0; jj<kk; jj++ ) {
// 						aux_x = conf_test[kk] - conf_test[jj];
// 						aux_y = conf_test[kk+Natoms] - conf_test[jj+Natoms];
// 						r2 = (aux_x*aux_x + aux_y*aux_y)/sigma2;
// 						r8 = pow(r2,4);
// 						r16 = r8*r8;
// 						aux = (1.0 - r8)/(1.0 - r16);
// 						coord_num_test[i][kk] += aux;
// 						coord_num_test[i][jj] += aux;
// 					}
// 				}
// 			}	
// 			fd = 0.5*(coord_num_test[0][k] - coord_num_test[1][k])/dd;
// 			printf("grad_coord_num[%i][%i] = %.10e, fdiff = %.10e\n",k,j,grad_coord_num[k][j],fd);
// 		}			
// 	}
	
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
// 	exit(1);
}

//------------------------------------------------------------
// Compute the gradient of Gaussian bumps
// V(conf) = LJpot(conf) + \sum_{j=0}^{Nbumps-1} h_j*
// exp(-0.5*{[CV1(conf) - val1_j]^2 + (CV2(conf) - val2_j)^2]}/sigma^2 )

void GaussBumps_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CV1,double *CV2,double *CV1grad,double *CV2grad,double *height,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot) {
	double bump,aux1,aux2;
	int j,k;
	int dim = 2*Natoms;
	
	// evaluate CVs and their gradients
	CVgrad(conf,CV1,CV2,CV1grad,CV2grad,Natoms);
	// compute the output gradient
	*biasing_pot = 0.0;
	for( j=0; j<Nbumps; j++ ) {
		aux1 = *CV1 - val1[j];
		aux2 = *CV2 - val2[j];
		bump = height[j]*exp(-0.5*(aux1*aux1 + aux2*aux2)/sig2);
		*biasing_pot += bump;
		for( k=0; k<dim; k++ ) {
			grad[k] += -bump*(aux1*CV1grad[k] + aux2*CV2grad[k])/sig2;
		}
	}
	*pot += *biasing_pot;
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

void total_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CV1,double *CV2,double *CV1grad,double *CV2grad,double *hight,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot) {

	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
	GaussBumps_pot_and_grad(conf,Natoms,val1,val2,CV1,CV2,CV1grad,CV2grad,hight,
			sig2,Nbumps,pot,grad,biasing_pot);	
}

//------------------------------------------------------------
void WTMetadynamics(int Nsteps_between_deposits,int Nbumps_max,int Natoms,
	double *conf0,double dt,double *val1,double *val2,double *height) {
	int n,k,dim = Natoms*2;
	int Nbumps = 0;
	struct vec2 gauss01;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *biasing_pot0,*biasing_pot1;
	double *CV1,*CV2,*CV1grad,*CV2grad;
	double sig2 = SIGMA*SIGMA;
	struct vec2 cv;
	char ch;
	
	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	biasing_pot0 = (double *)malloc(sizeof(double));
	biasing_pot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	CV1 = (double *)malloc(sizeof(double));
	CV2 = (double *)malloc(sizeof(double));
	CV1grad = (double *)malloc(dim*sizeof(double));
	CV2grad = (double *)malloc(dim*sizeof(double));
	
	total_pot_and_grad(conf0,Natoms,val1,val2,CV1,CV2,CV1grad,CV2grad,height,
			sig2,Nbumps,Vpot0,Vgrad0,biasing_pot0);
	for( Nbumps = 0; Nbumps<Nbumps_max; Nbumps++ ) {	
		if( Nbumps%100 == 0) printf("Nbumps = %i\n",Nbumps);	
		for( n=0; n < Nsteps_between_deposits; n++ ) {
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
			total_pot_and_grad(conf1,Natoms,val1,val2,CV1,CV2,CV1grad,CV2grad,height,
					sig2,Nbumps,Vpot1,Vgrad1,biasing_pot1);
			ch = MALAstep(conf0,conf1,Natoms,dt,Vpot0,Vpot1,Vgrad1,w);
			if( ch == 1 ) { // step was accepted
				// align configurations
				align(conf0,conf1,Natoms);
				for( k=0; k<dim; k++ ) {
					conf0[k] = conf1[k];
					Vgrad0[k] = Vgrad1[k];
				}
				*Vpot0 = *Vpot1;
				*biasing_pot0 = *biasing_pot1;		
			}
		}
		height[Nbumps] = HEIGHT*exp(-(*biasing_pot0)/GAMMA);
		cv = CVs(conf0,Natoms);
		val1[Nbumps] = cv.x;
		val2[Nbumps] = cv.y;
	}
}

//------------------------------------------------------------
void generate_trajectory(FILE *fid,int Nskip,int Nsave,int Natoms,
	double *conf0,double dt,int Nbumps,double *val1,double *val2,double *height) {
	int jsave,n,k,dim = Natoms*2;
	struct vec2 gauss01;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *biasing_pot0,*biasing_pot1;
	double *CV1,*CV2,*CV1grad,*CV2grad;
// 	double *CV1aux,*CV2aux;
	double sig2 = SIGMA*SIGMA;
	struct vec2 cv;
	char ch;
	
	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	biasing_pot0 = (double *)malloc(sizeof(double));
	biasing_pot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	CV1 = (double *)malloc(sizeof(double));
	CV2 = (double *)malloc(sizeof(double));
	CV1grad = (double *)malloc(dim*sizeof(double));
	CV2grad = (double *)malloc(dim*sizeof(double));
	
// 	// test the calculation of the gradient
// 	CV1aux = (double *)malloc(sizeof(double));
// 	CV2aux = (double *)malloc(sizeof(double));
// 	total_pot_and_grad(conf0,Natoms,val1,val2,CV1,CV2,CV1grad,CV2grad,height,
// 			sig2,Nbumps,Vpot0,Vgrad0);
// 	for( k=0; k<dim; k++ ) printf("grad[%i] = %.10e, CV1grad = %.10e, CV2grad = %.10e\n",k,Vgrad0[k],CV1grad[k],CV2grad[k]);		
// 	for( k = 0; k < dim; k++ ) {
// 		for( n=0; n<dim; n++ ) conf1[n] = conf0[n];
// 		conf1[k] = conf0[k] + 1.0e-6;
// 		total_pot_and_grad(conf1,Natoms,val1,val2,CV1,CV2,CV1grad,CV2grad,height,
// 			sig2,Nbumps,Vpot0,Vgrad0);
// 		for( n=0; n<dim; n++ ) conf1[n] = conf0[n];
// 		conf1[k] = conf0[k] - 1.0e-6;
// 		total_pot_and_grad(conf1,Natoms,val1,val2,CV1aux,CV2aux,CV1grad,CV2grad,height,
// 			sig2,Nbumps,Vpot1,Vgrad0);
// 		printf("finite difference: grad[%i] = %.10e, CV1grad = %.10e, CV2grad = %.10e\n",k,
// 				(*Vpot0 - *Vpot1)/2.0e-6,(*CV1 - *CV1aux)/2.0e-6,(*CV2 - *CV2aux)/2.0e-6);
// 	}		
// 			
// 				
	total_pot_and_grad(conf0,Natoms,val1,val2,CV1,CV2,CV1grad,CV2grad,height,
			sig2,Nbumps,Vpot0,Vgrad0,biasing_pot0);
	
		
			
	for( jsave = 0; jsave<Nsave; jsave++ ) {		
		for( n=0; n < Nskip; n++ ) {
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
			total_pot_and_grad(conf1,Natoms,val1,val2,CV1,CV2,CV1grad,CV2grad,height,
					sig2,Nbumps,Vpot1,Vgrad1,biasing_pot1);
			ch = MALAstep(conf0,conf1,Natoms,dt,Vpot0,Vpot1,Vgrad1,w);
			if( ch == 1 ) { // step was accepted
				// align configurations
				align(conf0,conf1,Natoms);
				for( k=0; k<dim; k++ ) {
					conf0[k] = conf1[k];
					Vgrad0[k] = Vgrad1[k];
				}
				*Vpot0 = *Vpot1;
				*biasing_pot0 = *biasing_pot1;		
			}
		}
		cv = CVs(conf0,Natoms);
		for( k=0; k<dim; k++ ) fprintf(fid,"%.10e\t",conf0[k]);
// 		fprintf(fid,"%.10e\t%.10e\t",cv.x,cv.y);
		fprintf(fid,"\n");
		if(jsave%100 == 0) printf("%i steps are done\n",jsave);
	}
}

//------------------------------------------------------------
int main(void){
	int Nsteps_between_deposits = NSTEPS_BETWEEN_DEPOSITS,Nbumps_max = NBUMPS_MAX;
	int Natoms = NATOMS,dim,jcall,n,k;
	int Nskip = NSKIP, Nsave = NSAVE;
	double dt = TAU;
	double *conf0;
	double *height,*val1,*val2;
	FILE *fpot;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    char ftraj_name[100],fbump_name[100];

	dim = 2*Natoms;
	conf0 = (double *)malloc(dim*sizeof(double));
	val1 = (double *)malloc(Nbumps_max*sizeof(double));
	val2 = (double *)malloc(Nbumps_max*sizeof(double));
	height = (double *)malloc(Nbumps_max*sizeof(double));
	
	sprintf(fbump_name,"GbumpsData/GaussianBumps_beta%.f.txt",BETA); 

	fpot = fopen(fbump_name,"w");
	
	init_conf(conf0,Natoms);
	
	CPUbegin=clock(); // start time measurement

	WTMetadynamics(Nsteps_between_deposits,Nbumps_max,Natoms,
		conf0,dt,val1,val2,height);
	for( k=0; k<Nbumps_max; k++ ) {
		fprintf(fpot,"%.10e\t%.10e\t%.10e\n",height[k],val1[k],val2[k]);
	}
	
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("CPU time = %g\n",cpu);
	
	fclose(fpot);
	
	return 0;
}
