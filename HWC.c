/**********************************************************************
  HWC.c: (based on polB.c)

  code for calculating hybrid Wannier centers (HWCs) and the electric
  layer polarization of bulk system
  using parallel transport construction.

  Log of HWC.c:

  10/Mar/2016   Released by Naoya Yamaguchi

  polB.c:

  code for calculating the electric polarization of bulk system using 
  Berry's phase.

  Log of polB.c:

  30/Nov/2005   Released by Taisuke Ozaki
  27/Feb/2006   Modified by Fumiyuki Ishii
  28/July/2006  Modified for MPI by Fumiyuki Ishii
  19/Jan/2007   Modified by Taisuke Ozaki
 ***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include "read_scfout.h"
#include "lapack_prototypes.h"

/* Added by N. Yamaguchi for HWC */
#include <string.h>
#define MAX_LINE_SIZE 256
#define fp_bsize         1048576     /* buffer size for setvbuf */
void zgeev_(char *jobvl, char *jobvr, int *n,
    dcomplex *a, int *lda,
    dcomplex *w, dcomplex *vl,
    int *ldvl, dcomplex *vr, int *ldvr, dcomplex *work,
    int *lwork, double *rwork, int *info);

#include "f77func.h"

#ifdef nompi
#include "mimic_mpi.h"
#else
#include "mpi.h"
#endif

#define Host_ID       0         /* ID of the host CPU in MPI */

#define printout  0    /* 0:off, 1:on */
#define PI   3.1415926535897932384626
#define measure_time   0
#define dste_flag      2
#define AU2Debye   2.54174776
#define AU2Mucm    5721.52891433 /* 100 e/bohr/bohr */
/* Added by N. Yamaguchi for HWC */
#define Bohr2Ang   0.52917721092

struct timeval2 {
  long tv_sec;    /* second */
  long tv_usec;   /* microsecond */
};

static void Overlap_k1k2(
    int diag_flag,
    double k1[4], double k2[4],
    int spinsize,
    int fsize, int fsize2, int fsize3, int fsize4,
    int *MP,
    dcomplex ***Sop, dcomplex ***Wk1, dcomplex ***Wk2,
    double **EigenVal1, double **EigenVal2);

static void Eigen_HH(dcomplex **ac, double *ko, int n, int EVmax);
static void lapack_dstevx2(INTEGER N, double *D, double *E, double *W, dcomplex **ev);
static void Overlap_Band(double ****OLP,
    dcomplex **S,int *MP,
    double k1, double k2, double k3);
static void Hamiltonian_Band(double ****RH, dcomplex **H, int *MP,
    double k1, double k2, double k3);
static void Hamiltonian_Band_NC(double *****RH, double *****IH,
    dcomplex **H, int *MP,
    double k1, double k2, double k3);
static void determinant( int spin, int N, dcomplex **a, INTEGER *ipiv, dcomplex *a1d,
    dcomplex *work, dcomplex Cdet[2] );

static void dtime(double *t);
static double tmp[4];

void LU_fact(int n, dcomplex **a);
dcomplex RCdiv(double x, dcomplex a);
dcomplex Csub(dcomplex a, dcomplex b);
dcomplex Cmul(dcomplex a, dcomplex b);

void Cross_Product(double a[4], double b[4], double c[4]);
double Dot_Product(double a[4], double b[4]);

int main(int argc, char *argv[])
{
  int fsize,fsize2,fsize3,fsize4;
  int spin,spinsize,diag_flag;
  /* int i,j,k,hos,po,wan,hog2,valnonmag; Disabled by N. Yamaguchi */
  /* Added by N. Yamaguchi ***/
  int i, j, k, *hos, NOS1, NOS2, upToOCC, po, wan, hog2, valnonmag;
  /* ***/
  int n1,n2,n3,i1,i2,i3;
  int Nk[4],kloop[4][4];
  int pflag[4];
  int dpflag[1];
  int hog[1];
  int bbnumb[11],bbnumbp1;
  int bbind,ii2,bandg,bandgsize,tband,bandgsizep1;
  int tbandi,tbandj,bandgm;
  int metal;
  double k1[4],k2[4];
  double pol_abc[4];
  double **polbb_abc;
  double Edpx,Edpy,Edpz,epa,epb,epc;
  double Edpbbx,Edpbby,Edpbbz,SumDbbx, SumDbby, SumDbbz;
  double Cdpx,Cdpy,Cdpz,Cdpi[4];
  double Tdpx,Tdpy,Tdpz;
  double Bdpx,Bdpy,Bdpz;
  double Ptx,Pty,Ptz;
  double Pex,Pey,Pez;
  double Pcx,Pcy,Pcz;
  double Pbx,Pby,Pbz;
  double Pbbex,Pbbey,Pbbez,SumPbbx,SumPbby,SumPbbz;
  double AbsD,Pion0;
  double Gabs[4],Parb[4];
  double Phidden[4], Phidden0;
  double tmpr,tmpi;
  double mulr[2],muli[2];
  double *tmpbbr, *tmpbbi;
  double **mulbbr,**mulbbi;
  double tmpSr, tmpSi;
  double **kg;
  double sum,d;
  double norm;
  double pele,piony,pall;
  double gdd;
  double CellV;
  double Cell_Volume;
  double psi,sumpsi[2],sumbbpsi,psip,psim,dpsi;
  double detr;
  double TZ;
  int ct_AN,h_AN;
  char *s_vec[20];
  int *MP;
  double **EigenVal1;
  double **EigenVal2;
  double **EigenVal3;
  dcomplex ***Sop;
  dcomplex ***Sopbb;
  dcomplex ***Wk1;
  dcomplex ***Wk2;
  dcomplex ***Wk3;
  MPI_Comm comm1;
  int numprocs,myid,ID,ID1;
  double TStime,TEtime;

  dcomplex *work1;   /* for determinant */
  dcomplex *work2;   /* for determinant */
  dcomplex Cdet[2];  /* for determinant */
  INTEGER *ipiv;     /* for determinant */
  /* for MPI*/
  int *arpo;
  int AB_knum,S_knum,E_knum,num_ABloop0;
  int ik1,ik2,ik3,ABloop,ABloop0,abcount;
  double tmp4;
  double *psiAB;
  double **psibbAB;
  int *AB_Nk2, *AB_Nk3, *ABmesh2ID;

  /* Added by N. Yamaguchi for HWC */

  dcomplex **Lambda;
  dcomplex *Sopin, *Lout;
  dcomplex alpha, beta;
  alpha.r = 1.0;
  alpha.i = 0.0;
  beta.r = 0.0;
  beta.i = 0.0;
  dcomplex *evout;
  dcomplex *vl, *vr;
  int Dummy;
  dcomplex *Work;
  int lWork;
  double *rWork;
  int Info;
  double **wfcllAB;
  double *wfcll;
  dcomplex *U, *VT;
  double *Sigma;
  int wfc_flag, wfc_flagp1;
  double lc1d;
  int svd_flag;
  dcomplex *work_svd;
  double *rwork_svd;
  double **wfcAB;
  double *wfc;
  int *wfcll_num;
  int **wfcllAB_num;
  int *wfcll_gap;
  double **dwfcAB;
  int lp_flag=0, lp_flagp1;
  double *wfc_lp;
  int *wfcll_wall;
  int *wfcll_layer;
  double *cc_lp;
  double *wfc_lp_mod;
  double *cc_lp_mod;

  /* MPI initialize */

  MPI_Status stat;
  MPI_Request request;
  MPI_Init(&argc, &argv);
  comm1 = MPI_COMM_WORLD;
  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  dtime(&TStime);
  if (myid==Host_ID){
    printf("\n******************************************************************\n");
    printf("******************************************************************\n");
    printf(" HWC (polB):\n"); 
    printf(" code for calculating the electric polarization,       \n");
    printf(" hybrid Wannier centers (HWCs) and the electric        \n");
    printf(" layer polarization of bulk system\n");
    printf(" Copyright (C), 2006-2016,\n"                             );
    printf(" Naoya Yamaguchi, Fumiyuki Ishii and Taisuke Ozaki     \n");
    printf(" This is free software, and you are welcome to         \n");
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("******************************************************************\n");
    printf("******************************************************************\n");
  }

  metal = 0;

  /******************************************
    read a scfout file
   ******************************************/

  read_scfout(argv);

  s_vec[0]="Recursion"; s_vec[1]="Cluster"; s_vec[2]="Band";
  s_vec[3]="NEGF";      s_vec[4]="DC";      s_vec[5]="GDC";
  s_vec[6]="Cluster2";  s_vec[7]="Krylov";

  if (myid==Host_ID){
    printf(" Previous eigenvalue solver = %s\n",s_vec[Solver-1]);
    printf(" atomnum                    = %i\n",atomnum);
    printf(" ChemP                      = %15.12f (Hartree)\n",ChemP);
    printf(" E_Temp                     = %15.12f (K)\n",E_Temp);
    printf(" Total_SpinS                = %15.12f (K)\n",Total_SpinS);
  }

  s_vec[0]="collinear spin-unpolarized";
  s_vec[1]="collinear spin-polarized";
  s_vec[3]="non-collinear";
  if (myid==Host_ID){
    printf(" Spin treatment             = %s\n",s_vec[SpinP_switch]);
  }

  /******************************************
    find the size of the full matrix
   ******************************************/

  /* MP:
     a pointer which shows the starting number
     of basis orbitals associated with atom i
     in the full matrix */

  MP = (int*)malloc(sizeof(int)*(atomnum+1)); 
  arpo = (int*)malloc(sizeof(int)*numprocs);

  fsize = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = fsize;
    fsize += Total_NumOrbs[i];
  }
  fsize--;

  /******************************************
    allocation arrays
   ******************************************/
  if      (SpinP_switch==0){ spinsize=1; fsize2=fsize;  fsize3=fsize+2; fsize4=Valence_Electrons/2;}
  else if (SpinP_switch==1){ spinsize=2; fsize2=fsize;  fsize3=fsize+2;
    fsize4=Valence_Electrons/2+abs(floor(Total_SpinS))*2+1;}
  else if (SpinP_switch==3){ spinsize=1; fsize2=2*fsize;fsize3=2*fsize+2; fsize4=Valence_Electrons;}

  /* if      (SpinP_switch==0){ spinsize=1; fsize2=fsize;  fsize3=fsize+2;  }
     else if (SpinP_switch==1){ spinsize=2; fsize2=fsize;  fsize3=fsize+2;  }
     else if (SpinP_switch==3){ spinsize=1; fsize2=2*fsize;fsize3=2*fsize+2;}*/

  /*if      (SpinP_switch==0){ spinsize=1; fsize2=Valence_Electrons/2;  fsize3=fsize+2;  }
    else if (SpinP_switch==1){ spinsize=2; fsize2=Valence_Electrons/2;  fsize3=fsize+2;  }
    else if (SpinP_switch==3){ spinsize=1; fsize2=Valence_Electrons;fsize3=2*fsize+2;}*/


  /* Sop:
     overlap matrix between one-particle wave functions
     calculated at two k-points, k1 and k2 */

  Sop = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    Sop[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3); 
    for (i=0; i<fsize3; i++){
      Sop[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3); 
      for (j=0; j<fsize3; j++){ Sop[spin][i][j].r=0.0; Sop[spin][i][j].i=0.0;}
    }
  }

  /* Wk1:
     wave functions at k1 */

  Wk1 = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    Wk1[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
    for (i=0; i<fsize3; i++){
      Wk1[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
      for (j=0; j<fsize3; j++){ Wk1[spin][i][j].r=0.0; Wk1[spin][i][j].i=0.0;}
    }
  }

  /* Wk2:
     wave functions at k2 */

  Wk2 = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    Wk2[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
    for (i=0; i<fsize3; i++){
      Wk2[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
      for (j=0; j<fsize3; j++){ Wk2[spin][i][j].r=0.0; Wk2[spin][i][j].i=0.0;}
    }
  }

  /* Wk3:
     wave functions for temporal storing */

  Wk3 = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    Wk3[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
    for (i=0; i<fsize3; i++){
      Wk3[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
      for (j=0; j<fsize3; j++){ Wk3[spin][i][j].r=0.0; Wk3[spin][i][j].i=0.0;}
    }
  }

  /* EigenVal1:
     eigenvalues at k1 */

  EigenVal1 = (double**)malloc(sizeof(double*)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    EigenVal1[spin] = (double*)malloc(sizeof(double)*fsize3);
    for (j=0; j<fsize3; j++) EigenVal1[spin][j] = 0.0;
  }

  /* EigenVal2:
     eigenvalues at k2 */

  EigenVal2 = (double**)malloc(sizeof(double*)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    EigenVal2[spin] = (double*)malloc(sizeof(double)*fsize3);
    for (j=0; j<fsize3; j++) EigenVal2[spin][j] = 0.0;
  }

  /* EigenVal3:
     eigenvalues for temporal storing */

  EigenVal3 = (double**)malloc(sizeof(double*)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    EigenVal3[spin] = (double*)malloc(sizeof(double)*fsize3);
    for (j=0; j<fsize3; j++) EigenVal3[spin][j] = 0.0;
  }

  /* for determinant */

  ipiv = (INTEGER*)malloc(sizeof(INTEGER)*fsize3);
  work1 = (dcomplex*)malloc(sizeof(dcomplex)*fsize3*fsize3);
  work2 = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);

  /* for storing the highest occupied state (Added by N. Yamaguchi) ***/
  hos=(int*)malloc(sizeof(int*)*spinsize);
  /* ***/

  /******************************************
    the standard output
   ******************************************/

  if (myid==Host_ID){
    /* printf("\n fsize4=%d\n",fsize4); */
    printf("\n r-space primitive vector (Bohr)\n");
    printf("  tv1=%10.6f %10.6f %10.6f\n",tv[1][1], tv[1][2], tv[1][3]);
    printf("  tv2=%10.6f %10.6f %10.6f\n",tv[2][1], tv[2][2], tv[2][3]);
    printf("  tv3=%10.6f %10.6f %10.6f\n",tv[3][1], tv[3][2], tv[3][3]);
    printf(" k-space primitive vector (Bohr^-1)\n");
    printf("  rtv1=%10.6f %10.6f %10.6f\n",rtv[1][1], rtv[1][2], rtv[1][3]);
    printf("  rtv2=%10.6f %10.6f %10.6f\n",rtv[2][1], rtv[2][2], rtv[2][3]);
    printf("  rtv3=%10.6f %10.6f %10.6f\n\n",rtv[3][1], rtv[3][2], rtv[3][3]);
  }

  Cross_Product(tv[2],tv[3],tmp);
  CellV = Dot_Product(tv[1],tmp); 
  Cell_Volume = fabs(CellV); 

  if (myid==Host_ID){
    printf("  Cell_Volume=%10.6f (Bohr^3)\n\n",Cell_Volume);
  }

  for (i=1; i<=3; i++){
    Gabs[i] = sqrt(rtv[i][1]*rtv[i][1]+rtv[i][2]*rtv[i][2]+rtv[i][3]*rtv[i][3]);
  }
  /******************************************
    input Nk1, Nk2, and Nk3
   ******************************************/


  po = 0;
  if (myid==Host_ID){
    printf("\n Specify the number of grids to discretize reciprocal a-, b-, and c-vectors\n");
  }


  do {
    if (myid==Host_ID){
      printf(" (e.g 2 4 3)  ");
      scanf("%d %d %d",&Nk[1],&Nk[2],&Nk[3]);
    }

    MPI_Bcast(Nk, 4, MPI_INT, 0, comm1); 

    if (1<=Nk[1] && 1<=Nk[2] && 1<=Nk[3]) po = 1;
    else {
      if (myid==Host_ID){
	printf(" The number of grids should be one or more\n");
      }
    }
  } while (po==0);

  kg = (double**)malloc(sizeof(double*)*4);
  kg[0] = (double*)malloc(sizeof(double)*1);
  for (i=1; i<=3; i++){
    kg[i] = (double*)malloc(sizeof(double)*(Nk[i]+1));
  }

  /* MPI_Barrier(comm1); */

  /* print & make  k-point */
  if (myid==Host_ID){
    printf("\n");
  }
  for (i=1; i<=3; i++){
    d = 1.0/(double)Nk[i];
    if (myid==Host_ID){
      printf("   k%i ",i);
    }
    for (j=0; j<=Nk[i]; j++){
      kg[i][j] = d*(double)j;
      if (myid==Host_ID){
	if (j!=Nk[i]) printf("%10.5f ",kg[i][j]);
      }
    }
    if (myid==Host_ID){
      printf("\n");
    }
  }
  if (myid==Host_ID){
    printf("\n");
  }


  /************************************************
    input calculate direction
   *************************************************/
  if (myid==Host_ID){
    printf("\n Specify the direction of polarization as reciprocal a-, b-, and c-vectors.\n");

    /* Added by N. Yamaguchi for HWC */
    printf(" (e.g 1 1 0 )  ");
    printf(" \n You have to specify only one direction to calculate Wannier function center.\n");

    printf(" (e.g 0 0 1 )  ");

    scanf("%d %d %d",&pflag[1],&pflag[2],&pflag[3]);
  }

  MPI_Bcast(pflag, 4, MPI_INT, 0, comm1); 

  /************************************************
    input set of bands for band by band decomposition (by F. Ishii)
   *************************************************/
  if (myid==Host_ID){

    /* for band by band F. Ishii */
    /* Disabled by N. Yamaguchi
    for (i=0; i<10; i++){
      bbnumb[i]=0;
    }
    */

    printf(" Do you want to decompose the electric polarization into separated bands? (only for nc and nonmag case. )\n");
    printf("    [N:Yes(number of separated bands), 0:No]");
    scanf("%d",&dpflag[0]);
    /*printf("dpflag[0]=%d",dpflag[0]); */

    if (dpflag[0] > 0){
      printf("\n For band decomposition, specify number of the each bands 1, n(1), 2, n(2), ...,N,n(N)\n");
      printf(" (e.g 1 N   ) ");

      for (i=0; i<10; i++){
	bbnumb[i]=0;
      }

      for (ii2=0; ii2<dpflag[0]; ii2++){
	scanf("%d %d",&bbind,&bbnumb[ii2]);
      }
    }

    /* Added by N. Yamaguchi for HWC */
    int wflag = 0;
    for (k=1; k<=3; k++){
      wflag += pflag[k]==1 ? 1 : 0;
    }

    if (wflag==1){
      printf(" \n Do you want to get 1D Wannier function centers?\n");
      printf("    [N:Yes(the number of windows to resolve Wannier function centers), 0:No(polB mode)]");
      scanf("%d",&wfc_flag);
      if (wfc_flag>0){
	printf("\n Do you omit the singular value composition of the overlap matrices between neighboring k points?\n");
	printf("    [1:Yes, 0:No(recommended)]");
	scanf("%d",&svd_flag);
	printf(" \n Do you want to calculate layer polarizations?\n");
	printf("    [N:Yes(the number of layers, more than the number of windows), 0:No(only normalized HWC)]");
	scanf("%d",&lp_flag);
      }
    } else {
      wfc_flag=0;
    }

  }

  MPI_Bcast(dpflag, 1, MPI_INT, 0, comm1);
  if (dpflag[0]>0){
    MPI_Bcast(bbnumb, dpflag[0], MPI_INT, 0, comm1);
  }

  /* Added by N. Yamaguchi for HWC */
  MPI_Bcast(&wfc_flag, 1, MPI_INT, 0, comm1);
  if (wfc_flag>0){
    MPI_Bcast(&svd_flag, 1, MPI_INT, 0, comm1);
    MPI_Bcast(&lp_flag, 1, MPI_INT, 0, comm1);
  }
  MPI_Barrier(comm1);


  /**** allocate Sopbb for band-by-band F. Ishii***/
  /* Sopbb:  F. Ishii added
     overlap matrix for band by band decomposition */
  if (dpflag[0]>0){
    bandgsize=dpflag[0];
    bandgsizep1=bandgsize+1;
    Sopbb = (dcomplex***)malloc(sizeof(dcomplex**)*bandgsize); 
    for (bandg=0; bandg<bandgsize; bandg++){
      bbnumbp1=bbnumb[bandg]+1;
      Sopbb[bandg] = (dcomplex**)malloc(sizeof(dcomplex*)*bbnumbp1); 
      for (i=0; i<bbnumbp1; i++){
	Sopbb[bandg][i] = (dcomplex*)malloc(sizeof(dcomplex)*bbnumbp1);
	for (j=0; j<bbnumbp1; j++){ Sopbb[bandg][i][j].r=0.0; Sopbb[bandg][i][j].i=0.0;}
      }
    }
  }


  /* for band-by-band F. Ishii */
  if (dpflag[0]>0){
    polbb_abc=(double**)malloc(sizeof(double*)*bandgsize);
    for (bandg=0; bandg<bandgsize; bandg++){
      polbb_abc[bandg] = (double*)malloc(sizeof(double)*4);
      for (j=0; j<4; j++) {polbb_abc[bandg][j] = 0.0; }
    }
    mulbbr=(double**)malloc(sizeof(double*)*bandgsize);
    mulbbi=(double**)malloc(sizeof(double*)*bandgsize);
    for (bandg=0; bandg<bandgsize; bandg++){
      mulbbr[bandg] = (double*)malloc(sizeof(double)*spinsize);
      mulbbi[bandg] = (double*)malloc(sizeof(double)*spinsize);
      for (j=0; j<spinsize; j++) {
	mulbbr[bandg][j] = 0.0;
	mulbbi[bandg][j] = 0.0;
      }
    }

    tmpbbr = (double*)malloc(sizeof(int)*bandgsize);
    tmpbbi = (double*)malloc(sizeof(int)*bandgsize);

  }

  /* Added by N. Yamaguchi for HWC */

  if (wfc_flag>0){
    wfc_flagp1 = wfc_flag + 1;
    Lambda = (dcomplex**)malloc(sizeof(dcomplex*)*spinsize);
    for (spin=0; spin<spinsize; spin++){
      Lambda[spin] = (dcomplex*)malloc(sizeof(dcomplex)*fsize4*fsize4);
    }
    Sopin = (dcomplex*)malloc(sizeof(dcomplex)*fsize4*fsize4);
    Lout = (dcomplex*)malloc(sizeof(dcomplex)*fsize4*fsize4);
    evout = (dcomplex*)malloc(sizeof(dcomplex)*fsize4);
    vl = (dcomplex*)malloc(sizeof(dcomplex)*fsize4);
    vr = (dcomplex*)malloc(sizeof(dcomplex)*fsize4);
    wfcll = (double*)malloc(sizeof(double)*wfc_flagp1);
    for (j=1; j<=wfc_flag; j++){wfcll[j] = 0.0;}
    wfc = (double*)malloc(sizeof(double)*fsize4);
    for (j=0; j<fsize4; j++){wfc[j] = 0.0;}
    if (svd_flag!=1){
      U = (dcomplex*)malloc(sizeof(dcomplex)*fsize4*fsize4);
      VT = (dcomplex*)malloc(sizeof(dcomplex)*fsize4*fsize4);
      work_svd = (dcomplex*)malloc(sizeof(dcomplex)*3*fsize4);
      rwork_svd = (double*)malloc(sizeof(double)*5*fsize4);
    }
    Work = (dcomplex*)malloc(sizeof(dcomplex)*2*fsize4);
    rWork = (double*)malloc(sizeof(double)*2*fsize4);
    Sigma = (double*)malloc(sizeof(double)*fsize4);
    wfcll_num = (int*)malloc(sizeof(int)*wfc_flagp1);
    for (j=1; j<=wfc_flag; j++){wfcll_num[j] = 0;}
    wfcll_gap = (int*)malloc(sizeof(int)*wfc_flagp1);
    lp_flagp1 = lp_flag + 1;
    wfc_lp = (double*)malloc(sizeof(double)*lp_flagp1);
    for (j=1; j<=lp_flag; j++){wfc_lp[j] = 0.0;}
    wfcll_wall = (int*)malloc(sizeof(int)*wfc_flagp1);
    for (j=1; j<=wfc_flag; j++){wfcll_wall[j] = 0;}
    wfcll_layer = (int*)malloc(sizeof(int)*wfc_flagp1);
    for (j=1; j<=wfc_flag; j++){wfcll_layer[j] = 0;}
    cc_lp = (double*)malloc(sizeof(double)*lp_flagp1);
    for (j=1; j<=lp_flag; j++){cc_lp[j] = 0.0;}
    wfc_lp_mod = (double*)malloc(sizeof(double)*lp_flagp1);
    for (j=1; j<=lp_flag; j++){wfc_lp_mod[j] = 0.0;}
    cc_lp_mod = (double*)malloc(sizeof(double)*lp_flagp1);
    for (j=1; j<=lp_flag; j++){cc_lp_mod[j] = 0.0;}
  }



  /************************************************
    calculate the electronic contribution of
    the macroscopic polarization
   *************************************************/

  /*Cross_Product(tv[2],tv[3],tmp);
    CellV = Dot_Product(tv[1],tmp);
    Cell_Volume = fabs(CellV);*/

  kloop[1][1] = 1;
  kloop[1][2] = 2;
  kloop[1][3] = 3;

  kloop[2][1] = 2;
  kloop[2][2] = 3;
  kloop[2][3] = 1;

  kloop[3][1] = 3;
  kloop[3][2] = 1;
  kloop[3][3] = 2;

  s_vec[1]="a-axis"; s_vec[2]="b-axis"; s_vec[3]="c-axis";

  for (k=1; k<=3; k++){

    if(pflag[k]==1){

      if (myid==Host_ID){
	printf("  \ncalculating the polarization along the %s ....\n\n",s_vec[k]);
      }

      n1 = kloop[k][1];
      n2 = kloop[k][2];
      n3 = kloop[k][3];

      for (spin=0; spin<spinsize; spin++){
	sumpsi[spin] = 0.0;
      }

      /* one-dimensionalize for MPI */

      AB_knum = 0;
      for (ik2=0; ik2<Nk[n2]; ik2++){
	for (ik3=0; ik3<Nk[n3]; ik3++){
	  AB_knum++;
	}
      }

      AB_Nk2= (int*)malloc(sizeof(int)*AB_knum);
      AB_Nk3= (int*)malloc(sizeof(int)*AB_knum);
      ABmesh2ID= (int*)malloc(sizeof(int)*AB_knum);
      psiAB = (double*)malloc(sizeof(double)*AB_knum);

      /* for band-by-band F. Ishii */
      if (dpflag[0]>0){
	psibbAB=(double**)malloc(sizeof(double*)*bandgsize);
	for (bandg=0; bandg<bandgsize; bandg++){
	  psibbAB[bandg] = (double*)malloc(sizeof(double)*AB_knum);
	  for (j=0; j<AB_knum; j++) psibbAB[bandg][j] = 0.0;
	}
      }

      /* Added by N. Yamaguchi for HWC */

      if (wfc_flag>0){
	wfcllAB = (double**)malloc(sizeof(double*)*AB_knum);
	wfcAB = (double**)malloc(sizeof(double*)*AB_knum);
	wfcllAB_num = (int**)malloc(sizeof(int*)*AB_knum);
	dwfcAB = (double**)malloc(sizeof(double*)*AB_knum);
	for (ABloop=0; ABloop<AB_knum; ABloop++){
	  wfcllAB[ABloop] = (double*)malloc(sizeof(double)*wfc_flagp1);
	  for (j=1; j<=wfc_flag; j++){wfcllAB[ABloop][j] = 0.0;}
	  wfcAB[ABloop] = (double*)malloc(sizeof(double)*fsize4);
	  dwfcAB[ABloop] = (double*)malloc(sizeof(double)*fsize4);
	  wfcllAB_num[ABloop] = (int*)malloc(sizeof(int)*wfc_flagp1);
	  for (j=1; j<=wfc_flag; j++){wfcllAB_num[ABloop][j] = 0;}
	}
      }

      AB_knum = 0;
      for (ik2=0; ik2<Nk[n2]; ik2++){
	for (ik3=0; ik3<Nk[n3]; ik3++){
	  psiAB[AB_knum] = 0.0;
	  AB_Nk2[AB_knum] = ik2;
	  AB_Nk3[AB_knum] = ik3;
	  /*
	     if (myid==Host_ID){
	     printf("   ABloop=%d, Nk2=%d, Nk3=%d\n",AB_knum,AB_Nk2[AB_knum], AB_Nk3[AB_knum]);
	     }
	     */
	  AB_knum++;
	}
      }

      if (myid==Host_ID){
	printf("  \nThe number of strings for Berry phase : AB mesh=%d\n\n",AB_knum);fflush(stdout);
      }

      /* allocate strings for Berry phase into proccessors */

      if (AB_knum<=myid){
	S_knum = -10;
	E_knum = -100;
	num_ABloop0 = 1;
      }
      /* Disabled by N. Yamaguchi else if (AB_knum<numprocs) {*/
      else if (AB_knum<=numprocs) {
	S_knum = myid;
	E_knum = myid;
	num_ABloop0 = 1;
      }
      else {
	tmp4 = (double)AB_knum/(double)numprocs;
	num_ABloop0 = (int)tmp4 + 1;
	S_knum = (int)((double)myid*(tmp4+0.0001));
	E_knum = (int)((double)(myid+1)*(tmp4+0.0001)) - 1;
	if (myid==(numprocs-1)) E_knum = AB_knum - 1;
	if (E_knum<0)           E_knum = 0;
      }

      /****************************************************
	start ABloop for Berry phase
       ****************************************************/

      for (ABloop0=0; ABloop0<num_ABloop0; ABloop0++){

	if (myid==Host_ID){
	  printf("  calculating the polarization along the %s .... %3d/%3d\n",
	      s_vec[k],ABloop0+1,num_ABloop0);fflush(stdout);
	}

	ABloop = ABloop0 + S_knum;

	arpo[myid] = -1;
	if (S_knum<=ABloop && ABloop<=E_knum) arpo[myid] = ABloop;
	for (ID=0; ID<numprocs; ID++){
	  MPI_Bcast(&arpo[ID], 1, MPI_INT, ID, comm1);
	}

	/* ABloop = arpo[myid]; Disabled by N. Yamaguchi */

	for (ID=0; ID<numprocs; ID++){

	  /* Added by N. Yamaguchi ***/
	  if (ABloop>=0) {
	    ABmesh2ID[ABloop] = ID;
	  }
	  /* ***/

	  MPI_Bcast(ABmesh2ID, AB_knum, MPI_INT, ID, comm1);
	}

	if (0<=ABloop){

	  /* for (i2=0; i2<Nk[n2]; i2++){ */
	  /* for (i3=0; i3<Nk[n3]; i3++){ */

	  i2=AB_Nk2[ABloop];
	  i3=AB_Nk3[ABloop];

	  for (spin=0; spin<spinsize; spin++){
	    mulr[spin] = 1.0;
	    muli[spin] = 0.0;
	    /* for band-by-band F. Ishii */
	    if (dpflag[0]>0){
	      for (bandg=0; bandg<bandgsize; bandg++){
		mulbbr[bandg][spin] = 1.0;
		mulbbi[bandg][spin] = 0.0;
	      }
	    }
	  }

	  for (i1=0; i1<Nk[n1]; i1++){

	    k1[n1] = kg[n1][i1];
	    k1[n2] = kg[n2][i2];
	    k1[n3] = kg[n3][i3];

	    k2[n1] = kg[n1][i1+1];
	    k2[n2] = kg[n2][i2];
	    k2[n3] = kg[n3][i3];

	    /*if (myid==Host_ID){
	      printf("\n k1:%10.6f %10.6f %10.6f\n",k1[n1],k1[n2],k1[n3]);
	      printf(" k2:%10.6f %10.6f %10.6f\n",k2[n1],k2[n2],k2[n3]);
	      } */

	    if      (i1==0)           diag_flag = 0;
	    else if (i1==(Nk[n1]-1))  diag_flag = 3;
	    else                      diag_flag = 2;

	    /* calculate the overlap matrix */

	    if (i1%2==0){

	      if (i1!=(Nk[n1]-1)){
		Overlap_k1k2( diag_flag, k1, k2, spinsize, fsize, fsize2, fsize3, fsize4,
		    MP, Sop, Wk1, Wk2, EigenVal1, EigenVal2 );
	      }
	      else{
		Overlap_k1k2( diag_flag, k1, k2, spinsize, fsize, fsize2, fsize3, fsize4,
		    MP, Sop, Wk1, Wk3, EigenVal1, EigenVal3 );
	      }
	    }
	    else{

	      if (i1!=(Nk[n1]-1)){
		Overlap_k1k2( diag_flag, k1, k2, spinsize, fsize, fsize2, fsize3, fsize4,
		    MP, Sop, Wk2, Wk1, EigenVal2, EigenVal1 );
	      }
	      else{
		Overlap_k1k2( diag_flag, k1, k2, spinsize, fsize, fsize2, fsize3, fsize4,
		    MP, Sop, Wk2, Wk3, EigenVal2, EigenVal3 );
	      }
	    }

	    /* store Wk1 and EigenVal1 at i1==0 */  

	    if (i1==0){
	      for (spin=0; spin<spinsize; spin++){
		for (i=0; i<fsize3; i++){
		  for (j=0; j<fsize3; j++){
		    Wk3[spin][i][j] = Wk1[spin][i][j];
		  }
		  EigenVal3[spin][i] = EigenVal1[spin][i];
		}
	      }
	    }

	    /* for (spin=0; spin<spinsize; spin++){ Disabled by N. Yamaguchi */

	      /* find the highest occupied state */

	    /* Added by N. Yamaguchi ***/
	    NOS1=0;
	    /* ***/
	    for (spin=0; spin<spinsize; spin++){ /* Added by N. Yamaguchi */

	      po = 0;
	      i = 1;
	      do {
		if (ChemP<EigenVal1[spin][i]){
		  po = 1;
		  /* hos = i - 1; Disabled by N. Yamaguchi */
		  /* Added by N. Yamaguchi ***/
		  hos[spin] = i - 1;
		  /* ***/
		}
		else i++;
	      } while (po==0 && i<=fsize2);

	      /* Added by N. Yamaguchi ***/
	      NOS1+=hos[spin];
	      /* ***/

	    } /* Added by N. Yamaguchi ***/

	    /* calculate the determinant of the overlap matrix between occupied states */

	    /* Disabled by N. Yamaguchi
	       if      (SpinP_switch==0){ hog2=Valence_Electrons/2;  }
	       else if (SpinP_switch==1){ hog2=Valence_Electrons/2;  }
	       else if (SpinP_switch==3){ hog2=Valence_Electrons;    }

	       if (SpinP_switch==1){ hog2=hos;}
	       if(hog2!=hos){metal=1;}
	       */

	    /* Added by N. Yamaguchi ***/
	    if (SpinP_switch==0){
	      NOS2=Valence_Electrons/2;
	    } else if (SpinP_switch==1){
	      NOS2=Valence_Electrons;
	    } else if (SpinP_switch==3){
	      NOS2=Valence_Electrons;
	    }
	    /* ***/

	    /* Added by N. Yamaguchi ***/
	    if (NOS2!=NOS1){
	      metal=1;
	    }
	    /* ***/

	    if(metal==1){printf("Metallic !! \n");}

	    for (spin=0; spin<spinsize; spin++){ /* Added by N. Yamaguchi */
	      /* Added by N. Yamaguchi ***/
	      upToOCC=0;
	      if (upToOCC==1 || SpinP_switch==1){
		hog2=hos[spin];
	      } else if (SpinP_switch==0){
		hog2=Valence_Electrons/2;
	      } else if (SpinP_switch==3){
		hog2=Valence_Electrons;
	      }
	      /* ***/

	      /*#########Band by band decomposition  F. Ishii */
	      if (dpflag[0]>0){
		/*check write Sop and Sopbb
		  for (i=1; i<=hog2; i++){
		  for (j=1; j<=hog2; j++){
		  printf("i,j, Sop(i,j).r Sop(i,j).i= %d %d %17.5f %17.5f\n",i,j,Sop[0][i][j].r, Sop[0][i][j].i);
		  }
		  } */
		tband=0;
		for (bandg=0; bandg<bandgsize; bandg++){
		  bandgm=bandg-1;
		  if(bandg!=0) tband=tband+bbnumb[bandgm];
		  tmpSr=0;
		  tmpSi=0;
		  for (i=1; i<=bbnumb[bandg]; i++){
		    for (j=1; j<=bbnumb[bandg]; j++){
		      tbandi=tband+i;
		      tbandj=tband+j;
		      tmpSr=Sop[spin][tbandi][tbandj].r;tmpSi=Sop[spin][tbandi][tbandj].i;
		      Sopbb[bandg][i][j].r=tmpSr; Sopbb[bandg][i][j].i=tmpSi;
		      /*   printf("bandg,bbnumb,i,j, Sopbb(i,j).r Sopbb(i,j).i= %d %d %d %d %17.5f %17.5f\n",bandg,bbnumb[bandg],i,j,Sopbb[bandg][i][j].r, Sopbb[bandg][i][j].i); */
		    }
		  }
		  determinant( spin, bbnumb[bandg], Sopbb[bandg], ipiv, work1, work2, Cdet );
		  /* printf("ABloop, bandg, Cdet.r(bbb), Cdet.i(bbb) =%d %d %17.5f %17.5f \n",ABloop, bandg, Cdet[0].r, Cdet[0].i); */
		  /* multiplication */
		  tmpbbr[bandg] = mulbbr[bandg][spin]*Cdet[spin].r - mulbbi[bandg][spin]*Cdet[spin].i;
		  tmpbbi[bandg] = mulbbr[bandg][spin]*Cdet[spin].i + mulbbi[bandg][spin]*Cdet[spin].r;
		  mulbbr[bandg][spin]=tmpbbr[bandg];
		  mulbbi[bandg][spin]=tmpbbi[bandg];
		}
	      }
	      determinant( spin, hog2, Sop[spin], ipiv, work1, work2, Cdet );
	      /*  printf("Cdet(all) = %17.5f %17.5f\n", Cdet[0].r, Cdet[0].i); */

	      /* multiplication */

	      tmpr = mulr[spin]*Cdet[spin].r - muli[spin]*Cdet[spin].i;
	      tmpi = mulr[spin]*Cdet[spin].i + muli[spin]*Cdet[spin].r;

	      mulr[spin] = tmpr;
	      muli[spin] = tmpi;

	      /* matrix multiplication (Added by N. Yamaguchi for HWC)
		 Lambda = Lambda * Sop
		 */

	      if (wfc_flag>0){
		/* Sop to Sopin */

		for (i=0; i<fsize4; i++) {
		  for (j=0; j<fsize4; j++) {
		    Sopin[j*fsize4+i] = Sop[spin][i+1][j+1];
		  }
		}
		if (svd_flag!=1){
		  lWork = 3 * fsize4;
		  zgesvd_("A", "A", &fsize4, &fsize4, Sopin, &fsize4, Sigma, U, &fsize4, VT, &fsize4, work_svd, &lWork, rwork_svd, &Info);
		  zgemm_("N", "N", &fsize4, &fsize4, &fsize4, &alpha, U, &fsize4, VT, &fsize4, &beta, Sopin, &fsize4);

		}
		if (i1==0) {

		  /* Sopin to Lambda */

		  for (i=0; i<fsize4*fsize4; i++) {
		    Lambda[spin][i] = Sopin[i];
		  }
		} else {

		  zgemm_("N", "N", &fsize4, &fsize4, &fsize4, &alpha, Lambda[spin], &fsize4, Sopin, &fsize4, &beta, Lout, &fsize4);

		  /* Lout to Lambda */

		  for (i=0; i<fsize4*fsize4; i++) {
		    Lambda[spin][i] = Lout[i];
		  }
		}
	      }
	    } /* spin */
	  }   /* i1   */

	  if (dpflag[0]>0){
	  /*check mulbbr&mulbbi
	    printf("mulr[0]= %17.5f\n",mulr[0]);
	    printf("muli[0]= %17.5f\n",muli[0]);
	    printf("mulbbr[0][0]= %17.5f\n",mulbbr[0][0]);
	    printf("mulbbi[0][0]= %17.5f\n",mulbbi[0][0]); */

	  /* for band-by-band F. Ishii */
	    for (spin=0; spin<spinsize; spin++){
	      for (bandg=0; bandg<bandgsize; bandg++){
		/****************************************
		  calculate Im[log(mul)]
note: acos(-1 to 1) gives PI to 0
		 ****************************************/

		/* Added by N. Yamaguchi ***/
		psi = atan2(mulbbi[bandg][spin], mulbbr[bandg][spin]);
		/* ***/

		/* Disabled by N. Yamaguchi
		   norm = sqrt(mulbbr[bandg][spin]*mulbbr[bandg][spin] + mulbbi[bandg][spin]*mulbbi[bandg][spin] );
		   detr = mulbbr[bandg][spin]/norm;
		   */

		/* the first and second quadrants */
		/* Disabled by N. Yamaguchi
		   if (0.0<=mulbbi[bandg][spin]){
		   psi = acos(detr);
		   }
		   */
		/* the third and fourth quadrants */
		/* Disabled by N. Yamaguchi
		   else {
		   psi = -acos(detr);
		   }
		   */

		psibbAB[bandg][ABloop]+= psi;
	      } /* bandg */
	    } /* spin */
	  }

	  for (spin=0; spin<spinsize; spin++){

	    /****************************************
	      calculate Im[log(mul)]

note: acos(-1 to 1) gives PI to 0
	     ****************************************/

	    /* Added by N. Yamaguchi ***/
	    psi = atan2(muli[spin], mulr[spin]);
	    /* ***/

	    /* Disabled by N. Yamaguchi
	       norm = sqrt( mulr[spin]*mulr[spin] + muli[spin]*muli[spin] );
	       detr = mulr[spin]/norm;
	       */

	    /* the first and second quadrants */
	    /* Disabled by N. Yamaguchi
	       if (0.0<=muli[spin]){
	       psi = acos(detr);
	       }
	       */
	    /* the third and fourth quadrants */
	    /* Disabled by N. Yamaguchi
	       else {
	       psi = -acos(detr);
	       }
	       */
	    /* add psi */

	    /* MPI psi from node */

	    psiAB[ABloop] += psi;

	    printf("\n proc=%d, psi=%17.8f\n\n",myid, psi);
	    /* ***/


	    /* Added by N. Yamaguchi for HWC */

	    if (wfc_flag>0){
	      lc1d = sqrt(tv[k][1] * tv[k][1] + tv[k][2] * tv[k][2] + tv[k][3] * tv[k][3]);
	      lWork = 2 * fsize4;
	      Dummy = 1;
	      zgeev_("N", "N", &fsize4, Lambda[spin], &fsize4, evout, vl, &Dummy, vr, &Dummy, Work, &lWork, rWork, &Info);

	      for (i=0; i<fsize4; i++){
		/* wfc(1d) = - c/(2*pi) * Im[log(evout[i])]*/
		double wfc1d = - lc1d * 0.5 / PI * atan2(evout[i].i, evout[i].r);
		if (wfc1d<0.0){
		  wfc1d += lc1d;
		}
		/* decompose HWCs to windows */
		int win_num = (int)(1 + wfc1d * wfc_flag / lc1d);
		if (win_num==wfc_flagp1){
		  win_num = wfc_flag;
		}
		wfcllAB[ABloop][win_num] += wfc1d;
		wfcAB[ABloop][i] = wfc1d;
		wfcllAB_num[ABloop][win_num]++;
	      }
	    }

	  } /* spin */
	} /* if(0=<ABloop) */
	} /* end of ABloop */

	MPI_Barrier(comm1);
	ABloop = 0;
	for (ik2=0; ik2<Nk[n2]; ik2++){
	  for (ik3=0; ik3<Nk[n3]; ik3++){
	    ID1 = ABmesh2ID[ABloop];
	    MPI_Bcast(&psiAB[ABloop], 1, MPI_DOUBLE, ID1, comm1);

	    /* Added by N. Yamaguchi for HWC */
	    if (wfc_flag>0){
	      MPI_Bcast(wfcllAB[ABloop], wfc_flagp1, MPI_DOUBLE, ID1, comm1);
	      MPI_Bcast(wfcAB[ABloop], fsize4, MPI_DOUBLE, ID1, comm1);
	      MPI_Bcast(wfcllAB_num[ABloop], wfc_flagp1, MPI_INT, ID1, comm1);
	    }

	    ABloop++;
	  }
	}

	MPI_Barrier(comm1);
	/* for band-by-band F. Ishii */
	if (dpflag[0]>0){
	  for (bandg=0; bandg<bandgsize; bandg++){
	    ABloop = 0;
	    for (ik2=0; ik2<Nk[n2]; ik2++){
	      for (ik3=0; ik3<Nk[n3]; ik3++){
		ID1 = ABmesh2ID[ABloop];
		MPI_Bcast(&psibbAB[bandg][ABloop], 1, MPI_DOUBLE, ID1, comm1);
		ABloop++;
	      }
	    }
	  }
	  MPI_Barrier(comm1);

	  /* Added by N. Yamaguchi ***/
	  free(AB_Nk2);
	  free(AB_Nk3);
	  free(ABmesh2ID);
	  /* ***/




	  /* for band-by-band F. Ishii */

	  MPI_Barrier(comm1);
	  if (myid==Host_ID){
	    printf("\n***************************************************************************\n");      fflush(stdout);
	    printf("  Electronic electric dipole and  polarization, D and P (Debye, muC/cm^2)\n");  fflush(stdout);
	    printf("  Band-by-band decomposition \n");  fflush(stdout);
	    printf("*****************************************************************************\n\n");      fflush(stdout);
	  }
	  SumDbbx=0;
	  SumDbby=0;
	  SumDbbz=0;
	  SumPbbx=0;
	  SumPbby=0;
	  SumPbbz=0;
	  for (bandg=0; bandg<bandgsize; bandg++){
	    sumbbpsi=0.0;
	    AB_knum = 0;
	    for (ik2=0; ik2<Nk[n2]; ik2++){
	      for (ik3=0; ik3<Nk[n3]; ik3++){
		sumbbpsi += psibbAB[bandg][AB_knum];
		AB_knum++;
	      }
	    }
	    MPI_Barrier(comm1);
	    /* calculate polarization for band-by-band F. Ishii */
	    /* collinear spin-unpolarized */

	    if (SpinP_switch==0) {
	      polbb_abc[bandg][k] = -2.0*sumbbpsi/(2.0*PI*(double)(Nk[n2]*Nk[n3]));
	    }
	    /* non-collinear case */

	    else if (SpinP_switch==3){
	      polbb_abc[bandg][k] = -sumbbpsi/(2.0*PI*(double)(Nk[n2]*Nk[n3]));
	    }
	    epa=polbb_abc[bandg][1];
	    epb=polbb_abc[bandg][2];
	    epc=polbb_abc[bandg][3];
	    Edpbbx = -AU2Debye*( epa*tv[1][1] + epb*tv[2][1] + epc*tv[3][1] );
	    Edpbby = -AU2Debye*( epa*tv[1][2] + epb*tv[2][2] + epc*tv[3][2] );
	    Edpbbz = -AU2Debye*( epa*tv[1][3] + epb*tv[2][3] + epc*tv[3][3] ); 
	    Pbbex = AU2Mucm*Edpbbx/Cell_Volume/AU2Debye;
	    Pbbey = AU2Mucm*Edpbby/Cell_Volume/AU2Debye;
	    Pbbez = AU2Mucm*Edpbbz/Cell_Volume/AU2Debye;


	    if (myid==Host_ID){
	      printf("Group of bands i=%d\n",bandg );    fflush(stdout);
	      printf("Dx(i), Dy(i), Dz(i)= %17.8f %17.8f %17.8f\n", Edpbbx,Edpbby,Edpbbz);    fflush(stdout);
	      printf("Px(i), Py(i), Pz(i)= %17.8f %17.8f %17.8f\n", Pbbex,Pbbey,Pbbez);    fflush(stdout);
	    }
	    SumDbbx=SumDbbx+Edpbbx;
	    SumDbby=SumDbby+Edpbby;
	    SumDbbz=SumDbbz+Edpbbz;
	    SumPbbx=SumPbbx+Pbbex;
	    SumPbby=SumPbby+Pbbey;
	    SumPbbz=SumPbbz+Pbbez;
	  } /* bandg */
	  if (myid==Host_ID){
	    printf("\n*************You should check sum of band dipole  coincide with total dipole !! \n");    fflush(stdout);
	    printf("Sum of band contribution, e.g Dex=Dx(1)+Dx(2)+....+Dx(N) \n");    fflush(stdout);
	    printf(" Dex, Dey, Dez= %17.8f %17.8f %17.8f\n", SumDbbx,SumDbby,SumDbbz);    fflush(stdout);
	    printf(" Pex, Pey, Pez= %17.8f %17.8f %17.8f\n", SumPbbx,SumPbby,SumPbbz);    fflush(stdout);
	  }
	}
	/* transfer psiAB as |psiAB[n]-psiAB[0]| <2pi */
	AB_knum = 0;
	for (ik2=0; ik2<Nk[n2]; ik2++){
	  for (ik3=0; ik3<Nk[n3]; ik3++){
	    /* transfer psiAB [-2pi,0]   */
	    while(psiAB[AB_knum] > 0.0)
	    {
	      psiAB[AB_knum]=psiAB[AB_knum]-2.0*PI;
	    }
	    while(psiAB[AB_knum] < -2.0*PI)
	    {
	      psiAB[AB_knum]=psiAB[AB_knum]+2.0*PI;
	    }

	    if (myid==Host_ID){
	      if(AB_knum==0){
		printf("\n transformed psi0=%17.8f\n\n", psiAB[0]);
	      }
	    }

	    if(AB_knum > 0){
	      dpsi=psiAB[AB_knum]-psiAB[0];
	      psip=dpsi+2.0*PI;
	      psim=dpsi-2.0*PI;
	      if(abs(psip) <= abs(dpsi)) psiAB[AB_knum]=psiAB[AB_knum]+2.0*PI;
	      if(abs(psim) <= abs(dpsi)) psiAB[AB_knum]=psiAB[AB_knum]-2.0*PI;
	      if (myid==Host_ID){
		printf("\n transformed psi=%17.8f\n\n", psiAB[AB_knum]);
	      }
	    }
	    AB_knum++;
	  }
	}


	AB_knum = 0;
	for (ik2=0; ik2<Nk[n2]; ik2++){
	  for (ik3=0; ik3<Nk[n3]; ik3++){
	    sumpsi[0] += psiAB[AB_knum];

	    /* Added by N. Yamaguchi for HWC */
	    if (wfc_flag>0){
	      for (j=1; j<=wfc_flag; j++){
		wfcll[j] += wfcllAB[AB_knum][j];
		wfcll_num[j] += wfcllAB_num[AB_knum][j];
	      }
	    }

	    AB_knum++;
	  }
	}
	/* Added by N. Yamaguchi for HWC */
	if (lp_flag>0){
	  int win_end = wfc_flag;
	  while (wfcll_num[win_end]==0 && win_end!=1){
	    win_end--;
	  }
	  int win_start = win_end==wfc_flag ? 1 : win_end + 1;
	  j = win_start;
	  int gap_max = 0;
	  while (j!=win_end){
	    int jm1 = j==1 ? wfc_flag : j - 1;
	    if (wfcll_num[j]>0){
	      wfcll_gap[j] = 0;
	    } else if (wfcll_num[jm1]==0){
	      wfcll_gap[j] = wfcll_gap[jm1] + 1;
	    } else {
	      wfcll_gap[j] = 1;
	    }
	    if (gap_max<wfcll_gap[j]){
	      gap_max=wfcll_gap[j];
	    }
	    j = j==wfc_flag ? 1 : j + 1;
	  }
	  int win_pbc = 1;
	  while (wfcll_gap[win_pbc]<gap_max && win_pbc!=wfc_flag){
	    win_pbc++;
	  }
	  double pbc = lc1d * (win_pbc - 1) / wfc_flag;
	  AB_knum = 0;
	  for (ik2=0; ik2<Nk[n2]; ik2++){
	    for (ik3=0; ik3<Nk[n3]; ik3++){
	      for (i=0; i<fsize4; i++){
		wfcAB[AB_knum][i] -= pbc;
		if (wfcAB[AB_knum][i]<0.0){
		  wfcAB[AB_knum][i] += lc1d;
		}
	      }
	      for (i=0; i<fsize4-1; i++){
		for (j=fsize4-1; j>i; j--){
		  if (wfcAB[AB_knum][j-1]>wfcAB[AB_knum][j]){
		    double temp = wfcAB[AB_knum][j];
		    wfcAB[AB_knum][j] = wfcAB[AB_knum][j-1];
		    wfcAB[AB_knum][j-1] = temp;
		  }
		}
	      }
	      for (j=0; j<fsize4; j++){
		wfc[j] += wfcAB[AB_knum][j];
	      }
	      AB_knum++;
	    }
	  }
	  for (j=0; j<fsize4; j++){
	    wfc[j] /= AB_knum;
	  }
	  AB_knum = 0;
	  for (ik2=0; ik2<Nk[n2]; ik2++){
	    for (ik3=0; ik3<Nk[n3]; ik3++){
	      for (i=0; i<fsize4; i++){
		dwfcAB[AB_knum][i] = wfcAB[AB_knum][i] - wfc[i];
	      }
	      AB_knum++;
	    }
	  }
	  for (j=0; j<fsize4; j++){
	    wfc[j] += pbc;
	    if (wfc[j]>lc1d){
	      wfc[j] -= lc1d;
	    }
	  }
	  for (i=0; i<fsize4-1; i++){
	    for (j=fsize4-1; j>i; j--){
	      if (wfc[j-1]>wfc[j]){
		double temp = wfc[j];
		wfc[j] = wfc[j-1];
		wfc[j-1] = temp;
		AB_knum = 0;
		for (ik2=0; ik2<Nk[n2]; ik2++){
		  for (ik3=0; ik3<Nk[n3]; ik3++){
		    temp = dwfcAB[AB_knum][j];
		    dwfcAB[AB_knum][j] = dwfcAB[AB_knum][j-1];
		    dwfcAB[AB_knum][j-1] = temp;
		    AB_knum++;
		  }
		}
	      }
	    }
	  }
	  AB_knum = 0;
	  for (ik2=0; ik2<Nk[n2]; ik2++){
	    for (ik3=0; ik3<Nk[n3]; ik3++){
	      for (i=0; i<fsize4; i++){
		wfcAB[AB_knum][i] = wfc[i] + dwfcAB[AB_knum][i];
	      }
	      AB_knum++;
	    }
	  }
	  int wall_num = 0;
	  int gap_width = gap_max;
	  j = 1;
	  while (wall_num<lp_flag && gap_width>0){
	    int jp1 = j==wfc_flag ? 1 : j + 1;
	    if (wfcll_gap[j]==gap_width && wfcll_gap[jp1]==0){
	      wfcll_wall[j] = 1;
	      wall_num++;
	    }
	    if (j==wfc_flag){
	      gap_width--;
	      j = 1;
	    } else {
	      j++;
	    }
	  }
	  int lp_num = 0;
	  j = win_pbc;
	  while (wfcll_layer[j]==0 && lp_num<=lp_flag){
	    if (wfcll_wall[j]==1){
	      lp_num++;
	    }
	    wfcll_layer[j] = lp_num;
	    j = j==wfc_flag ? 1 : j + 1;
	  }
	  for (j=0; j<fsize4; j++){
	    int win_num = (int)(1 + wfc[j] * wfc_flag / lc1d);
	    wfc_lp[wfcll_layer[win_num]] += wfc[j];
	  }
	}

	/* collinear spin-unpolarized */

	if (SpinP_switch==0) {
	  pol_abc[k] = -2.0*sumpsi[0]/(2.0*PI*(double)(Nk[n2]*Nk[n3]));
	}

	/* collinear spin-polarized */

	else if (SpinP_switch==1){ 
	  pol_abc[k] = -sumpsi[0]/(2.0*PI*(double)(Nk[n2]*Nk[n3])); 
	}

	/* non-collinear case */

	else if (SpinP_switch==3){
	  pol_abc[k] = -sumpsi[0]/(2.0*PI*(double)(Nk[n2]*Nk[n3])); 
	}

	} /* if pflag */
      } /* k:direction 1,2,3*/

      /************************************************
	estimate the phase disappeared by modulo 2PI 
       *************************************************/

      /*
	 Cdpx = dipole_moment_core[1];
	 Cdpy = dipole_moment_core[2];
	 Cdpz = dipole_moment_core[3];

	 for (i=1; i<=3; i++){
	 Phidden[i] = (rtv[i][1]*Cdpx + rtv[i][2]*Cdpy + rtv[i][3]*Cdpz)/(2.0*PI*AU2Debye);

	 i1 = floor(Phidden[i]);
	 i2 = i1 + 1;
	 i3 = i1 - 1;
	 Phidden0=Phidden[i];
	 if ( fabs((double)i1+pol_abc[i]-Phidden0)<fabs((double)i2+pol_abc[i]-Phidden0)){
	 Phidden[i] = Phidden0-(double)i1;}
	 else {
	 Phidden[i] = Phidden0-(double)i2;}
	 }

	 Cdpx = AU2Debye*(Phidden[1]*tv[1][1]+Phidden[2]*tv[2][1]+Phidden[3]*tv[3][1]);
	 Cdpy = AU2Debye*(Phidden[1]*tv[1][2]+Phidden[2]*tv[2][2]+Phidden[3]*tv[3][2]);
	 Cdpz = AU2Debye*(Phidden[1]*tv[1][3]+Phidden[2]*tv[2][3]+Phidden[3]*tv[3][3]);
	 */

      /*modified by F. Ishii Mar 24 2006 */
      Cdpx = dipole_moment_core[1];
      Cdpy = dipole_moment_core[2];
      Cdpz = dipole_moment_core[3];

      for (i=1; i<=3; i++){
	Phidden[i] = (rtv[i][1]*Cdpx + rtv[i][2]*Cdpy + rtv[i][3]*Cdpz)/(2.0*PI*AU2Debye);
	i1 = floor(Phidden[i]);
	Phidden[i]=Phidden[i]-(double)i1;
	Cdpi[i]=0.0;
      }

      for (i=1; i<=3; i++){
	Cdpi[i] = (tv[i][1]*Phidden[i] + tv[i][2]*Phidden[i] + tv[i][3]*Phidden[i])*AU2Debye;
	Pion0 = (tv[i][1]*(Phidden[i]-1) + tv[i][2]*(Phidden[i]-1) + tv[i][3]*(Phidden[i]-1))*AU2Debye;
	if (Pion0 > 0 && fabs(Pion0)<fabs(Cdpi[i])) Cdpi[i]=Pion0;
	Pion0 = (tv[i][1]*(Phidden[i]+1) + tv[i][2]*(Phidden[i]+1) + tv[i][3]*(Phidden[i]+1))*AU2Debye;
	if (Pion0 > 0 && fabs(Pion0)<fabs(Cdpi[i])) Cdpi[i]=Pion0; 
      }


      Cdpx = Cdpi[1];
      Cdpy = Cdpi[2];
      Cdpz = Cdpi[3];


      /* a.u to Debye, where the minus sign corresponds to that an electron has negative charge */

      Edpx = -AU2Debye*( pol_abc[1]*tv[1][1] + pol_abc[2]*tv[2][1] + pol_abc[3]*tv[3][1] );
      Edpy = -AU2Debye*( pol_abc[1]*tv[1][2] + pol_abc[2]*tv[2][2] + pol_abc[3]*tv[3][2] );
      Edpz = -AU2Debye*( pol_abc[1]*tv[1][3] + pol_abc[2]*tv[2][3] + pol_abc[3]*tv[3][3] );

      /************************************************
	find the core charge contribution of
	the macroscopic polarization
       *************************************************/

      /* Cdpx = dipole_moment_core[1];
	 Cdpy = dipole_moment_core[2];
	 Cdpz = dipole_moment_core[3]; */

      /************************************************
	find the background charge contribution of
	the macroscopic polarization
       *************************************************/

      Bdpx = dipole_moment_background[1];
      Bdpy = dipole_moment_background[2];
      Bdpz = dipole_moment_background[3];

      /************************************************
	calculate the total macroscopic polarization 
	as dipolemoment (for molecule)
       *************************************************/

      Tdpx = Cdpx + Edpx + Bdpx;
      Tdpy = Cdpy + Edpy + Bdpy;
      Tdpz = Cdpz + Edpz + Bdpz;

      AbsD = sqrt(Tdpx*Tdpx + Tdpy*Tdpy + Tdpz*Tdpz);

      /************************************************
	calculate the total macroscopic polarization 
	in micro C/cm^2
       *************************************************/

      /* Tdpx = Cdpx + Edpx + Bdpx;
	 Tdpy = Cdpy + Edpy + Bdpy;
	 Tdpz = Cdpz + Edpz + Bdpz; */

      Tdpx = Cdpx + Edpx;
      Tdpy = Cdpy + Edpy;
      Tdpz = Cdpz + Edpz;

      Ptx = AU2Mucm*Tdpx/Cell_Volume/AU2Debye;
      Pty = AU2Mucm*Tdpy/Cell_Volume/AU2Debye;
      Ptz = AU2Mucm*Tdpz/Cell_Volume/AU2Debye;
      Pbx = AU2Mucm*Bdpx/Cell_Volume/AU2Debye;
      Pby = AU2Mucm*Bdpy/Cell_Volume/AU2Debye;
      Pbz = AU2Mucm*Bdpz/Cell_Volume/AU2Debye;
      Pex = AU2Mucm*Edpx/Cell_Volume/AU2Debye;
      Pey = AU2Mucm*Edpy/Cell_Volume/AU2Debye;
      Pez = AU2Mucm*Edpz/Cell_Volume/AU2Debye;
      Pcx = AU2Mucm*Cdpx/Cell_Volume/AU2Debye;
      Pcy = AU2Mucm*Cdpy/Cell_Volume/AU2Debye;
      Pcz = AU2Mucm*Cdpz/Cell_Volume/AU2Debye;

      AbsD = sqrt(Tdpx*Tdpx + Tdpy*Tdpy + Tdpz*Tdpz);

      /************************************************
	print results
       *************************************************/

      if (myid==Host_ID){
	printf("\n*******************************************************\n");      fflush(stdout);
	printf("              Electric dipole  (Debye) : Berry phase         \n");  fflush(stdout);
	printf("*******************************************************\n\n");      fflush(stdout);

	printf(" Absolute dipole moment %17.8f\n\n",AbsD); 

	/* printf("               Background        Core             Electron          Total\n\n"); fflush(stdout); */
	printf("                 Core             Electron          Total\n\n"); fflush(stdout);

	printf(" Dx     %17.8f %17.8f %17.8f\n",Cdpx,Edpx,Tdpx);fflush(stdout);
	printf(" Dy     %17.8f %17.8f %17.8f\n",Cdpy,Edpy,Tdpy);fflush(stdout);
	printf(" Dz     %17.8f %17.8f %17.8f\n",Cdpz,Edpz,Tdpz);fflush(stdout);
	printf("\n\n");

	/************************************************
	  print results
	 *************************************************/
	if(metal==1){
	  printf("\n ##!! Warning!! The system may be metallic! Result is not correct!  \n"); fflush(stdout);
	}
	printf("\n***************************************************************\n"); fflush(stdout);
	printf("              Electric polarization (muC/cm^2) : Berry phase          \n");  fflush(stdout);
	printf("***************************************************************\n\n"); fflush(stdout);

	/*printf(" Absolute D %17.8f\n\n",AbsD); */

	printf("                 Core             Electron          Total\n\n"); fflush(stdout);

	printf(" Px     %17.8f %17.8f %17.8f\n",Pcx,Pex,Ptx);fflush(stdout);
	printf(" Py     %17.8f %17.8f %17.8f\n",Pcy,Pey,Pty);fflush(stdout);
	printf(" Pz     %17.8f %17.8f %17.8f\n",Pcz,Pez,Ptz);fflush(stdout);
	printf("\n\n");

	printf("\n*************************************************************************************\n"); fflush(stdout);
	printf("  Note: Bulk electric polarization have arbitrariness 2eR/omega or eR/omega (muC/cm^2)        \n");  fflush(stdout);
	printf("***************************************************************************************\n\n"); fflush(stdout);

	printf("                      R1                R2                R3\n"); fflush(stdout);

	for (i=0; i<=3; i++){
	  Parb[i] = AU2Mucm*2.0*PI/Gabs[i]/Cell_Volume;
	}

	if (SpinP_switch==1){
	  printf(" Mod      %17.8f %17.8f %17.8f\n",Parb[1],Parb[2],Parb[3]);fflush(stdout);
	}
	else if (SpinP_switch==0){
	  printf(" Mod      %17.8f %17.8f %17.8f\n",2.0*Parb[1],2.0*Parb[2],2.0*Parb[3]);fflush(stdout);
	}
	else if (SpinP_switch==3){
	  printf(" Mod      %17.8f %17.8f %17.8f\n",Parb[1],Parb[2],Parb[3]);fflush(stdout);
	}

	/* Added by N. Yamaguchi for HWC */

	if (lp_flag>0){
	  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	    double cc1d = Gxyz[ct_AN][n1];
	    if (cc1d<0.0){
	      cc1d += lc1d;
	    }
	    int win_num = (int)(1 + cc1d * wfc_flag / lc1d);
	    cc_lp[wfcll_layer[win_num]] += cc_vec[ct_AN-1] * cc1d;
	  }
	}

	if (wfc_flag>0){
	  for (j=0;j<fsize4;j++){
	    for (i=0;i<AB_knum;i++){
	      if (lp_flag>0){
		printf("<HWC, num=%3d, AB=%3d> %17.8f\n", j, i, wfcAB[i][j]*Bohr2Ang);
	      } else {
		printf("<theta, num=%3d, AB=%3d> %17.8f\n", j, i, wfcAB[i][j]/lc1d*2.0*PI);
	      }
	    }
	  }
	  printf("\n***************************************************************\n"); fflush(stdout);
	  printf("            the sum of Wannier function center (Bohr) : window by window          \n");  fflush(stdout);
	  printf("***************************************************************\n\n"); fflush(stdout);
	  double Parb1d = SpinP_switch==0 ? 2.0 * Parb[n1] : Parb[n1];
	  double degeneracy = SpinP_switch==0 ? 2.0 : 1.0;
	  for (j=1; j<=wfc_flag; j++){
	    printf(" <HWC, win=%4d> %17.8f %5d %5d %1d %5d\n", j, wfcll[j], wfcll_num[j], wfcll_gap[j], wfcll_wall[j], wfcll_layer[j]);
	  }
	  if (lp_flag>0){
	    char root[MAX_LINE_SIZE];
	    strncpy(root, argv[1], MAX_LINE_SIZE);
	    char *p_tmp;
	    if((p_tmp = strchr(root, '.')) != NULL) *p_tmp = '\0';
	    char fname_lpcc[MAX_LINE_SIZE+5];
	    strncpy(fname_lpcc, root, MAX_LINE_SIZE);
	    strncat(fname_lpcc, ".LPCC", 5);
	    FILE *fp_lpcc = fopen(fname_lpcc, "w");
	    char buf[fp_bsize];          /* setvbuf */
#ifdef xt3
	    setvbuf(fp_lpcc,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif
	    fputs("# num, LP(Core, e-10C/m)\n", fp_lpcc);
	    for (j=1; j<=lp_flag; j++){
	      double lp_cc = AU2Mucm / Cell_Volume * cc_lp[j];
	      double lp_mod = fmod(lp_cc, Parb[n1]);
	      if (lp_mod>0.0) {
		double temp = lp_mod - Parb[n1];
		if (fabs(lp_mod)>fabs(temp)){
		  lp_mod = temp;
		}
	      } else {
		double temp = lp_mod + Parb[n1];
		if (fabs(lp_mod)>fabs(temp)){
		  lp_mod = temp;
		}
	      }
	      cc_lp_mod[j] = lp_mod;
	      printf("<LP(Core, e-10C/m), num=%4d> %17.8f\n", j, cc_lp_mod[j]*Bohr2Ang*1.0e-2*lc1d);
	      fprintf(fp_lpcc, "%4d  %17.8f\n", j, cc_lp_mod[j]*Bohr2Ang*1.0e-2*lc1d);
	    }
	    fclose(fp_lpcc);

	    char fname_lpwfc[MAX_LINE_SIZE+6];
	    strncpy(fname_lpwfc, root, MAX_LINE_SIZE);
	    strncat(fname_lpwfc, ".LPHWC", 6);
	    FILE *fp_lpwfc = fopen(fname_lpwfc, "w");
#ifdef xt3
	    setvbuf(fp_lpwfc,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif
	    fputs("# num, LP(Electron, e-10C/m)\n", fp_lpwfc);
	    for (j=1; j<=lp_flag; j++){
	      double lp = -degeneracy * AU2Mucm / Cell_Volume * wfc_lp[j];
	      double lp_mod = fmod(lp, Parb1d);
	      if (lp_mod>0.0) {
		double temp = lp_mod - Parb1d;
		if (fabs(lp_mod)>fabs(temp)){
		  lp_mod = temp;
		}
	      } else {
		double temp = lp_mod + Parb1d;
		if (fabs(lp_mod)>fabs(temp)){
		  lp_mod = temp;
		}
	      }
	      wfc_lp_mod[j] = lp_mod;
	      printf("<LP(Electron, e-10C/m), num=%4d> %17.8f\n", j, wfc_lp_mod[j]*Bohr2Ang*1.0e-2*lc1d);
	      fprintf(fp_lpwfc, "%4d  %17.8f\n", j, wfc_lp_mod[j]*Bohr2Ang*1.0e-2*lc1d);
	    }
	    fclose(fp_lpwfc);

	    char fname_lp[MAX_LINE_SIZE+3];
	    strncpy(fname_lp, root, MAX_LINE_SIZE);
	    strncat(fname_lp, ".LP", 3);
	    FILE *fp_lp = fopen(fname_lp, "w");
#ifdef xt3
	    setvbuf(fp_lp,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif
	    fputs("# num, LP(Total, e-10C/m)\n", fp_lp);
	    for (j=1; j<=lp_flag; j++){
	      double lp_t = cc_lp_mod[j] + wfc_lp_mod[j];
	      double lp_mod = fmod(lp_t, Parb[n1]);
	      if (lp_mod>0.0) {
		double temp = lp_mod - Parb[n1];
		if (fabs(lp_mod)>fabs(temp)){
		  lp_mod = temp;
		}
	      } else {
		double temp = lp_mod + Parb[n1];
		if (fabs(lp_mod)>fabs(temp)){
		  lp_mod = temp;
		}
	      }
	      double lpt = lp_mod*Bohr2Ang*1.0e-2*lc1d;
	      printf("<LP(Total, e-10C/m), num=%4d> %17.8f\n", j, lpt);
	      fprintf(fp_lp, "%4d  %17.8f\n", j, lpt);
	    }
	    fclose(fp_lp);

	    double P1dC = 0.0;
	    char fname_cc[MAX_LINE_SIZE+3];
	    strncpy(fname_cc, root, MAX_LINE_SIZE);
	    strncat(fname_cc, ".CC", 3);
	    FILE *fp_cc = fopen(fname_cc, "w");
#ifdef xt3
	    setvbuf(fp_cc,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif
	    fputs("# num, position(CC), charge num\n", fp_cc);
	    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	      printf(" <CC, num=%4d> %17.8f %17.8f\n", ct_AN, Gxyz[ct_AN][n1]*Bohr2Ang, cc_vec[ct_AN-1]);
	      fprintf(fp_cc, "%4d  %17.8f  %17.8f\n", ct_AN, Gxyz[ct_AN][n1]*Bohr2Ang, cc_vec[ct_AN-1]);
	      P1dC += AU2Mucm / Cell_Volume * cc_vec[ct_AN-1] * Gxyz[ct_AN][n1];
	    }
	    fclose(fp_cc);

	    double P1d = 0.0;
	    char fname_wfc[MAX_LINE_SIZE+4];
	    strncpy(fname_wfc, root, MAX_LINE_SIZE);
	    strncat(fname_wfc, ".HWC", 4);
	    FILE *fp_wfc = fopen(fname_wfc, "w");
#ifdef xt3
	    setvbuf(fp_wfc,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif
	    fputs("# num, position(HWC)\n", fp_wfc);
	    for (j=0; j<fsize4; j++){
	      printf(" <HWC, num=%4d> %17.8f\n", j, wfc[j]*Bohr2Ang);
	      fprintf(fp_wfc, "%4d  %17.8f\n", j, wfc[j]*Bohr2Ang);
	      P1d += -degeneracy * AU2Mucm / Cell_Volume * wfc[j];
	    }
	    fclose(fp_wfc);

	    P1dC = fmod(P1dC, Parb[n1]);
	    if (P1dC>0.0) {
	      double temp = P1dC - Parb[n1];
	      if (fabs(P1dC)>fabs(temp)){
		P1dC = temp;
	      }
	    } else {
	      double temp = P1dC + Parb[n1];
	      if (fabs(P1dC)>fabs(temp)){
		P1dC = temp;
	      }
	    }
	    printf("\n P1dC = %17.8f\n", P1dC);

	    P1d = fmod(P1d, Parb1d);
	    if (P1d>0.0) {
	      double temp = P1d - Parb1d;
	      if (fabs(P1d)>fabs(temp)){
		P1d = temp;
	      }
	    } else {
	      double temp = P1d + Parb1d;
	      if (fabs(P1d)>fabs(temp)){
		P1d = temp;
	      }
	    }
	    printf("\n P1d = %17.8f\n", P1d);

	    double P1dT = P1dC + P1d;
	    printf("\n P1dC + P1d = %17.8f\n", P1dT);

	    char fname_out[MAX_LINE_SIZE+7];
	    strncpy(fname_out, root, MAX_LINE_SIZE);
	    strncat(fname_out, ".HWCout", 7);
	    FILE *fp_out = fopen(fname_out, "w");
#ifdef xt3
	    setvbuf(fp_out,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif
	    fputs("# Electric polarization (muC/cm^2)\n", fp_out);
	    fprintf(fp_out, "# Mod(one charge) = %17.8f\n", Parb[n1]);
	    fputs("#              Core             Electron          Total\n\n", fp_out);
	    fprintf(fp_out, " P1d   %17.8f %17.8f %17.8f\n", P1dC, P1d, P1dT);
	    fclose(fp_out);

	    printf("\n sum of lp(core) = %17.8f\n", P1dC*Bohr2Ang*1.0e-2*lc1d);
	    printf("\n sum of lp(Electron) = %17.8f\n", P1d*Bohr2Ang*1.0e-2*lc1d);
	    printf("\n sum of lp(Total) = %17.8f\n", P1dT*Bohr2Ang*1.0e-2*lc1d);
	    printf("\n Parblp(one charge, e-10C/m) = %17.8f\n", Parb[n1]*Bohr2Ang*1.0e-2*lc1d);
	  } else {
	    for (j=0; j<fsize4; j++){
	      printf(" <theta(average), num=%4d> %17.8f\n", j, wfc[j]/lc1d*2.0*PI);
	    }
	  }
	  printf("\n lc1d = %17.8f\n", lc1d);
	  printf("\n the number of HWC = %d\n", fsize4*AB_knum);
	  for (j=0; j<fsize4; j++){
	    double kpath_d = 0.0;
	    double pre_d;
	    double kg_d;
	    int ik2max = Nk[n2] / 2;
	    int ik3max = Nk[n3] / 2;
	    ik3 = 0;
	    pre_d = 0.0;
	    for (ik2=0; ik2<=ik2max; ik2++){
	      kg_d = kg[n2][ik2];
	      kpath_d += fabs(kg_d - pre_d);
	      if (lp_flag>0){
		printf(" <HWCAB> %17.8f  %17.8f\n", kpath_d, Bohr2Ang * wfcAB[ik2*Nk[n3]+ik3][j]);
	      } else {
		printf(" <HWCAB> %17.8f  %17.8f\n", kpath_d, Bohr2Ang * wfcAB[ik2*Nk[n3]+ik3][j]/lc1d*2.0*PI);
	      }
	      pre_d = kg_d;
	    }
	    ik2 = ik2max;
	    pre_d = 0.0;
	    for (ik3=1; ik3<=ik3max; ik3++){
	      kg_d = kg[n3][ik3];
	      kpath_d += fabs(kg_d - pre_d);
	      if (lp_flag>0){
		printf(" <HWCAB> %17.8f  %17.8f\n", kpath_d, Bohr2Ang * wfcAB[ik2*Nk[n3]+ik3][j]);
	      } else {
		printf(" <HWCAB> %17.8f  %17.8f\n", kpath_d, Bohr2Ang * wfcAB[ik2*Nk[n3]+ik3][j]/lc1d*2.0*PI);
	      }
	      pre_d = kg_d;
	    }
	    pre_d = sqrt(kg[n2][ik2max] * kg[n2][ik2max] + kg[n3][ik3max] * kg[n3][ik3max]);
	    for (ik3=ik3max-1; ik3>=0; ik3--){
	      ik2 = ik3;
	      kg_d = sqrt(kg[n2][ik2] * kg[n2][ik2] + kg[n3][ik3] * kg[n3][ik3]);
	      kpath_d += fabs(kg_d - pre_d);
	      if (lp_flag>0){
		printf(" <HWCAB> %17.8f  %17.8f\n", kpath_d, Bohr2Ang * wfcAB[ik2*Nk[n3]+ik3][j]);
	      } else {
		printf(" <HWCAB> %17.8f  %17.8f\n", kpath_d, Bohr2Ang * wfcAB[ik2*Nk[n3]+ik3][j]/lc1d*2.0*PI);
	      }
	      pre_d = kg_d;
	    }
	    printf(" <HWCAB> \n");
	  }
	}

      } /* print if(myid==Host) */

      /******************************************
	free arrays
       ******************************************/

      free(MP);

      for (spin=0; spin<spinsize; spin++){
	for (i=0; i<fsize3; i++){
	  free(Sop[spin][i]);
	}
	free(Sop[spin]);
      }
      free(Sop);

      for (spin=0; spin<spinsize; spin++){
	for (i=0; i<fsize3; i++){
	  free(Wk1[spin][i]);
	}
	free(Wk1[spin]);
      }
      free(Wk1);

      for (spin=0; spin<spinsize; spin++){
	for (i=0; i<fsize3; i++){
	  free(Wk2[spin][i]);
	}
	free(Wk2[spin]);
      }
      free(Wk2);

      for (spin=0; spin<spinsize; spin++){
	for (i=0; i<fsize3; i++){
	  free(Wk3[spin][i]);
	}
	free(Wk3[spin]);
      }
      free(Wk3);

      for (spin=0; spin<spinsize; spin++){
	free(EigenVal1[spin]);
      }
      free(EigenVal1);

      for (spin=0; spin<spinsize; spin++){
	free(EigenVal2[spin]);
      }
      free(EigenVal2);

      for (spin=0; spin<spinsize; spin++){
	free(EigenVal3[spin]);
      }
      free(EigenVal3);

      free(ipiv);
      free(work1);
      free(work2);

      for (i=0; i<=3; i++){
	free(kg[i]);
      }
      free(kg);

      /* Added by N. Yamaguchi for HWC */

      if (wfc_flag>0){
	for (ABloop=0; ABloop<AB_knum; ABloop++){
	  free(wfcllAB[ABloop]);
	  free(wfcllAB_num[ABloop]);
	}
	free(wfcllAB);
	free(wfcllAB_num);
      }

      free(psiAB);

      /* Added by N. Yamaguchi for HWC */

      if (wfc_flag>0){
	for (spin=0; spin<spinsize; spin++){
	  free(Lambda[spin]);
	}
	free(Lambda);
	free(Sopin);
	free(Lout);
	free(evout);
	free(Work);
	free(rWork);
	free(wfcll);
	if (svd_flag!=1){
	  free(U);
	  free(VT);
	  free(work_svd);
	  free(rwork_svd);
	}
	free(Sigma);
	for (ABloop=0; ABloop<AB_knum; ABloop++){
	  free(wfcAB[ABloop]);
	  free(dwfcAB[ABloop]);
	}
	free(wfcAB);
	free(dwfcAB);
	free(wfc);
	free(wfcll_num);
	free(wfcll_gap);
	free(wfc_lp);
	free(wfcll_wall);
	free(wfcll_layer);
	free(cc_lp);
      }

      /* print message */

      MPI_Barrier(comm1);

      dtime(&TEtime);

      printf(" \nElapsed time = %lf (s) for myid=%3d\n",TEtime-TStime,myid);fflush(stdout);

      MPI_Barrier(comm1);
      printf(" \nThe calculation was finished normally in myid=%2d.\n",myid);fflush(stdout);

      /* MPI_Finalize */

      MPI_Finalize();

      /* return */

      return 0;
    }










    void Overlap_k1k2(int diag_flag,
	double k1[4], double k2[4],
	int spinsize,
	int fsize, int fsize2, int fsize3, int fsize4,
	int *MP,
	dcomplex ***Sop, dcomplex ***Wk1, dcomplex ***Wk2,
	double **EigenVal1, double **EigenVal2)
    {
      /********************************************************************
	void Overlap_k1k2( int diag_flag,
	double k1[4], double k2[4],
	int spinsize,
	int fsize, int fsize2, int fsize3,
	int *MP,
	dcomplex ***Sop, dcomplex ***Wk1, dcomplex ***Wk2,
	double **EigenVal1, double **EigenVal2)

	a routine for calculating the overlap matrix between one-particle
	wave functions calculated at two k-points, k1 and k2.

Note:
(1) tv_i \dot rtv_j = 2PI * Kronecker's delta_{ij}

(2) The k1- and k2-points are defined by
k1 -> k1[1]*rtv[1] + k1[2]*rtv[2] + k1[3]*rtv[3]
k2 -> k2[1]*rtv[1] + k2[2]*rtv[2] + k2[3]*rtv[3]

(3) The first Brillouin zone is given by
-0.5 to 0.5 for k1[1-3] or k2[1-3]

int diag_flag (input variable)

diagonalize k1 and k2:  diag_flag = 0
diagonalize k1:         diag_flag = 1
diagonalize k2:         diag_flag = 2
diagonalize no:         diag_flag = 3

double k1,k2 (input variables)

k-points at which wave functions are calculated.
The first Brillouin zone is given by
-0.5 to 0.5 for k1[1-3] or k2[1-3]

int spinsize (input variable)

collinear spin-unpolarized:    spinsize = 1
collinear spin-polarized:      spinsize = 2
non-collinear spin-polarized:  spinsize = 1

int fsize (input variable)

collinear spin-unpolarized:    fsize = sum of basis orbitals
collinear spin-polarized:      fsize = sum of basis orbitals
non-collinear spin-polarized:  fsize = 2 * sum of basis orbitals

int fsize2 (input variable)

collinear spin-unpolarized:    fsize2 = fsize
collinear spin-polarized:      fsize2 = fsize
non-collinear spin-polarized:  fsize2 = 2*fsize

int fsize3 (input variable)

collinear spin-unpolarized:    fsize3 = fsize + 2
collinear spin-polarized:      fsize3 = fsize + 2
non-collinear spin-polarized:  fsize3 = 2*fsize + 2

int *MP (input variable)

a pointer which shows the starting number
of basis orbitals associated with atom i
in the full matrix

dcomplex ***Sop (output variable)

overlap matrix between one-particle wave functions calculated
at two k-points, k1 and k2. S[spin][i][j] is the matrix element
between a state i at k1 and a state j at k2 with a spin index,
where the variables i and j run 1 to fsize2.


      dcomplex ***Wk1 (input/output variable)

	one-particle wave functions at the k1-point. Wk1[spin][i][m] 
	is the component m of the state i and spin index 'spin'.
	If diag_flag=2, then the input of Wk1 is used for calculation
	of the overlap matrix. The variables i and m run from 1 to fsize2.
	The corresponding eigenvalues are found in EigenVal1. 

	dcomplex ***Wk2 (input/output variable)

	one-particle wave functions at the k2-point. Wk2[spin][i][m] 
	is the component m of the state i and spin index 'spin'.
	If diag_flag=1, then the input of Wk2 is used for calculation
	of the overlap matrix. The variables i and m run from 1 to fsize2.
	The corresponding eigenvalues are found in EigenVal2. 

	double **EigenVal1 (input/output variable)

	one-particle eigenenegies at the k1-point which are stored in 
	ascending order. EigenVal1[spin][i] is the eigenenegy of the
	state i and spin index 'spin', where the variable i runs from
	1 to fsize2. If diag_flag=2, then the input of EigenVal1 is
	not overwritten.

	double **EigenVal2 (input/output variable)

	one-particle eigenenegies at the k1-point which are stored in 
	ascending order. EigenVal2[spin][i] is the eigenenegy of the
	state i and spin index 'spin', where the variable i runs from
	1 to fsize2. If diag_flag=1, then the input of EigenVal2 is
	not overwritten.

	**********************************************************************/

	int spin;
      int ik,i1,j1,i,j,l,k;
      int ct_AN,h_AN,mu1,mu2;
      int Anum,Bnum,tnoA,tnoB;
      int mn,jj1,ii1,m;
      int Rnh,Gh_AN,l1,l2,l3;
      int recalc[2];
      double kpoints[2][4];
      double *ko,*M1;
      double sumr,sumi;
      double tmp1r,tmp1i;
      double tmp2r,tmp2i;
      double tmp3r,tmp3i;
      double si,co,kRn,k1da,k1db,k1dc;
      double dx,dy,dz;
      double dkx,dky,dkz;
      double dka,dkb,dkc;
      double k1x,k1y,k1z;
      dcomplex **S,**H,**C;  
      double OLP_eigen_cut = 1.0e-12;
      dcomplex Ctmp1,Ctmp2;  

      /****************************************************
	allocation of arrays:
       ****************************************************/

      ko = (double*)malloc(sizeof(double)*fsize3);
      M1 = (double*)malloc(sizeof(double)*fsize3);

      S = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
      for (i=0; i<fsize3; i++){
	S[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
	for (j=0; j<fsize3; j++){ S[i][j].r = 0.0; S[i][j].i = 0.0; }
      }

      H = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
      for (i=0; i<fsize3; i++){
	H[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
	for (j=0; j<fsize3; j++){ H[i][j].r = 0.0; H[i][j].i = 0.0; }
      }

      C = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
      for (i=0; i<fsize3; i++){
	C[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
	for (j=0; j<fsize3; j++){ C[i][j].r = 0.0; C[i][j].i = 0.0; }
      }

      /* set kpoints */

      kpoints[0][1] = k1[1];
      kpoints[0][2] = k1[2];
      kpoints[0][3] = k1[3];

      kpoints[1][1] = k2[1];
      kpoints[1][2] = k2[2];
      kpoints[1][3] = k2[3];

      k1x = k1[1]*rtv[1][1] + k1[2]*rtv[2][1] + k1[3]*rtv[3][1];
      k1y = k1[1]*rtv[1][2] + k1[2]*rtv[2][2] + k1[3]*rtv[3][2];
      k1z = k1[1]*rtv[1][3] + k1[2]*rtv[2][3] + k1[3]*rtv[3][3];

      dka = k2[1] - k1[1];
      dkb = k2[2] - k1[2];
      dkc = k2[3] - k1[3];

      dkx = dka*rtv[1][1] + dkb*rtv[2][1] + dkc*rtv[3][1];
      dky = dka*rtv[1][2] + dkb*rtv[2][2] + dkc*rtv[3][2];
      dkz = dka*rtv[1][3] + dkb*rtv[2][3] + dkc*rtv[3][3];

      if      (diag_flag==0) {recalc[0]=1; recalc[1]=1; }
      else if (diag_flag==1) {recalc[0]=1; recalc[1]=0; }
      else if (diag_flag==2) {recalc[0]=0; recalc[1]=1; }
      else if (diag_flag==3) {recalc[0]=0; recalc[1]=0; }

      /****************************************************
	diagonalize Bloch matrix at k-points, k1 and k2
       ****************************************************/

      for (ik=0; ik<2; ik++){

	if (recalc[ik]){     

	  Overlap_Band(OLP,S,MP,kpoints[ik][1],kpoints[ik][2],kpoints[ik][3]);

	  Eigen_HH(S,ko,fsize,fsize);

	  for (l=1; l<=fsize; l++){
	    if (ko[l]<OLP_eigen_cut){
	      printf("found an overcomplete basis set\n");
	      MPI_Finalize();
	      exit(0);
	    }
	  }

	  for (l=1; l<=fsize; l++) M1[l] = 1.0/sqrt(ko[l]);

	  /*
	     for (i1=1; i1<=fsize; i1++){
	     printf("ik=%2d i1=%3d ko[i1]=%15.12f\n",ik,i1,ko[i1]);
	     }
	     */

	  /* S * M1  */

	  for (i1=1; i1<=fsize; i1++){
	    for (j1=1; j1<=fsize; j1++){
	      S[i1][j1].r = S[i1][j1].r*M1[j1];
	      S[i1][j1].i = S[i1][j1].i*M1[j1];
	    } 
	  } 

	  for (spin=0; spin<spinsize; spin++){

	    /* transpose S */

	    for (i1=1; i1<=fsize; i1++){
	      for (j1=i1+1; j1<=fsize; j1++){
		Ctmp1 = S[i1][j1];
		Ctmp2 = S[j1][i1];
		S[i1][j1] = Ctmp2;
		S[j1][i1] = Ctmp1;
	      }
	    }

	    /****************************************************
	      collinear case
	     ****************************************************/

	    if (SpinP_switch==0 || SpinP_switch==1){

	      Hamiltonian_Band(Hks[spin],H,MP,kpoints[ik][1],kpoints[ik][2],kpoints[ik][3]);

	      /****************************************************
		M1 * U^t * H * U * M1
	       ****************************************************/

	      /* H * U * M1 */

	      for (j1=1; j1<=fsize; j1++){
		for (i1=1; i1<=fsize; i1++){

		  sumr = 0.0;
		  sumi = 0.0;

		  for (l=1; l<=fsize; l++){
		    sumr += H[i1][l].r*S[j1][l].r - H[i1][l].i*S[j1][l].i;
		    sumi += H[i1][l].r*S[j1][l].i + H[i1][l].i*S[j1][l].r;
		  }

		  C[j1][i1].r = sumr;
		  C[j1][i1].i = sumi;
		}
	      }     

	      /* M1 * U^+ H * U * M1 */

	      for (i1=1; i1<=fsize; i1++){
		for (j1=1; j1<=fsize; j1++){
		  sumr = 0.0;
		  sumi = 0.0;
		  for (l=1; l<=fsize; l++){
		    sumr +=  S[i1][l].r*C[j1][l].r + S[i1][l].i*C[j1][l].i;
		    sumi +=  S[i1][l].r*C[j1][l].i - S[i1][l].i*C[j1][l].r;
		  }
		  H[i1][j1].r = sumr;
		  H[i1][j1].i = sumi;
		}
	      } 

	      /* H to C */

	      for (i1=1; i1<=fsize; i1++){
		for (j1=1; j1<=fsize; j1++){
		  C[i1][j1] = H[i1][j1];
		}
	      }

	      /* solve eigenvalue problem */

	      if (ik==0){
		Eigen_HH(C,EigenVal1[spin],fsize,fsize);

		/*
		   for (i1=1; i1<=fsize; i1++){
		   printf("ik=%2d i1=%3d spin=%2d EigenVal2[spin]=%15.12f\n", 
		   ik,i1,spin,EigenVal1[spin][i1]);
		   }
		   */


	      }
	      else if (ik==1){
		Eigen_HH(C,EigenVal2[spin],fsize,fsize);

		/*
		   for (i1=1; i1<=fsize; i1++){
		   printf("ik=%2d i1=%3d spin=%2d EigenVal2[spin]=%15.12f\n",
		   ik,i1,spin,EigenVal2[spin][i1]);
		   }
		   */
	      }

	      /****************************************************
		transformation to the original eigenvectors.
		NOTE JRCAT-244p and JAIST-2122p 
	       ****************************************************/

	      /* transpose */
	      for (i1=1; i1<=fsize; i1++){
		for (j1=i1+1; j1<=fsize; j1++){
		  Ctmp1 = S[i1][j1];
		  Ctmp2 = S[j1][i1];
		  S[i1][j1] = Ctmp2;
		  S[j1][i1] = Ctmp1;
		}
	      }

	      /* transpose */
	      for (i1=1; i1<=fsize; i1++){
		for (j1=i1+1; j1<=fsize; j1++){
		  Ctmp1 = C[i1][j1];
		  Ctmp2 = C[j1][i1];
		  C[i1][j1] = Ctmp2;
		  C[j1][i1] = Ctmp1;
		}
	      }

	      /* calculate wave functions */

	      for (i1=1; i1<=fsize; i1++){
		for (j1=1; j1<=fsize; j1++){
		  sumr = 0.0;
		  sumi = 0.0;
		  for (l=1; l<=fsize; l++){
		    sumr +=  S[i1][l].r*C[j1][l].r - S[i1][l].i*C[j1][l].i;
		    sumi +=  S[i1][l].r*C[j1][l].i + S[i1][l].i*C[j1][l].r;
		  }

		  if (ik==0){
		    Wk1[spin][j1][i1].r = sumr;
		    Wk1[spin][j1][i1].i = sumi;
		  }
		  else if (ik==1){
		    Wk2[spin][j1][i1].r = sumr;
		    Wk2[spin][j1][i1].i = sumi;
		  }
		}
	      }
	    }

	    /****************************************************
	      non-collinear case
	     ****************************************************/

	    else if (SpinP_switch==3){

	      Hamiltonian_Band_NC(Hks,iHks,H,MP,kpoints[ik][1],kpoints[ik][2],kpoints[ik][3]);

	      /* H * U * M1 */

	      for (j1=1; j1<=fsize; j1++){
		for (i1=1; i1<=fsize2; i1++){
		  for (m=0; m<=1; m++){

		    sumr = 0.0;
		    sumi = 0.0;
		    mn = m*fsize;
		    for (l=1; l<=fsize; l++){
		      sumr += H[i1][l+mn].r*S[j1][l].r - H[i1][l+mn].i*S[j1][l].i;
		      sumi += H[i1][l+mn].r*S[j1][l].i + H[i1][l+mn].i*S[j1][l].r;
		    }

		    jj1 = 2*j1 - 1 + m;

		    C[jj1][i1].r = sumr;
		    C[jj1][i1].i = sumi;
		  }
		}
	      }     

	      /* M1 * U^+ H * U * M1 */

	      for (i1=1; i1<=fsize; i1++){

		for (m=0; m<=1; m++){

		  ii1 = 2*i1 - 1 + m;

		  for (j1=1; j1<=fsize2; j1++){
		    sumr = 0.0;
		    sumi = 0.0;
		    mn = m*fsize;
		    for (l=1; l<=fsize; l++){
		      sumr +=  S[i1][l].r*C[j1][l+mn].r + S[i1][l].i*C[j1][l+mn].i;
		      sumi +=  S[i1][l].r*C[j1][l+mn].i - S[i1][l].i*C[j1][l+mn].r;
		    }
		    H[ii1][j1].r = sumr;
		    H[ii1][j1].i = sumi;
		  }
		}
	      }

	      /* solve eigenvalue problem */

	      if (ik==0){
		Eigen_HH(H,EigenVal1[0],fsize2,fsize2);


		/*
		   for (i1=1; i1<=fsize2; i1++){
		   printf("ik=%2d i1=%3d spin=%2d EigenVal2[0]=%15.12f\n", 
		   ik,i1,spin,EigenVal1[0][i1]);
		   }
		   */


	      }
	      else if (ik==1){
		Eigen_HH(H,EigenVal2[0],fsize2,fsize2);
	      }

	      /****************************************************
		transformation to the original eigenvectors
		NOTE JRCAT-244p and JAIST-2122p 
		C = U * lambda^{-1/2} * D
	       ****************************************************/

	      /* transpose */

	      for (i1=1; i1<=fsize; i1++){
		for (j1=i1+1; j1<=fsize; j1++){
		  Ctmp1 = S[i1][j1];
		  Ctmp2 = S[j1][i1];
		  S[i1][j1] = Ctmp2;
		  S[j1][i1] = Ctmp1;
		}
	      }

	      for (i1=1; i1<=fsize2; i1++){
		for (j1=1; j1<=fsize2; j1++){
		  C[i1][j1].r = 0.0;
		  C[i1][j1].i = 0.0;
		}
	      }

	      for (m=0; m<=1; m++){
		for (i1=1; i1<=fsize; i1++){
		  for (j1=1; j1<=fsize2; j1++){

		    sumr = 0.0; 
		    sumi = 0.0;

		    for (l=1; l<=fsize; l++){
		      sumr +=  S[i1][l].r*H[2*l-1+m][j1].r - S[i1][l].i*H[2*l-1+m][j1].i;
		      sumi +=  S[i1][l].r*H[2*l-1+m][j1].i + S[i1][l].i*H[2*l-1+m][j1].r;
		    } 

		    if (ik==0){
		      Wk1[0][j1][i1+m*fsize].r = sumr;
		      Wk1[0][j1][i1+m*fsize].i = sumi;
		    }
		    else if (ik==1){
		      Wk2[0][j1][i1+m*fsize].r = sumr;
		      Wk2[0][j1][i1+m*fsize].i = sumi;
		    }
		  }
		}
	      }

	    }

	  } /* spin */
	} /* if (recalc[ik]) */
      } /* ik */

      /****************************************************
	calculate an overlap matrix between one-particle
	wave functions  calculated at two k-points, k1 and k2
       ****************************************************/

      /* collinear case */

      if (SpinP_switch==0 || SpinP_switch==1){

	for (spin=0; spin<spinsize; spin++){
	  for (mu1=1; mu1<=fsize4; mu1++){   /* for k1 */
	    for (mu2=1; mu2<=fsize4; mu2++){ /* for k2 */

	      sumr = 0.0; 
	      sumi = 0.0; 

	      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){ 

		tnoA = Total_NumOrbs[ct_AN];
		Anum = MP[ct_AN];

		for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){

		  Rnh = ncn[ct_AN][h_AN];
		  Gh_AN = natn[ct_AN][h_AN];
		  tnoB = Total_NumOrbs[Gh_AN];
		  Bnum = MP[Gh_AN];

		  l1 = atv_ijk[Rnh][1];
		  l2 = atv_ijk[Rnh][2];
		  l3 = atv_ijk[Rnh][3];

		  kRn = 2.0*PI*(k2[1]*(double)l1 + k2[2]*(double)l2 + k2[3]*(double)l3);
		  kRn -= (dkx*Gxyz[ct_AN][1] + dky*Gxyz[ct_AN][2] + dkz*Gxyz[ct_AN][3]);

		  si = sin(kRn);  /* Imaginary part */
		  co = cos(kRn);  /* Real part      */

		  for (i1=0; i1<tnoA; i1++){
		    for (j1=0; j1<tnoB; j1++){

		      tmp1r = Wk1[spin][mu1][Anum+i1].r*Wk2[spin][mu2][Bnum+j1].r
			+ Wk1[spin][mu1][Anum+i1].i*Wk2[spin][mu2][Bnum+j1].i;
		      tmp1i = Wk1[spin][mu1][Anum+i1].r*Wk2[spin][mu2][Bnum+j1].i
			- Wk1[spin][mu1][Anum+i1].i*Wk2[spin][mu2][Bnum+j1].r;

		      tmp2r = co*tmp1r - si*tmp1i;
		      tmp2i = co*tmp1i + si*tmp1r;

		      tmp3r = OLP[ct_AN][h_AN][i1][j1];
		      tmp3i =-dkx*OLPpox[ct_AN][h_AN][i1][j1]
			-dky*OLPpoy[ct_AN][h_AN][i1][j1]
			-dkz*OLPpoz[ct_AN][h_AN][i1][j1];

		      sumr += tmp2r*tmp3r - tmp2i*tmp3i;
		      sumi += tmp2r*tmp3i + tmp2i*tmp3r;
		    }
		  }

		}   
	      }       

	      Sop[spin][mu1][mu2].r = sumr;
	      Sop[spin][mu1][mu2].i = sumi;

	      /*
		 printf("%8.4f ",sumi);
		 */

	    }
	    /*
	       printf("\n");
	       */
	  }
	}

      } /* if (SpinP_switch==0 || SpinP_switch==1) */

      /* non-collinear case */

      else if (SpinP_switch==3){

	for (mu1=1; mu1<=fsize4; mu1++){   /* for k1 */
	  for (mu2=1; mu2<=fsize4; mu2++){ /* for k2 */

	    sumr = 0.0; 
	    sumi = 0.0; 

	    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){ 

	      tnoA = Total_NumOrbs[ct_AN];
	      Anum = MP[ct_AN];

	      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){

		Rnh = ncn[ct_AN][h_AN];
		Gh_AN = natn[ct_AN][h_AN];
		tnoB = Total_NumOrbs[Gh_AN];
		Bnum = MP[Gh_AN];

		l1 = atv_ijk[Rnh][1];
		l2 = atv_ijk[Rnh][2];
		l3 = atv_ijk[Rnh][3];

		kRn = 2.0*PI*(k2[1]*(double)l1 + k2[2]*(double)l2 + k2[3]*(double)l3);
		kRn -= (dkx*Gxyz[ct_AN][1] + dky*Gxyz[ct_AN][2] + dkz*Gxyz[ct_AN][3]);

		si = sin(kRn);  /* Imaginary part */
		co = cos(kRn);  /* Real part      */

		for (i1=0; i1<tnoA; i1++){
		  for (j1=0; j1<tnoB; j1++){

		    tmp1r = Wk1[0][mu1][Anum      +i1].r*Wk2[0][mu2][Bnum      +j1].r
		      + Wk1[0][mu1][Anum      +i1].i*Wk2[0][mu2][Bnum      +j1].i
		      + Wk1[0][mu1][Anum+fsize+i1].r*Wk2[0][mu2][Bnum+fsize+j1].r
		      + Wk1[0][mu1][Anum+fsize+i1].i*Wk2[0][mu2][Bnum+fsize+j1].i;

		    tmp1i = Wk1[0][mu1][Anum      +i1].r*Wk2[0][mu2][Bnum      +j1].i
		      - Wk1[0][mu1][Anum      +i1].i*Wk2[0][mu2][Bnum      +j1].r
		      + Wk1[0][mu1][Anum+fsize+i1].r*Wk2[0][mu2][Bnum+fsize+j1].i
		      - Wk1[0][mu1][Anum+fsize+i1].i*Wk2[0][mu2][Bnum+fsize+j1].r;

		    tmp2r = co*tmp1r - si*tmp1i;
		    tmp2i = co*tmp1i + si*tmp1r;

		    tmp3r = OLP[ct_AN][h_AN][i1][j1];
		    tmp3i =-dkx*OLPpox[ct_AN][h_AN][i1][j1]
		      -dky*OLPpoy[ct_AN][h_AN][i1][j1]
		      -dkz*OLPpoz[ct_AN][h_AN][i1][j1];

		    sumr += tmp2r*tmp3r - tmp2i*tmp3i;
		    sumi += tmp2r*tmp3i + tmp2i*tmp3r;
		  }
		}

	      }   
	    }       

	    Sop[0][mu1][mu2].r = sumr;
	    Sop[0][mu1][mu2].i = sumi;

	  }
	}

      } /* else if (SpinP_switch==3) */

      /****************************************************
	allocation of arrays:
       ****************************************************/

      free(ko);
      free(M1);

      for (i=0; i<fsize3; i++){
	free(S[i]);
      }
      free(S);

      for (i=0; i<fsize3; i++){
	free(H[i]);
      }
      free(H);

      for (i=0; i<fsize3; i++){
	free(C[i]);
      }
      free(C);

    }










    void Overlap_Band(double ****OLP,
	dcomplex **S,int *MP,
	double k1, double k2, double k3)
    {
      static int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
      static int l1,l2,l3,Rn,n2;
      static double **S1,**S2;
      static double kRn,si,co,s;

      Anum = 1;
      for (i=1; i<=atomnum; i++){
	MP[i] = Anum;
	Anum += Total_NumOrbs[i];
      }
      NUM = Anum - 1;

      /****************************************************
	Allocation
       ****************************************************/

      n2 = NUM + 2;

      S1 = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	S1[i] = (double*)malloc(sizeof(double)*n2);
      }

      S2 = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	S2[i] = (double*)malloc(sizeof(double)*n2);
      }

      /****************************************************
	set overlap
       ****************************************************/

      S[0][0].r = NUM;

      for (i=1; i<=NUM; i++){
	for (j=1; j<=NUM; j++){
	  S1[i][j] = 0.0;
	  S2[i][j] = 0.0;
	}
      }

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	tnoA = Total_NumOrbs[GA_AN];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  tnoB = Total_NumOrbs[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);
	  Bnum = MP[GB_AN];
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      s = OLP[GA_AN][LB_AN][i][j];
	      S1[Anum+i][Bnum+j] += s*co;
	      S2[Anum+i][Bnum+j] += s*si;
	    }
	  }
	}
      }

      for (i=1; i<=NUM; i++){
	for (j=1; j<=NUM; j++){
	  S[i][j].r =  S1[i][j];
	  S[i][j].i =  S2[i][j];
	}
      }

      /****************************************************
	free arrays
       ****************************************************/

      for (i=0; i<n2; i++){
	free(S1[i]);
	free(S2[i]);
      }
      free(S1);
      free(S2);

    }





    void Hamiltonian_Band(double ****RH, dcomplex **H, int *MP,
	double k1, double k2, double k3)
    {
      static int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
      static int l1,l2,l3,Rn,n2;
      static double **H1,**H2;
      static double kRn,si,co,h;

      Anum = 1;
      for (i=1; i<=atomnum; i++){
	MP[i] = Anum;
	Anum += Total_NumOrbs[i];
      }
      NUM = Anum - 1;

      /****************************************************
	Allocation
       ****************************************************/

      n2 = NUM + 2;

      H1 = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	H1[i] = (double*)malloc(sizeof(double)*n2);
      }

      H2 = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	H2[i] = (double*)malloc(sizeof(double)*n2);
      }

      /****************************************************
	set Hamiltonian
       ****************************************************/

      H[0][0].r = 2.0*NUM;
      for (i=1; i<=NUM; i++){
	for (j=1; j<=NUM; j++){
	  H1[i][j] = 0.0;
	  H2[i][j] = 0.0;
	}
      }

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	tnoA = Total_NumOrbs[GA_AN];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  tnoB = Total_NumOrbs[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);
	  Bnum = MP[GB_AN];
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      h = RH[GA_AN][LB_AN][i][j];
	      H1[Anum+i][Bnum+j] += h*co;
	      H2[Anum+i][Bnum+j] += h*si;
	    }
	  }
	}
      }

      for (i=1; i<=NUM; i++){
	for (j=1; j<=NUM; j++){
	  H[i][j].r = H1[i][j];
	  H[i][j].i = H2[i][j];
	}
      }

      /****************************************************
	free arrays
       ****************************************************/

      for (i=0; i<n2; i++){
	free(H1[i]);
	free(H2[i]);
      }
      free(H1);
      free(H2);
    }

















    void Hamiltonian_Band_NC(double *****RH, double *****IH,
	dcomplex **H, int *MP,
	double k1, double k2, double k3)
    {
      static int i,j,k,wanA,wanB,tnoA,tnoB,Anum,Bnum;
      static int NUM,GA_AN,LB_AN,GB_AN;
      static int l1,l2,l3,Rn,n2;
      static double **H11r,**H11i;
      static double **H22r,**H22i;
      static double **H12r,**H12i;
      static double kRn,si,co,h;

      /* set MP */

      Anum = 1;
      for (i=1; i<=atomnum; i++){
	MP[i] = Anum;
	Anum += Total_NumOrbs[i];
      }
      NUM = Anum - 1;
      n2 = NUM + 2;

      /*******************************************
	allocation of H11r, H11i,
	H22r, H22i,
	H12r, H12i
       *******************************************/

      H11r = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	H11r[i] = (double*)malloc(sizeof(double)*n2);
      }

      H11i = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	H11i[i] = (double*)malloc(sizeof(double)*n2);
      }

      H22r = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	H22r[i] = (double*)malloc(sizeof(double)*n2);
      }

      H22i = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	H22i[i] = (double*)malloc(sizeof(double)*n2);
      }

      H12r = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	H12r[i] = (double*)malloc(sizeof(double)*n2);
      }

      H12i = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	H12i[i] = (double*)malloc(sizeof(double)*n2);
      }

      /****************************************************
	set Hamiltonian
       ****************************************************/

      H[0][0].r = 2.0*NUM;
      for (i=1; i<=NUM; i++){
	for (j=1; j<=NUM; j++){
	  H11r[i][j] = 0.0;
	  H11i[i][j] = 0.0;
	  H22r[i][j] = 0.0;
	  H22i[i][j] = 0.0;
	  H12r[i][j] = 0.0;
	  H12i[i][j] = 0.0;
	}
      }

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	tnoA = Total_NumOrbs[GA_AN];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  tnoB = Total_NumOrbs[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);
	  Bnum = MP[GB_AN];

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      H11r[Anum+i][Bnum+j] += co*RH[0][GA_AN][LB_AN][i][j] -  si*IH[0][GA_AN][LB_AN][i][j];
	      H11i[Anum+i][Bnum+j] += si*RH[0][GA_AN][LB_AN][i][j] +  co*IH[0][GA_AN][LB_AN][i][j];
	      H22r[Anum+i][Bnum+j] += co*RH[1][GA_AN][LB_AN][i][j] -  si*IH[1][GA_AN][LB_AN][i][j];
	      H22i[Anum+i][Bnum+j] += si*RH[1][GA_AN][LB_AN][i][j] +  co*IH[1][GA_AN][LB_AN][i][j];
	      H12r[Anum+i][Bnum+j] += co*RH[2][GA_AN][LB_AN][i][j] - si*(RH[3][GA_AN][LB_AN][i][j]
		  + IH[2][GA_AN][LB_AN][i][j]);
	      H12i[Anum+i][Bnum+j] += si*RH[2][GA_AN][LB_AN][i][j] + co*(RH[3][GA_AN][LB_AN][i][j]
		  + IH[2][GA_AN][LB_AN][i][j]);
	    }
	  }

	}
      }

      /******************************************************
	the full complex matrix of H
       ******************************************************/

      for (i=1; i<=NUM; i++){
	for (j=1; j<=NUM; j++){
	  H[i    ][j    ].r =  H11r[i][j];
	  H[i    ][j    ].i =  H11i[i][j];
	  H[i+NUM][j+NUM].r =  H22r[i][j];
	  H[i+NUM][j+NUM].i =  H22i[i][j];
	  H[i    ][j+NUM].r =  H12r[i][j];
	  H[i    ][j+NUM].i =  H12i[i][j];
	  H[j+NUM][i    ].r =  H[i][j+NUM].r;
	  H[j+NUM][i    ].i = -H[i][j+NUM].i;
	}
      }

      /****************************************************
	free arrays
       ****************************************************/

      for (i=0; i<n2; i++){
	free(H11r[i]);
      }
      free(H11r);

      for (i=0; i<n2; i++){
	free(H11i[i]);
      }
      free(H11i);

      for (i=0; i<n2; i++){
	free(H22r[i]);
      }
      free(H22r);

      for (i=0; i<n2; i++){
	free(H22i[i]);
      }
      free(H22i);

      for (i=0; i<n2; i++){
	free(H12r[i]);
      }
      free(H12r);

      for (i=0; i<n2; i++){
	free(H12i[i]);
      }
      free(H12i);

    }



    void Eigen_HH(dcomplex **ac, double *ko, int n, int EVmax)
    {
      /**********************************************************************
Eigen_HH:

Eigen_HH.c is a subroutine to solve a standard eigenvalue problem
with a Hermite complex matrix using Householder method and lapack's
F77_NAME(dstegr,DSTEGR)() or dstedc_().

Log of Eigen_HH.c:

Dec/07/2004  Released by T.Ozaki

       ***********************************************************************/

      static double ABSTOL=1.0e-13;

      static dcomplex **ad,*u,*b1,*p,*q,tmp1,tmp2,u1,u2,p1,ss;
      static double *D,*E,*uu,*alphar,*alphai,
		    s1,s2,s3,r,
		    sum,ar,ai,br,bi,e,
		    a1,a2,a3,a4,a5,a6,b7,r1,r2,
		    r3,x1,x2,xap,
		    bb,bb1,ui,uj,uij;

      static int jj,jj1,jj2,k,ii,ll,i3,i2,j2,
		 i,j,i1,j1,n1,n2,ik,
		 jk,po1,nn,count;

      static double Stime, Etime;
      static double Stime1, Etime1;
      static double Stime2, Etime2;
      static double time1,time2;

      /****************************************************
	allocation of arrays:
       ****************************************************/

      n2 = n + 5;

      ad = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
      for (i=0; i<n2; i++){
	ad[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
      }

      b1 = (dcomplex*)malloc(sizeof(dcomplex)*n2);
      u = (dcomplex*)malloc(sizeof(dcomplex)*n2);
      uu = (double*)malloc(sizeof(double)*n2);
      p = (dcomplex*)malloc(sizeof(dcomplex)*n2);
      q = (dcomplex*)malloc(sizeof(dcomplex)*n2);

      D = (double*)malloc(sizeof(double)*n2);
      E = (double*)malloc(sizeof(double)*n2);

      alphar = (double*)malloc(sizeof(double)*n2);
      alphai = (double*)malloc(sizeof(double)*n2);

      for (i=1; i<=(n+2); i++){
	uu[i] = 0.0;
      }

      if (measure_time==1) printf("size n=%3d EVmax=%2d\n",n,EVmax);
      if (measure_time==1) dtime(&Stime);

      /****************************************************
	Householder transformation
       ****************************************************/

      for (i=1; i<=(n-1); i++){

	s1 = ac[i+1][i].r * ac[i+1][i].r + ac[i+1][i].i * ac[i+1][i].i;
	s2 = 0.0;

	u[i+1].r = ac[i+1][i].r;
	u[i+1].i = ac[i+1][i].i;

	for (i1=i+2; i1<=n; i1++){

	  tmp1.r = ac[i1][i].r; 
	  tmp1.i = ac[i1][i].i; 

	  s2 += tmp1.r*tmp1.r + tmp1.i*tmp1.i;

	  u[i1].r = tmp1.r;
	  u[i1].i = tmp1.i;
	}

	s3 = fabs(s1 + s2);

	if ( ABSTOL<(fabs(ac[i+1][i].r)+fabs(ac[i+1][i].i)) ){
	  if (ac[i+1][i].r<0.0)  s3 =  sqrt(s3);
	  else                   s3 = -sqrt(s3);
	}
	else{
	  s3 = sqrt(s3);
	}

	if ( ABSTOL<fabs(s2) || i==(n-1) ){

	  ss.r = ac[i+1][i].r;
	  ss.i = ac[i+1][i].i;

	  ac[i+1][i].r = s3;
	  ac[i+1][i].i = 0.0;
	  ac[i][i+1].r = s3;
	  ac[i][i+1].i = 0.0;

	  u[i+1].r = u[i+1].r - s3;
	  u[i+1].i = u[i+1].i;

	  u1.r = s3 * s3 - ss.r * s3;
	  u1.i =         - ss.i * s3;
	  u2.r = 2.0 * u1.r;
	  u2.i = 2.0 * u1.i;

	  e = u2.r/(u1.r*u1.r + u1.i*u1.i);
	  ar = e*u1.r;
	  ai = e*u1.i;

	  /* store alpha */
	  alphar[i] = ar;
	  alphai[i] = ai;

	  /* store u2 */
	  uu[i] = u2.r;

	  /* store the first component of u */
	  b1[i].r = ss.r - s3;
	  b1[i].i = ss.i;

	  r = 0.0;
	  for (i1=i+1; i1<=n; i1++){

	    p1.r = 0.0;
	    p1.i = 0.0;
	    for (j=i+1; j<=n; j++){
	      p1.r += ac[i1][j].r * u[j].r - ac[i1][j].i * u[j].i;
	      p1.i += ac[i1][j].r * u[j].i + ac[i1][j].i * u[j].r;
	    }
	    p[i1].r = p1.r / u1.r;
	    p[i1].i = p1.i / u1.r;

	    r += u[i1].r * p[i1].r + u[i1].i * p[i1].i;
	  }

	  r = 0.5*r / u2.r;

	  br =  ar*r;
	  bi = -ai*r;

	  for (i1=i+1; i1<=n; i1++){
	    tmp1.r = 0.5*(p[i1].r - (br * u[i1].r - bi*u[i1].i));
	    tmp1.i = 0.5*(p[i1].i - (br * u[i1].i + bi*u[i1].r));
	    q[i1].r = ar * tmp1.r - ai * tmp1.i; 
	    q[i1].i = ar * tmp1.i + ai * tmp1.r; 
	  }

	  for (i1=i+1; i1<=n; i1++){
	    tmp1.r = u[i1].r;
	    tmp1.i = u[i1].i;
	    tmp2.r = q[i1].r; 
	    tmp2.i = q[i1].i; 
	    for (j1=i+1; j1<=n; j1++){
	      ac[i1][j1].r -= ( tmp1.r * q[j1].r + tmp1.i * q[j1].i
		  +tmp2.r * u[j1].r + tmp2.i * u[j1].i );
	      ac[i1][j1].i -= (-tmp1.r * q[j1].i + tmp1.i * q[j1].r
		  -tmp2.r * u[j1].i + tmp2.i * u[j1].r );
	    }
	  }
	}
      }

      for (i=1; i<=n; i++){
	for (j=1; j<=n; j++){
	  ad[i][j].r = ac[i][j].r;
	  ad[i][j].i = ac[i][j].i;
	}
      }

      if (measure_time==1){
	dtime(&Etime);
	printf("T1   %15.12f\n",Etime-Stime);
      }

      /****************************************************
	call a lapack routine
       ****************************************************/

      if (measure_time==1) dtime(&Stime);

      for (i=1; i<=n; i++){
	D[i-1] = ad[i][i  ].r;
	E[i-1] = ad[i][i+1].r;
      }

      /*
	 if      (dste_flag==0) lapack_dstegr2(n,D,E,ko,ac);
	 else if (dste_flag==1) lapack_dstedc2(n,D,E,ko,ac);
	 else if (dste_flag==2) lapack_dstevx2(n,D,E,ko,ac);
	 */

      lapack_dstevx2(n,D,E,ko,ac);

      if (measure_time==1){
	dtime(&Etime);
	printf("T2   %15.12f\n",Etime-Stime);
      }

      /****************************************************
	transformation of eigenvectors to original space
       ****************************************************/

      if (measure_time==1) dtime(&Stime);

      /* ad stores u */
      for (i=2; i<=n; i++){
	ad[i-1][i].r = b1[i-1].r;
	ad[i-1][i].i =-b1[i-1].i;
	ad[i][i-1].r = b1[i-1].r;
	ad[i][i-1].i = b1[i-1].i;
      }

      for (k=1; k<=EVmax; k++){

	for (nn=1; nn<=n-1; nn++){

	  if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){

	    tmp1.r = 0.0;
	    tmp1.i = 0.0;

	    for (i=n-nn+1; i<=n; i++){
	      tmp1.r += ad[n-nn][i].r * ac[k][i].r - ad[n-nn][i].i * ac[k][i].i;
	      tmp1.i += ad[n-nn][i].i * ac[k][i].r + ad[n-nn][i].r * ac[k][i].i;
	    }

	    ss.r = (alphar[n-nn]*tmp1.r - alphai[n-nn]*tmp1.i) / uu[n-nn];
	    ss.i = (alphar[n-nn]*tmp1.i + alphai[n-nn]*tmp1.r) / uu[n-nn];

	    for (i=n-nn+1; i<=n; i++){
	      ac[k][i].r -= ss.r * ad[n-nn][i].r + ss.i * ad[n-nn][i].i;
	      ac[k][i].i -=-ss.r * ad[n-nn][i].i + ss.i * ad[n-nn][i].r;
	    }
	  }
	}
      }

      if (measure_time==1){
	dtime(&Etime);
	printf("T4   %15.12f\n",Etime-Stime);
      }

      /****************************************************
	normalization
       ****************************************************/

      if (measure_time==1) dtime(&Stime);

      for (j=1; j<=EVmax; j++){
	sum = 0.0;
	for (i=1; i<=n; i++){
	  sum += ac[j][i].r * ac[j][i].r + ac[j][i].i * ac[j][i].i;
	}
	sum = 1.0/sqrt(sum);
	for (i=1; i<=n; i++){
	  ac[j][i].r = ac[j][i].r * sum;
	  ac[j][i].i = ac[j][i].i * sum;
	}
      }

      if (measure_time==1){
	dtime(&Etime);
	printf("T5   %15.12f\n",Etime-Stime);
      }

      /****************************************************
	transpose ac
       ****************************************************/

      for (i=1; i<=n; i++){
	for (j=(i+1); j<=n; j++){
	  tmp1 = ac[i][j];
	  tmp2 = ac[j][i];
	  ac[i][j] = tmp2;
	  ac[j][i] = tmp1;
	}
      }

      /****************************************************
	freeing of arrays:
       ****************************************************/

      for (i=0; i<n2; i++){
	free(ad[i]);
      }
      free(ad);

      free(b1);
      free(u);
      free(uu);
      free(p);
      free(q);
      free(D);
      free(E);
      free(alphar);
      free(alphai);
    }



    void lapack_dstevx2(INTEGER N, double *D, double *E, double *W, dcomplex **ev)
    {
      static int i,j;

      char  *JOBZ="V";
      char  *RANGE="A";

      double VL,VU; /* dummy */
      INTEGER IL,IU; 
      double ABSTOL=1.0e-12;
      INTEGER M;
      double *Z;
      INTEGER LDZ;
      double *WORK;
      INTEGER *IWORK;
      INTEGER *IFAIL;
      INTEGER INFO;

      M = N;
      LDZ = N;

      Z = (double*)malloc(sizeof(double)*LDZ*N);
      WORK = (double*)malloc(sizeof(double)*5*N);
      IWORK = (INTEGER*)malloc(sizeof(INTEGER)*5*N);
      IFAIL = (INTEGER*)malloc(sizeof(INTEGER)*N);

      F77_NAME(dstevx,DSTEVX)( JOBZ, RANGE, &N,  D, E, &VL, &VU, &IL, &IU, &ABSTOL,
	  &M, W, Z, &LDZ, WORK, IWORK, IFAIL, &INFO );

      /* store eigenvectors */

      for (i=0; i<N; i++) {
	for (j=0; j<N; j++) {
	  ev[i+1][j+1].r = Z[i*N+j];
	  ev[i+1][j+1].i = 0.0;
	}
      }

      /* shift ko by 1 */
      for (i=N; i>=1; i--){
	W[i]= W[i-1];
      }

      if (INFO>0) {
	printf("\n error in dstevx_, info=%d\n\n",INFO);
      }
      if (INFO<0) {
	printf("info=%d in dstevx_\n",INFO);
	exit(0);
      }

      free(Z);
      free(WORK);
      free(IWORK);
      free(IFAIL);
    }



    void determinant( int spin, int N, dcomplex **a, INTEGER *ipiv, dcomplex *a1d,
	dcomplex *work, dcomplex Cdet[2] )
    {
      /********************************************************************
	void determinant( int spin, int N, dcomplex **a, INTEGER *ipiv,
	dcomplex *a1d,
	dcomplex *work, dcomplex Cdet[2] )

	a routine for calculating the determinant of overlap matrix
	for occupied states at k1- and k2-points.

	int spin (input variable) 

	collinear spin-unpolarized:    spin = 0
	collinear spin-polarized:      spin = 0 or 1
	non-collinear spin-polarized:  spin = 0

	int N (input variable)

	the number of occupied states

	dcomplex **a (input variable)

	overlap matrix whose size is [fsize3][fsize3].

	INTEGER *ipiv (work space)

	work space for a lapack routine, zgetrf, 
	whose size is fsize3. 

	dcomplex *a1d (work space)

	work space for a lapack routine, zgetrf,
	whose size is fsize3*fsize3.

	dcomplex *work (work space)

	work space for a lapack routine, zgetrf, 
	whose size is fsize3. 

	dcomplex Cdet[2] (output variable)   

	the determinant of the matrix 'a' for occupied states
	at k1- and k2-points. Cdet[0] and Cdet[1] correspond
	to those with spin index 0 and 1.
       ********************************************************************/

      static int i,j;
      static INTEGER lda,info,lwork;
      dcomplex Ctmp;

      lda = N;
      lwork = N;

      /****************************************************
	a -> a1d
       ****************************************************/

      for (i=0;i<N;i++) {
	for (j=0;j<N;j++) {
	  a1d[j*N+i] = a[i+1][j+1];
	}
      }

      /****************************************************
	call zgetrf_() in clapack
       ****************************************************/

      F77_NAME(zgetrf,ZGETRF)(&N, &N, a1d, &lda, ipiv, &info);
      if (info!=0){
	printf("ERROR in zgetrf_(), info=%2d\n",info);
      }

      /*
	 for (i=0;i<=N;i++) {
	 ipiv[i] = i + 1; 
	 }
	 LU_fact(N,a);

	 for (i=0; i<N; i++){
	 a1d[i*N+i] = a[i+1][i+1];
	 }
	 */

      /****************************************************
	a1d -> a
       ****************************************************/

      /*
	 printf("Re \n");
	 for (i=0;i<N;i++) {
	 for (j=0;j<N;j++) {
	 printf("%15.12f ",a1d[j*N+i].r);
	 }
	 printf("\n");
	 }

	 printf("Im\n");
	 for (i=0;i<N;i++) {
	 for (j=0;j<N;j++) {
	 printf("%15.12f ",a1d[j*N+i].i);
	 }
	 printf("\n");
	 }
	 */

      /*
	 for (i=0;i<=N;i++) {
	 printf("i=%2d ipiv[i]=%3d\n",i,ipiv[i]); 
	 }
	 */

      Cdet[spin].r = 1.0;
      Cdet[spin].i = 0.0;
      for (i=0; i<N; i++){

	Ctmp = Cdet[spin];
	if ( ipiv[i]!=(i+1) ){ 
	  Ctmp.r = -Ctmp.r;
	  Ctmp.i = -Ctmp.i;
	} 

	Cdet[spin].r = Ctmp.r*a1d[i*N+i].r - Ctmp.i*a1d[i*N+i].i;
	Cdet[spin].i = Ctmp.r*a1d[i*N+i].i + Ctmp.i*a1d[i*N+i].r;
      }
    }





    void dtime(double *t)
    {
      /* real time */
      struct timeval timev;
      gettimeofday(&timev, NULL);
      *t = timev.tv_sec + (double)timev.tv_usec*1e-6;

      /* user time + system time */
      /*
	 float tarray[2];
	 clock_t times(), wall;
	 struct tms tbuf;
	 wall = times(&tbuf);
	 tarray[0] = (float) (tbuf.tms_utime / (float)CLOCKS_PER_SEC);
	 tarray[1] = (float) (tbuf.tms_stime / (float)CLOCKS_PER_SEC);
       *t = (double) (tarray[0]+tarray[1]);
       printf("dtime: %lf\n",*t);
       */
    }







    void LU_fact(int n, dcomplex **a)
    {
      /*    LU factorization */
      int i,j,k;
      dcomplex w,sum;

      for (k=1; k<=n-1; k++){
	w = RCdiv(1.0,a[k][k]);
	for (i=k+1; i<=n; i++){
	  a[i][k] = Cmul(w,a[i][k]);
	  for (j=k+1; j<=n; j++){
	    a[i][j] = Csub(a[i][j], Cmul(a[i][k],a[k][j]));
	  }
	}
      }

    }




    dcomplex RCdiv(double x, dcomplex a)
    {
      dcomplex c;
      double xx,yy,w;
      xx = a.r;
      yy = a.i;
      w = xx*xx+yy*yy;
      c.r = x*a.r/w;
      c.i = -x*a.i/w;
      return c;
    }


    dcomplex Csub(dcomplex a, dcomplex b)
    {
      dcomplex c;
      c.r = a.r - b.r;
      c.i = a.i - b.i;
      return c;
    }

    dcomplex Cmul(dcomplex a, dcomplex b)
    {
      dcomplex c;
      c.r = a.r*b.r - a.i*b.i;
      c.i = a.i*b.r + a.r*b.i;
      return c;
    }

    void Cross_Product(double a[4], double b[4], double c[4])
    {
      c[1] = a[2]*b[3] - a[3]*b[2]; 
      c[2] = a[3]*b[1] - a[1]*b[3]; 
      c[3] = a[1]*b[2] - a[2]*b[1];
    }

    double Dot_Product(double a[4], double b[4])
    {
      static double sum;
      sum = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; 
      return sum;
    }
