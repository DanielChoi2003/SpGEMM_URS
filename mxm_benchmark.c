#include <GraphBLAS.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


GrB_Matrix load_coo_csv(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("open"); exit(1); }

    GrB_Index i, j;
    double v;
    size_t nvals = 0;
    GrB_Index max_i = 0, max_j = 0;

    char line[256];
    while(fgets(line, sizeof(line), f)){
        int parsed = sscanf(line, "%lu,%lu,%lf", &i, &j, &v);
        if (i > max_i) max_i = i;
        if (j > max_j) max_j = j;
        nvals++;
    }
    while (fscanf(f, "%lu,%lu", &i, &j) == 2) {
    
    }

    // allocate arrays
    GrB_Index *rows = malloc(nvals * sizeof(GrB_Index));
    GrB_Index *cols = malloc(nvals * sizeof(GrB_Index));
    double    *values = malloc(nvals * sizeof(double));

    // ---------- Second pass: load arrays ----------
    rewind(f);
    size_t k = 0;
    while (fgets(line, sizeof(line), f)) {
        int parsed = sscanf(line, "%lu,%lu,%lf", &i, &j, &v);
        rows[k]   = i;
        cols[k]   = j;
        values[k] = (parsed == 3) ? v : 1.0;
        k++;
    }
    fclose(f);

    // ---------- Build GraphBLAS matrix ----------
    GrB_Matrix A;
    GrB_Index nrows = max_i + 1;
    GrB_Index ncols = max_j + 1;

    printf("max row: %d and max col: %d\n", nrows, ncols);

    GrB_Matrix_new(&A, GrB_FP64, nrows, ncols);

    // Combine duplicates with addition
    GrB_Matrix_build_FP64(A, rows, cols, values, nvals, GrB_PLUS_FP64);

    free(rows);
    free(cols);
    free(values);

    return A;
}

void write_matrix_csv(const char *path, GrB_Matrix C) {

    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, C);

    GrB_Index *rows = malloc(nvals * sizeof(GrB_Index));
    GrB_Index *cols = malloc(nvals * sizeof(GrB_Index));
    double    *vals = malloc(nvals * sizeof(double));

    GrB_Matrix_extractTuples_FP64(rows, cols, vals, &nvals, C);

    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); exit(1); }

    for (GrB_Index k = 0; k < nvals; k++) {
        fprintf(f, "%lu,%lu,%.17g\n",
                (unsigned long)(rows[k]),
                (unsigned long)(cols[k]),
                vals[k]);
    }

    fclose(f);
    free(rows);
    free(cols);
    free(vals);
}
int main() {
    /*
        include the transpose and wait time in the research paper, or at least
        include the total time and explain how much of it belongs to transpose and wait time (unjumbling / sorting)
    
    */
    printf("Threads: %d\n", omp_get_max_threads());
    GrB_init(GrB_NONBLOCKING);
    GxB_Global_Option_set(GxB_BURBLE, true);
    int nthreads ;
    //GrB_set (GrB_GLOBAL,  224, GxB_NTHREADS) ;
    GrB_get (GrB_GLOBAL, &nthreads, GxB_NTHREADS) ;
    printf ("# of threads default: %d\n", nthreads);

    GrB_Matrix A, B, C;

    A = load_coo_csv("/p/lustre1/choi26/rmat/scale20_ef16_nonsymmetric.csv");
    //B = load_coo_csv("/usr/workspace/choi26/uni_scale_24.csv");
    GrB_Matrix_dup(&B, A);
    GrB_Matrix_wait(A, GrB_MATERIALIZE);
    GrB_Matrix_wait(B, GrB_MATERIALIZE);

    GrB_Index A_rows, A_ncols, B_nrows, B_ncols ;
    GrB_Matrix_nrows(&B_nrows, B);
    GrB_Matrix_ncols(&B_ncols, B);
    GrB_Matrix_nrows(&A_rows, A) ;
    GrB_Matrix_new(&C, GrB_FP64, A_rows, A_rows);

    GrB_Semiring semiring = GxB_PLUS_TIMES_FP64;
    GrB_Descriptor desc;
    GrB_Descriptor_new(&desc);
    GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
    
    double t0 = omp_get_wtime ( ) ;
    GrB_Matrix BT ;
    GrB_Matrix_new (&BT, GrB_FP64, B_ncols, B_nrows);
    GrB_transpose (BT, NULL, NULL, B, NULL);
    t0 = omp_get_wtime ( ) - t0 ;
    // GxB_print (BT, 2);
    // printf ("transpose time: %g\n", t0);
    //GxB_print (A, 2); 
    //GxB_print (C, 2);

    double t1 = omp_get_wtime();

    GrB_Info info = GrB_mxm(C, NULL, NULL, semiring, A, BT, NULL);
    //printf ("info: %d\n", info);

    //GxB_print(C,2) ;
    //printf ("\nnow doing the wait:\n");
    GrB_Matrix_wait(C, GrB_MATERIALIZE);

    printf("mxm time: %.6f seconds\n", omp_get_wtime() - t1);
    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, C);
    printf("size: %llu\n", nvals);

    // GxB_print (BT, 2); 

    // WRITE THE RESULT TO CSV
    // write_matrix_csv("/g/g14/choi26/graphBLAS_sandbox/rmat17_ef16_graphblas.csv", C);
    // GrB_finalize();
    return 0;
}