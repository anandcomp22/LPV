#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    int m, n, p;

    cout << "Enter rows and cols of Matrix A: ";
    cin >> m >> n;

    cout << "Enter cols of Matrix B: ";
    cin >> p;

    int A[m][n], B[n][p], C[m][p]; 

    cout << "Enter Matrix A:\n";
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            cin >> A[i][j];

    cout << "Enter Matrix B:\n";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            cin >> B[i][j];

    double start = omp_get_wtime();

    // Parallel multiplication
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end = omp_get_wtime();

    cout << "Result Matrix:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Execution Time: " << (end - start) << " seconds\n";

    return 0;
}